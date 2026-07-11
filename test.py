import os
import sys
import torch


def load_state_dict_from_best(best_path):
    # Try full unpickle first (Ultralytics checkpoints often store model objects)
    try:
        ckpt = torch.load(best_path, map_location='cpu', weights_only=False)
    except TypeError:
        ckpt = torch.load(best_path, map_location='cpu')

    if isinstance(ckpt, dict):
        # Prefer EMA weights if present
        if 'ema' in ckpt and hasattr(ckpt['ema'], 'state_dict'):
            return ckpt['ema'].state_dict()
        if 'model' in ckpt and hasattr(ckpt['model'], 'state_dict'):
            return ckpt['model'].state_dict()
        if 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
            return ckpt['state_dict']

    if hasattr(ckpt, 'state_dict'):
        return ckpt.state_dict()

    # Fallback: try weights_only load for pure state_dict checkpoints
    try:
        ckpt_wo = torch.load(best_path, map_location='cpu', weights_only=True)
        if isinstance(ckpt_wo, dict):
            return ckpt_wo
    except Exception:
        pass

    raise RuntimeError('Failed to extract state_dict from checkpoint')


def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    os.environ.setdefault("CEASC_DEBUG_BBOX", "1")

    weight_path = os.path.join('runs', 'train', 'mymodel_coco128_103', 'weights', 'best.pt')
    img_path = 'test.jpg'

    if not os.path.exists(weight_path):
        raise FileNotFoundError(f'weights not found: {weight_path}')
    if not os.path.exists(img_path):
        raise FileNotFoundError(f'image not found: {img_path}')

    from pram.tasks import MyModel
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.ops import scale_coords
    from ultralytics.utils.plotting import Annotator

    # Build model and load state_dict manually
    model = MyModel(cfg='pram/cfg/model_0.yaml', verbose=False)
    state_dict = load_state_dict_from_best(weight_path)
    load_ret = model.load_state_dict(state_dict, strict=False)
    try:
        missing = list(load_ret.missing_keys)
        unexpected = list(load_ret.unexpected_keys)
        print(f'Load state_dict: missing={len(missing)}, unexpected={len(unexpected)}')
        if len(missing) > 0:
            print(f'  missing sample: {missing[:5]}')
        if len(unexpected) > 0:
            print(f'  unexpected sample: {unexpected[:5]}')
    except Exception:
        pass
    model.eval()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Load image using ultralytics helper
    from ultralytics.data.augment import LetterBox
    import cv2
    import numpy as np

    im0 = cv2.imread(img_path)
    if im0 is None:
        raise FileNotFoundError(f'failed to read image: {img_path}')

    # Preprocess
    lb = LetterBox(new_shape=320, auto=False, stride=32)
    im = lb(image=im0)
    im = im.transpose((2, 0, 1))[::-1]  # BGR -> RGB, HWC -> CHW
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).float() / 255.0
    im = im.unsqueeze(0).to(device)

    # Forward
    with torch.no_grad():
        preds = model(im)

    # Convert to NMS input [B, N, 6] if needed
    if isinstance(preds, (list, tuple)):
        # assume CEASC style outputs -> use inference_batch if available
        if hasattr(model.model[-1], 'inference_batch'):
            preds = model.model[-1].inference_batch(
                preds[0], preds[1], preds[2], img_shape=im.shape[-2:]
            )
        else:
            raise RuntimeError('Model output is a list/tuple but inference_batch not found')

    # Debug stats for predictions before NMS
    if isinstance(preds, torch.Tensor) and preds.ndim == 3 and preds.shape[-1] >= 6:
        confs = preds[..., 4]
        print(f'Preds shape: {tuple(preds.shape)}')
        if confs.numel() == 0:
            print('Conf stats: empty (no candidates)')
        else:
            print(f'Conf stats: min={confs.min().item():.6f}, max={confs.max().item():.6f}, mean={confs.mean().item():.6f}')
        cls_ids = preds[..., 5]
        if cls_ids.numel() == 0:
            print('Cls id stats: empty (no candidates)')
        else:
            print(f'Cls id stats: min={cls_ids.min().item():.1f}, max={cls_ids.max().item():.1f}')
    else:
        print(f'Preds type/shape not recognized for debug: {type(preds)}')

    # Sweep thresholds and save results under predict with threshold tags
    conf_list = [0.05, 0.1, 0.2, 0.25]
    nms_list = [0.5]
    max_det_list = [300]

    for conf_thres in conf_list:
        for iou_thres in nms_list:
            for max_det in max_det_list:
                det = non_max_suppression(preds, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det)[0]
                kept = 0 if det is None else len(det)
                print(f'NMS kept (conf={conf_thres}, nms={iou_thres}, max_det={max_det}): {kept}')

                save_dir = os.path.join(
                    'runs', 'predict', 'test_best_state_dict',
                    f'conf_{conf_thres}', f'nms_{iou_thres}', f'maxdet_{max_det}'
                )
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, 'test.jpg')

                annotator = Annotator(im0.copy(), line_width=2)
                if det is not None and len(det):
                    # map boxes back to original image size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                    det = det.cpu().numpy()
                    for *xyxy, conf, cls in det:
                        label = f'{int(cls)} {conf:.2f}'
                        annotator.box_label(xyxy, label, color=(0, 255, 0))

                result_img = annotator.result()
                cv2.imwrite(save_path, result_img)
                print(f'Result saved to: {save_path}')


if __name__ == '__main__':
    main()

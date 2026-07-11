import os
import sys
import shutil
import glob
import torch


def prepare_subset(num_images=10):
    src_img_dir = os.path.join('datasets', 'coco128', 'images', 'train2017')
    src_lbl_dir = os.path.join('datasets', 'coco128', 'labels', 'train2017')
    dst_root = os.path.join('datasets', 'coco128_10')
    dst_img_dir = os.path.join(dst_root, 'images', 'train2017')
    dst_lbl_dir = os.path.join(dst_root, 'labels', 'train2017')

    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)

    imgs = sorted(glob.glob(os.path.join(src_img_dir, '*.jpg')))
    if len(imgs) == 0:
        raise FileNotFoundError(f'No images found in {src_img_dir}')

    pick = imgs[:num_images]
    for img_path in pick:
        name = os.path.basename(img_path)
        stem = os.path.splitext(name)[0]
        lbl_path = os.path.join(src_lbl_dir, stem + '.txt')

        shutil.copy2(img_path, os.path.join(dst_img_dir, name))
        if os.path.exists(lbl_path):
            shutil.copy2(lbl_path, os.path.join(dst_lbl_dir, stem + '.txt'))
        else:
            # Ensure empty label file exists for images without labels
            open(os.path.join(dst_lbl_dir, stem + '.txt'), 'w', encoding='utf-8').close()

    # Write dataset yaml
    yaml_path = os.path.join('datasets', 'coco128_10.yaml')
    yaml_text = (
        "path: datasets/coco128_10\n"
        "train: images/train2017\n"
        "val: images/train2017\n"
        "names: [\n"
        "  person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light,\n"
        "  fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow,\n"
        "  elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee,\n"
        "  skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard,\n"
        "  tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple,\n"
        "  sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch,\n"
        "  potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone,\n"
        "  microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear,\n"
        "  hair drier, toothbrush\n"
        "]\n"
    )
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_text)

    return yaml_path


def register_custom_modules():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)

    from ultralytics import YOLO
    from module.model_registrar import MyModelPredictor, MyModelTrainer, MyModelValidator
    from module.new_block import SPD_SCConv, DySample, SimAM_C3k2, BiFPN, Sequential_BiFPN
    from module.Head import CEASC

    import ultralytics.nn.modules as _ul_modules
    import ultralytics.nn.tasks as _ul_tasks
    import pram.tasks as _pram_tasks
    from pram.tasks import MyModel

    _custom_classes = (SPD_SCConv, DySample, SimAM_C3k2, BiFPN, Sequential_BiFPN, CEASC)
    for _cls in _custom_classes:
        name = _cls.__name__
        setattr(_ul_modules, name, _cls)
        setattr(_ul_tasks, name, _cls)

    _ul_tasks.BaseModel = _pram_tasks.BaseModel
    _ul_tasks.DetectionModel = _pram_tasks.DetectionModel

    original_init = YOLO.__init__

    def new_init(self, model='yolo11n.pt', task=None, verbose=False):
        if isinstance(model, str) and model.endswith('.yaml') and 'model_' in model:
            self.ckpt = None
            self.cfg = model
            self.task = 'detect'
            self.ModelClass = MyModel
            self.TrainerClass = MyModelTrainer
            self.ValidatorClass = MyModelValidator
            self.PredictorClass = MyModelPredictor

            torch.nn.Module.__init__(self)
            self.session = None
            self.callbacks = []

            self.model = MyModel(cfg=self.cfg, verbose=verbose)
            self.current_model = model
        else:
            original_init(self, model, task, verbose)

    YOLO.__init__ = new_init
    return YOLO


def main():
    yaml_path = prepare_subset(num_images=10)

    if torch.cuda.is_available():
        device = 0
    else:
        device = 'cpu'

    YOLO = register_custom_modules()

    model = YOLO('pram/cfg/model_0.yaml')

    # Custom train flow to ensure MyModelTrainer is used (avoid ultralytics default parse_model)
    from module.model_registrar import MyModelTrainer
    model.TrainerClass = MyModelTrainer
    model.cfg = 'pram/cfg/model_0.yaml'

    train_args = dict(
        data=yaml_path,
        epochs=100,
        imgsz=320,
        batch=1,
        device=device,
        workers=0,
        project='runs/train',
        name='mymodel_coco128_10',
        save_period=10,
        patience=50,
        half=False,
        amp=False,
        model=model.cfg,
    )

    model.trainer = model.TrainerClass(overrides=train_args)
    from ultralytics.utils import RANK
    model.trainer.model = model.trainer.get_model(weights=None, cfg=model.cfg, verbose=RANK == -1)
    model.trainer.train()


if __name__ == '__main__':
    main()

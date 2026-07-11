"""数据集工具函数。"""

import os
import shutil
import glob


def prepare_subset(src_img_dir="datasets/coco128/images/train2017",
                   src_lbl_dir="datasets/coco128/labels/train2017",
                   dst_root="datasets/coco128_10",
                   num_images=10):
    """从 COCO128 复制前 N 张图片创建子集，返回生成的 yaml 路径。"""
    dst_img_dir = os.path.join(dst_root, "images", "train2017")
    dst_lbl_dir = os.path.join(dst_root, "labels", "train2017")
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)

    imgs = sorted(glob.glob(os.path.join(src_img_dir, "*.jpg")))
    if not imgs:
        raise FileNotFoundError(f"图片目录为空: {src_img_dir}")

    for img_path in imgs[:num_images]:
        name = os.path.basename(img_path)
        stem = os.path.splitext(name)[0]
        lbl_path = os.path.join(src_lbl_dir, stem + ".txt")
        shutil.copy2(img_path, os.path.join(dst_img_dir, name))
        if os.path.exists(lbl_path):
            shutil.copy2(lbl_path, os.path.join(dst_lbl_dir, stem + ".txt"))
        else:
            open(os.path.join(dst_lbl_dir, stem + ".txt"), "w", encoding="utf-8").close()

    yaml_path = os.path.join("datasets", "coco128_10.yaml")
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
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_text)
    return yaml_path

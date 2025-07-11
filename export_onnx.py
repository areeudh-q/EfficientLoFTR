import os
from copy import deepcopy
import torch
import cv2
import numpy as np
import matplotlib.cm as cm
from src.utils.plotting import make_matching_figure
from src.loftr import LoFTR, full_default_cfg, opt_default_cfg, reparameter

model_type = 'full'  # 'full' 表示最佳质量，'opt' 表示最佳效率
precision = 'fp32'  # 如果您拥有现代 GPU（推荐 NVIDIA 架构 >= SM_70），则可以通过混合精度 (MP) / FP16 计算享受近乎无损的精度。

if model_type == 'full':
    _default_cfg = deepcopy(full_default_cfg)
elif model_type == 'opt':
    _default_cfg = deepcopy(opt_default_cfg)

if precision == 'mp':
    _default_cfg['mp'] = True
elif precision == 'fp16':
    _default_cfg['half'] = True

print(_default_cfg)
matcher = LoFTR(config=_default_cfg)

matcher.load_state_dict(
    torch.load("./weights/eloftr_outdoor.ckpt", map_location=torch.device('cpu'), weights_only=False)['state_dict'])
matcher = reparameter(matcher)  # 如果不进行重新参数化，模型性能会下降

if precision == 'fp16':
    matcher = matcher.half()

if torch.cuda.is_available():
    matcher = matcher.eval().cuda()
else:
    matcher = matcher.eval()

img0_pth = "./images/c1.jpg"
img1_pth = "./images/c2.jpg"
img0_raw = cv2.imread(img0_pth, cv2.IMREAD_GRAYSCALE)
img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)
# 输入大小应该能被 32
img0_raw = cv2.resize(img0_raw, (512, 512))
img1_raw = cv2.resize(img1_raw, (512, 512))

if precision == 'fp16':
    if torch.cuda.is_available():
        img0 = torch.from_numpy(img0_raw)[None][None].half().cuda() / 255.
        img1 = torch.from_numpy(img1_raw)[None][None].half().cuda() / 255.
    else:
        img0 = torch.from_numpy(img0_raw)[None][None].half() / 255.
        img1 = torch.from_numpy(img1_raw)[None][None].half() / 255.
else:
    if torch.cuda.is_available():
        img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
        img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
    else:
        img0 = torch.from_numpy(img0_raw)[None][None] / 255.
        img1 = torch.from_numpy(img1_raw)[None][None] / 255.

eloftr_path = "./eloftr.onnx"

dynamic_axes = {
    "image0": {2: "height0", 3: "width0"},
    "image1": {2: "height1", 3: "width1"},
    "mkpts0": {0: "points_0"},
    "mkpts1": {0: "points_1"},
    "mconf": {0: "nums"}
}

print("Exporting LoFTR to ONNX...")

print(img0[None].shape)

torch.onnx.export(
    matcher,
    (img0, img1),
    eloftr_path,
    input_names=["image0", "image1"],
    output_names=["mkpts0", "mkpts1", "mconf"],
    opset_version=17,
    dynamic_axes=dynamic_axes
)

print("Done!")
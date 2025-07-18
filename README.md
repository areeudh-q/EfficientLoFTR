# add export to onnx

```shell
python export_onnx.py
```

# Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed

### [Project Page](https://zju3dv.github.io/efficientloftr) | [Paper](https://zju3dv.github.io/efficientloftr/files/EfficientLoFTR.pdf) 
<br/>

> Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed  
> [Yifan Wang](https://github.com/wyf2020)<sup>\*</sup>, [Xingyi He](https://github.com/hxy-123)<sup>\*</sup>, [Sida Peng](https://pengsida.net), [Dongli Tan](https://github.com/Cuistiano), [Xiaowei Zhou](http://xzhou.me)  
> CVPR 2024 Highlight

https://github.com/zju3dv/EfficientLoFTR/assets/69951260/40890d21-180e-4e70-aeba-219178b0d824

## 🌟News🌟
To enhance multi-modality matching with EfficientLoFTR and improve its applicability to UAV localization, autonomous driving, and beyond, check out our latest work, [MatchAnything](https://github.com/zju3dv/MatchAnything)! Try our demo and see it in action!
## TODO List
- [x] Inference code and pretrained models
- [x] Code for reproducing the test-set results
- [ ] Add options of flash-attention and torch.compiler for better performance
- [x] jupyter notebook demo for matching a pair of images
- [x] Training code

## Installation
```shell
conda env create -f environment.yaml
conda activate eloftr
pip install torch==2.0.0+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt 
```
The test and training can be downloaded by [download link](https://drive.google.com/drive/folders/1DOcOPZb3-5cWxLqn256AhwUVjBPifhuf?usp=sharing) provided by LoFTR

We provide our pretrained model in [download link](https://drive.google.com/drive/folders/1GOw6iVqsB-f1vmG6rNmdCcgwfB4VZ7_Q?usp=sharing)


## Reproduce the testing results with pytorch-lightning
You need to first set up the testing subsets of ScanNet and MegaDepth. We create symlinks from the previously downloaded datasets to `data/{{dataset}}/test`.

```shell
# set up symlinks
ln -s /path/to/scannet-1500-testset/* /path/to/EfficientLoFTR/data/scannet/test
ln -s /path/to/megadepth-1500-testset/* /path/to/EfficientLoFTR/data/megadepth/test
```
### Inference time
```shell
conda activate eloftr
bash scripts/reproduce_test/indoor_full_time.sh
bash scripts/reproduce_test/indoor_opt_time.sh
```

### Accuracy
```shell
conda activate eloftr
bash scripts/reproduce_test/outdoor_full_auc.sh
bash scripts/reproduce_test/outdoor_opt_auc.sh
bash scripts/reproduce_test/indoor_full_auc.sh
bash scripts/reproduce_test/indoor_opt_auc.sh
```

## Training
```shell
conda env create -f environment_training.yaml  # used a different version of pytorch, maybe slightly different from the inference environment
pip install -r requirements.txt
conda activate eloftr_training
bash scripts/reproduce_train/eloftr_outdoor.sh eloftr_outdoor
```

## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```bibtex
@inproceedings{wang2024eloftr,
  title={{Efficient LoFTR}: Semi-Dense Local Feature Matching with Sparse-Like Speed},
  author={Wang, Yifan and He, Xingyi and Peng, Sida and Tan, Dongli and Zhou, Xiaowei},
  booktitle={CVPR},
  year={2024}
}
```

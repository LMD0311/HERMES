# Data and Weights Preparation

This guide walks you through preparing datasets and pretrained weights required for HERMES.

---

## 1. Prepare nuScenes Dataset

- First, create a `data` directory:
  ```bash
  mkdir -p data
  ```
- Place or symlink your nuScenes dataset to the `data` directory:
  ```bash
  ln -s /path/to/your/nuscenes data/nuscenes
  ```
- Download the following `.pkl` files and `mask_cam_img.jpg` to `data/nuscenes/`:
  - [nuscenes_advanced_12Hz_infos_train.pkl](https://github.com/LMD0311/HERMES/releases/download/data/nuscenes_advanced_12Hz_infos_train.pkl)
  - [nuscenes_masked_only_infos_temporal_train.pkl](https://huggingface.co/LMD0311/HERMES/resolve/main/data/nuscenes_masked_only_infos_temporal_train.pkl)
  - [nuscenes_infos_temporal_train.pkl](https://github.com/LMD0311/HERMES/releases/download/data/nuscenes_infos_temporal_train.pkl)
  - [nuscenes_infos_temporal_val.pkl](https://github.com/LMD0311/HERMES/releases/download/data/nuscenes_infos_temporal_val.pkl)
  - [mask_cam_img.jpg](https://github.com/LMD0311/HERMES/releases/download/data/mask_cam_img.jpg)  (required for Stage2-1 data augmentation)
- We also provide a Baidu Netdisk download [link](https://pan.baidu.com/s/1BdFAI3Cj8mWI3bdf9IT8QA?pwd=w63t) for your convenience.
---

## 2. Prepare Text Annotations

- Download and unzip the following files into the `data` directory:
  - [omnidrive_nusc.zip](https://github.com/LMD0311/HERMES/releases/download/data/omnidrive_nusc.zip)
  - [NuInteractCaption.zip](https://github.com/LMD0311/HERMES/releases/download/data/NuInteractCaption.zip)
- Example:
  ```bash
  unzip omnidrive_nusc.zip -d data/
  unzip NuInteractCaption.zip -d data/
  ```

---

## 3. Prepare Pretrained Weights

### a) Download InternVL-2 Pretraining

```bash
cd projects/mmdet3d_plugin/models/internvl_chat
mkdir pretrained
cd pretrained
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2-2B --local-dir InternVL2-2B
```

### b) Download Project Checkpoints

- Create a `ckpt` directory in your project root, and download the following model weights into it:
  - [open_clip_convnext_base_w-320_laion_aesthetic-s13B-b82k.bin](https://huggingface.co/LMD0311/HERMES/blob/main/ckpt/open_clip_convnext_base_w-320_laion_aesthetic-s13B-b82k.bin)
  - [hermes_stage1.pth](https://huggingface.co/LMD0311/HERMES/blob/main/ckpt/hermes_stage1.pth)
  - [hermes_stage2_1.pth](https://huggingface.co/LMD0311/HERMES/blob/main/ckpt/hermes_stage2_1.pth)
  - [hermes_stage2_2.pth](https://huggingface.co/LMD0311/HERMES/blob/main/ckpt/hermes_stage2_2.pth)
  - [hermes_final.pth](https://huggingface.co/LMD0311/HERMES/blob/main/ckpt/hermes_final.pth)
- We also provide a Baidu Netdisk download [link](https://pan.baidu.com/s/1BdFAI3Cj8mWI3bdf9IT8QA?pwd=w63t) for your convenience.
---

## Directory Structure

Your project directory should look like this after setup:

```
HERMES
├── data
│   ├── nuscenes
│   │   ├── mask_cam_img.jpg
│   │   ├── nuscenes_advanced_12Hz_infos_train.pkl
│   │   ├── nuscenes_masked_only_infos_temporal_train.pkl
│   │   ├── nuscenes_infos_temporal_train.pkl
│   │   └── nuscenes_infos_temporal_val.pkl
│   ├── omnidrive_nusc
│   └── NuInteractCaption
├── ckpt
│   ├── open_clip_convnext_base_w-320_laion_aesthetic-s13B-b82k.bin
│   ├── hermes_stage1.pth
│   ├── hermes_stage2_1.pth
│   ├── hermes_stage2_2.pth
│   └── hermes_final.pth
```

---
Please refer to [Usage.md](./Usage.md) for instructions on HERMES training and evaluation.
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
  
- Download the `.pkl` files and `mask_cam_img.jpg` to `data/nuscenes/`:

  ```bash
  huggingface-cli download LMD0311/HERMES --include="*pkl" --local-dir ./
  mv ./data/*pkl ./data/nuscenes/
  huggingface-cli download LMD0311/HERMES --include="*mask_cam_img.jpg" --local-dir ./
  mv ./data/*jpg ./data/nuscenes/
  ```

---

## 2. Prepare Text Annotations

- Download and unzip the following files into the `data` directory:
  ```bash
  huggingface-cli download LMD0311/HERMES --include="*zip" --local-dir ./
  unzip data/omnidrive_nusc.zip -d data/
  unzip data/NuInteractCaption.zip -d data/
  ```

---

## 3. Prepare Pretrained Weights

### a) Download InternVL-2 Pretraining

```bash
cd projects/mmdet3d_plugin/models/internvl_chat
mkdir pretrained
cd pretrained
huggingface-cli download OpenGVLab/InternVL2-2B --local-dir InternVL2-2B
```

### b) Download Project Checkpoints

- Create a `ckpt` directory in your project root, and download the following model weights into it:
  ```bash
  huggingface-cli download LMD0311/HERMES --include="ckpt/*" --local-dir ./
  ```
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
# Environment Setup

This project uses **CUDA 11.8** and recommends **Conda** for environment management.

Follow this guide to set up your environment for running HERMES.  

---

## 1. Create and Activate Conda Environment

```bash
conda create -n hermes python=3.9 -y
conda activate hermes
pip install --upgrade pip setuptools wheel
```

## 2. Install InternVL and Dependencies

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install mmcv-full==1.5.0
pip install mmdet==2.24.0
pip install -r requirements_internvl.txt
pip install flash-attn==2.3.6 --no-build-isolation
```
If you encounter installation issues with `mmcv-full` on H20, try:  
<code>TORCH_CUDA_ARCH_LIST="9.0" pip install --force-reinstall mmcv-full==1.5.0</code>

## 3. Install MMDet3D and Additional Packages

```bash
pip install einops fvcore seaborn iopath==0.1.9 timm==0.6.13 typing-extensions==4.5.0 pylint ipython==8.12  numpy==1.22 matplotlib==3.5.2 numba==0.57 pandas==1.4.4 scikit-image==0.19.3 setuptools==59.5.0, boto3
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install plyfile==1.0.3, nuscenes-devkit==1.1.10, plotly==5.22.0, pandas==1.4.4, scipy==1.10.1, flake8==7.1.0, pytest==8.2.2, lyft_dataset_sdk, yapf==0.40.1
python setup.py install
```

## 4. Compile Third-Party Libraries

```bash
cd third_lib/chamfer_dist/chamferdist/
pip install .
cd ../../..

cd projects/mmdet3d_plugin/bevformer/backbones/ops_dcnv3
sh make.sh
cd ../../../../..
```

## 5. Install PyG Dependency

```bash
pip install https://data.pyg.org/whl/torch-2.0.0%2Bcu118/torch_scatter-2.1.2%2Bpt20cu118-cp39-cp39-linux_x86_64.whl
```

## 6. Install InternVL-Chat Plugin

```bash
cd projects/mmdet3d_plugin/models/internvl_chat
pip install -e .
pip install numba==0.57, torchmetrics==1.4.1, networkx==2.5
```

---

Please refer to [Data.md](./Data.md) for instructions on handling datasets and pretrained weights.
# Usage Guide

We have tested HERMES on NVIDIA H20 servers and recommend using GPUs with at least 80GB memory for smooth operation.  
We plan to release DeepSpeed deployment support in the future to further reduce GPU memory usage.
> If you are using NVIDIA H20 96GB, you may remove some checkpoint usage in [hermes.py](../projects/mmdet3d_plugin/models/detectors/hermes.py) and [hermes_future_render_head.py](../projects/mmdet3d_plugin/models/dense_heads/hermes_future_render_head.py) for improved efficiency.

---

## 1. Training

### Multi-GPU / Multi-Node Training

```bash
torchrun --nproc_per_node $PROC_PER_NODE \
        --master_addr $MASTER_ADDR \
        --master_port $MASTER_PORT \
        --nnodes $NODE_COUNT \
        --node_rank $NODE_RANK \
        extra_tools/train.py \
        projects/configs/hermes/stage3.py \
        --work-dir ./work-dir \
        --launcher pytorch \
        --no-validate
```

### Single-GPU Debugging

```bash
python extra_tools/train.py projects/configs/hermes/stage3.py --work-dir ./work-dir --no-validate
```
You can modify different config files to train different stages as needed.

### Checkpoint Cleanup (Optional)

Since we use ```mmcv``` for training pipeline management, the saved checkpoints include the frozen LLM pretrained parameters, which can be large.  
You can use our provided script to remove unnecessary parts and save storage space:
```bash
python extra_tools/ckpt_convertor.py ./path/to/your_custom_trained.pth --delete_optimizer --save_path path/to/your_custom_trained_cleaned.pth
```
Both `--delete_optimizer` and `--save_path` are optional.

---

## 2. Inference (Testing)

```bash
torchrun --nproc_per_node $PROC_PER_NODE \
        --master_addr $MASTER_ADDR \
        --master_port $MASTER_PORT \
        --nnodes $NODE_COUNT \
        --node_rank $NODE_RANK \
        extra_tools/test.py \
        projects/configs/hermes/stage3.py \
        ckpt/hermes_final.pth \
        --launcher pytorch
```

- The results for each scenario will be saved under `outputs/stage3/hermes_eval/results`.
- As nuScenes contains a large number of scenes, we recommend using multi-GPU/multi-node inference.

---

## 3. Evaluation

We evaluate the results by reading the saved JSON files and calculating Chamfer Distance and text metrics.

### Install necessary libraries:

```bash
pip install pycocoevalcap nltk openai pyfiglet
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('punkt_tab')"
```

### Run evaluation

```bash
python extra_tools/eval_hermes_results.py ./outputs/stage3/hermes_eval/results
```
Due to the nature of LLMs, inference results may vary slightly each time.

---

For more details and custom configuration, please refer to the [config files](../projects/configs/hermes/).
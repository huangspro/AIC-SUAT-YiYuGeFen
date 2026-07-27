# EoMT Semantic Segmentation Adaptation for AIC

This project is inherited from **EoMT**. I (Haoyu Huang) adapted it for the semantic segmentation task in AIC (AI Competition).

Thanks to the authors of EoMT for releasing their work under the MIT License, which allowed us to build upon this exceptional model as the foundation of our project.

---

# Project Structure

```text
.
├── configs/                                   # Configuration files for EoMT
│   ├── dinov2/
│   │   └── cityscapes/
│   └── dinov3/
│
├── datasets/                                  # Dataset processing
│   ├── cityscapes_semantic.py                 # Modified for our 9-class semantic segmentation task (see Changes section)
│   ├── dataset.py
│   ├── __init__.py
│   ├── lightning_data_module.py
│   └── transforms.py
│
├── main.py                                    # Training / validation / prediction entry
│
├── models/
│   ├── eomt.py
│   ├── scale_block.py
│   └── vit.py
│
├── model_zoo/
│   ├── dinov2.md
│   └── dinov3.md
│
├── myDataset/
│   └── data/
│       ├── gtFine_trainvaltest.zip
│       └── leftImg8bit_trainvaltest.zip
│
└── training/
    ├── lightning_module.py
```

---

# Usage

## 1. Training

Run the following command to start training:

```bash
CUDA_VISIBLE_DEVICES=1 python3 main.py fit \
    -c configs/dinov2/cityscapes/semantic/eomt_large_1024.yaml \
    --trainer.devices 1 \
    --data.batch_size 4 \
    --data.path ./myDataset/data
```

---

## 2. Prediction

Predict images in `img/` and save the segmentation masks to `img_out/`:

```bash
python3 main.py predict \
    -c configs/dinov2/cityscapes/semantic/eomt_large_1024.yaml \
    --model.network.masked_attn_enabled False \
    --trainer.devices 1 \
    --data.batch_size 4 \
    --model.ckpt_path ./eomt/q36nuln3/checkpoints/epoch=49-step=81050.ckpt \
    --data.predict_img_dir ./img \
    --data.predict_save_dir ./img_out
```

---

## 3. Validation (mIoU Evaluation)

Evaluate the model and calculate the mIoU:

```bash
python3 main.py validate \
    -c configs/dinov2/cityscapes/semantic/eomt_large_1024.yaml \
    --model.network.masked_attn_enabled False \
    --trainer.devices 1 \
    --data.batch_size 4 \
    --data.path ./myDataset/data \
    --model.ckpt_path ./eomt/q36nuln3/checkpoints/epoch=49-step=81050.ckpt
```

---

# Modifications

## Dataset

The following modifications were made in `datasets/cityscapes_semantic.py`:

- Adapted the dataset parser for our semantic segmentation task.
- Changed the label mapping from the original Cityscapes categories to **9 classes (0–8)**.
- Added `predict_dataset.py` for loading images during prediction.

---

## Training Module

In `training/lightning_module.py`:

- Added a `predict_step()` method to support the prediction pipeline.

---

# Changes in `cityscapes_semantic.py`

The `target_parser()` function was modified to fit our dataset.

Our dataset contains **9 semantic classes**, indexed from **0 to 8** (including the background class). Labels outside this range are ignored, and the ignore label (`255`) is skipped.

Thanks to **Kimi-2.6 AI LLM** for assisting in reconstructing this method.

```python
@staticmethod
def target_parser(target, **kwargs):
    masks, labels = [], []

    for label_id in target[0].unique():
        # Ignore regions (if any)
        if label_id == 255:
            continue

        # Keep only valid classes (0~8)
        if label_id < 0 or label_id > 8:
            continue

        masks.append(target[0] == label_id)
        labels.append(int(label_id))

    return masks, labels, [False for _ in range(len(masks))]
```
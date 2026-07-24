# EoMT Semantic Segmentation Adaptation for AIC

This is a project inherited from [EoMT](https://github.com/...). I (Haoyu Huang) adapted it to our semantic segmentation task in AIC (AI Competition).

Thanks for the generous authors of EoMT for using MIT license, allowing us to use this exceptional model as our fundamental model.

## Project Structure

├── configs                                             -------This directory defines all the configuraton for EoMT, in which we use CityScapes-1024*1024 version of EoMT.
│   ├── dinov2
│   │   ├── cityscapes
│   └── dinov3
├── datasets                                            -------In this directory the author defines the dataset process methods.
│   ├── cityscapes_semantic.py                          -------We change this file to fit our specific task(9 classes, 0-8), the changing can be found at [1].
│   ├── dataset.py
│   ├── __init__.py
│   ├── lightning_data_module.py
│   └── transforms.py
├── main.py                                             -------The training process are start here
├── models
│   ├── eomt.py
│   ├── scale_block.py
│   └── vit.py
├── model_zoo
│   ├── dinov2.md
│   └── dinov3.md
├── myDataset                                           -------according to the original dataset settinig 
│   └── data
│       ├── gtFine_trainvaltest.zip
│       └── leftImg8bit_trainvaltest.zip
├── training
│   ├── lightning_module.py
│   ├── mask_classification_instance.py
│   ├── mask_classification_loss.py
│   ├── mask_classification_panoptic.py
│   ├── mask_classification_semantic.py
│   └── two_stage_warmup_poly_schedule.py


## [1] Changes in `cityscapes_semantic.py`

We change the `target_parser` function to fit our task. Since our dataset has only 9 classes (including background), index from 0 to 8.

Thanks to Kimi-2.6 AI LLM, helping me to reconstruct this method.

```python
@staticmethod
def target_parser(target, **kwargs):
    masks, labels = [], []
    
    for label_id in target[0].unique():
        # 跳过忽略区域（如果有的话）
        if label_id == 255:
            continue
            
        # 只保留 0~8 的有效类别
        if label_id < 0 or label_id > 8:
            continue
            
        masks.append(target[0] == label_id)
        labels.append(int(label_id))
        
    return masks, labels, [False for _ in range(len(masks))]
```


note:
1. If you want to run training, use command "CUDA_VISIBLE_DEVICES=1 python3 main.py fit -c configs/dinov2/cityscapes/semantic/eomt_large_1024.yaml --trainer.devices 1 --data.batch_size 4 --data.path ./myDataset/data" .
2. HaoyuHuang add a method named "test_step()" in the training/lightening_module.py , which will automatically implemented by lightening_module, and predict the images in img/. The output will be put in img_out/
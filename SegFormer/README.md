# SegFormer Project

> This is a project implementing SegFormer.
> This Project is original from the github repository bubbliiiing/segformer-pytorch, adapted to the AI Competition project by HaoyuHuang at 2026-7-14.
>
> We express our sincere gratitude toward the author here. He generously used the MIT protocol so that we can take part in the competition based on his code.

---

## Project Structure

```text
├── nets/
│   └── The net structuct of the net is defined here.
│
├── utils/
│   └── some tool function and class like dataloader is defined here.
│
├── VOCdevkit/
│   ├── The image data for semantic segmentation is put here, following the VOC data format.
│   │
│   ├── JPEGImages/
│   │   └── I put all the train image here.
│   │
│   ├── SegmentationClass/
│   │   └── I put all the mask images here.
│   │
│   ├── ImagesSets/
│   │   └── In which I define the trian and validation set by train.txt and val.txt
│   │
│   └── Label.txt
│       └── All the class is defined here.
│
│   > note: The images in the validation set are also in the train set, beacuse the size of train is so small, and I have to use the validation to train the net.
│
├── model_data/
│   ├── The weights of the model is stored here.
│   │
│   ├── segformer_b1_weights_voc.pth
│   │   └── The weights of the whole model.
│   │
│   └── segformer_b1_backbone_weights.pth
│       └── The weights of the backbone.
│
│   > note: When training, the train.py will load the weights in the model_data/ and start training based on the weights, However, the new model which is generated during training will be stored in logs/. Everytime before we start training, we should move the weights in the logs/ to model_data/, rename it as "segformer_b1_weights_voc.pth" and then start training.
│
├── logs/
│   └── Every training epoch, the train.py will save the LOSS information and the model file here, and flag the best model.
│
├── img/
│   └── All the test data are stored here, when testing, the output images will be stored in img_out/.
│
├── train.py
│   └── All the code related to training is here.
│
├── predict.py
│   └── Three modes of prediction is here, namely Video, single picture and multiple pictures. Note that I use the gray picture output, meaning that it will output picture with 9 channels instead of 3. If it is necessary to get color output, change the function in this file to another function. See:segformer.py
│
├── segformer.py
│   └── The preprocess.
│
├── get_miou.py
│   └── Through which we can calculate the miou.
│
└── Others
    └── (summary.py json_to_dataset.py)
```
## Adaptation

1. I change the num_class and some other configurations to fit our task, inculding:
   - `num_class` is changed to 9 (indexed from 0 to 8)

I download the b1 and b2 pretrained model weights and train them on the dataset provided by the AI Competition.

## Install

bash
pip3 install -r requirememts.txt

## Run

### 1. Training

bash
torchrun --nproc_per_node=4 train.py

> **Note:** When training, the script will load the `pretrained backbone weights (named with keyword "backbone")` and the `whole model weights (named with keyword "voc")`.
> To train model, you should make sure that the 2 files above are put in model_Data/.
> During training, the best model weights will be put in logs/.
> Next time for continuing training, you should move the weights in logs/ to model_data/ by hand.
> IMPORTANT! Though I split the dataset into trainset and validationset. I used the pictures in the validationset to train. So if you want to validate the train, you should summit the predict output to the competition website and get the mIou score.

### 2. Predict

bash
python3 predict.py

And the output will be put in `img_out/`

### 3. Calculate the mIoU on validation

bash
python3 get_miou.py

The mIou information will be output in `moiu/`

## LOGs

1. 20+20+85 epochs have been run yo train b1.
2. 60 epochs have been run to trsin b5.

## Future Work

To enhance the performance of the model, I propose the following measures:

1. Data enforcement.
2. test-time-adaptation / test-time-training
3. Implement multiple stages prediting.
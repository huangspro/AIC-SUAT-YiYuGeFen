# This Project is original from the github repository bubbliiiing/segformer-pytorch, and is adapted to the AI Competition project by Haoyu Huang int 2026-7-14.

> We express our sincere gratitude toward the author here. He generously used the MIT protocol so that we can take part in the competition based on his code.

The original project defines all the segformer net structure and train-test precess, and what I did is add the pretrained weigths and change some of code to fit our competition requirements. The code now can train and test normally.

## The work process of the project is that(After I rewrite some code):

1. `nets/` --------------------------The net structuct of the net is defined here.
2. `utils/` -------------------------some tool function and class like dataloader is defined here.
3. `VOCdevkit/` --------------------The image data for semantic segmentation is put here, following the VOC data format.
   - `JPEGImages/` ----------------I put all the train image here.
   - `SegmentationClass/` -----------I put all the mask images here.
   - `ImagesSets/` -----------------In which I define the trian and validation set by train.txt and val.txt
   - `Label.txt` -------------------All the class is defined here.
   - **note:** The images in the validation set are also in the train set, beacuse the size of train is so small, and I have to use the validation to train the net.

4. `model_data/` ----------------The weights of the model is stored here.
   - `segformer_b1_weights_voc.pth` ---------The weights of the whole model.
   - `segformer_b1_backbone_weights.pth` -----The weights of the backbone.
   - **note:** When training, the train.py will load the weights in the model_data/ and start training based on the weights, However, the new model which is generated during training will be stored in logs/. Everytime before we start training, we should move the weights in the logs/ to model_data/, rename it as "segformer_b1_weights_voc.pth" and then start training.

5. `logs/` -----------------------Every training epoch, the train.py will save the LOSS information and the model file here, and flag the best model.

6. `img/` ------------------All the test data are stored here, when testing, the output images will be stored in img_out/.

7. `train.py` ------------All the code related to training is here.
8. `predict.py` ---------Three modes of prediction is here, namely Video, single picture and multiple pictures. Note that I use the gray picture output, meaning that it will output picture with 9 channels instead of 3. If it is necessary to get color output, change the function in this file to another function. See:segformer.py
9. `segformer.py` ------The preprocess.
10. `get_miou.py` ------Through which we can calculate the miou.
11. Others(summary.py json_to_dataset.py)

## How to run the project ?

1. When we want to start training, run the code `"torchrun --nproc_per_node=4 train.py"` And this will use 4 GPUs on the machine.
2. If we want to predict, run the code `"python3 predict.py"` And the output will be put in img_out/
3. If we want to calculate the miou, run the code `"python3 get_miou.py"`

20+20+85 epoch have been run.
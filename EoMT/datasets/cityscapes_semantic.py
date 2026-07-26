# ---------------------------------------------------------------  
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.  
# Licensed under the MIT License.  
# ---------------------------------------------------------------  
  
  
from pathlib import Path  
from typing import Union  
from torch.utils.data import DataLoader  
from torchvision.datasets import Cityscapes  
  
from datasets.lightning_data_module import LightningDataModule  
from datasets.dataset import Dataset  
from datasets.transforms import Transforms  
from datasets.predict_dataset import PredictDataset  
  
class CityscapesSemantic(LightningDataModule):  
    def __init__(  
        self,  
        path,  
        num_workers: int = 4,  
        batch_size: int = 4,  
        img_size: tuple[int, int] = (1024, 1024),  
        num_classes: int = 9,  
        color_jitter_enabled=True,  
        scale_range=(0.5, 2.0),  
        check_empty_targets=True,  
        predict_img_dir=None,  
        predict_save_dir=None  
    ) -> None:  
        super().__init__(  
            path=path,  
            batch_size=batch_size,  
            num_workers=num_workers,  
            num_classes=num_classes,  
            img_size=img_size,  
            check_empty_targets=check_empty_targets,  
        )  
        self.save_hyperparameters(ignore=["_class_path"])  
        self.predict_img_dir = predict_img_dir
        self.predict_save_dir = predict_save_dir
        self.transforms = Transforms(  
            img_size=img_size,  
            color_jitter_enabled=color_jitter_enabled,  
            scale_range=scale_range,  
        )  
  
    @staticmethod  
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
  
    def setup(self, stage: Union[str, None] = None):  
  
        # ---------- Predict ----------  
        if stage == "predict":  
            self.predict_dataset = PredictDataset(  
                image_dir=self.predict_img_dir,  
                save_dir=self.predict_save_dir,  
            )  
            return  
  
        # ---------- Train / Val ----------  
        cityscapes_dataset_kwargs = {  
            "img_suffix": ".png",  
            "target_suffix": ".png",  
            "img_stem_suffix": "",  
            "target_stem_suffix": "",  
            "zip_path": Path(self.path, "leftImg8bit_trainvaltest.zip"),  
            "target_zip_path": Path(self.path, "gtFine_trainvaltest.zip"),  
            "target_parser": self.target_parser,  
            "check_empty_targets": self.check_empty_targets,  
        }  
  
        self.cityscapes_train_dataset = Dataset(  
            transforms=self.transforms,  
            img_folder_path_in_zip=Path("./leftImg8bit/train"),  
            target_folder_path_in_zip=Path("./gtFine/train"),  
            **cityscapes_dataset_kwargs,  
        )  
  
        self.cityscapes_val_dataset = Dataset(  
            img_folder_path_in_zip=Path("./leftImg8bit/val"),  
            target_folder_path_in_zip=Path("./gtFine/val"),  
            **cityscapes_dataset_kwargs,  
        )  
  
    def train_dataloader(self):  
        return DataLoader(  
            self.cityscapes_train_dataset,  
            shuffle=True,  
            drop_last=True,  
            collate_fn=self.train_collate,  
            **self.dataloader_kwargs,  
        )  
  
    def val_dataloader(self):  
        return DataLoader(  
            self.cityscapes_val_dataset,  
            collate_fn=self.eval_collate,  
            **self.dataloader_kwargs,  
        )  
  
    # added by HaoyuHuang 2026-7-26  
    def predict_dataloader(self):  
        return DataLoader(  
            self.predict_dataset,  
            batch_size=2,  
            shuffle=False,  
            num_workers=4,  
            pin_memory=True,  
        )  
  

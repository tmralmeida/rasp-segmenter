import os
import cv2
from torch.utils.data import Dataset
from typing import List, Tuple
from datasets.utils import exr_to_jpg, category_label
from constants.config import *
import torch
import numpy as np
from copy import deepcopy
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
import random


os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

class SyntDs(Dataset):
    """_summary_

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, data : List[Tuple], mode : str, transforms = False) -> None:
        super().__init__()
        self.data = data[mode]
        self.mode = mode
        self.transforms = transforms
        
    
    @staticmethod
    def my_segmentation_transforms(image : torch.Tensor, segmentation : torch.Tensor, mode : str) -> Tuple[torch.Tensor]:
        image = TF.resize(image, INPUT_SHAPE)
        segmentation = TF.resize(segmentation, INPUT_SHAPE, interpolation = InterpolationMode.NEAREST)
        if random.random() > 0.5 and mode == "train":
            angle = random.randint(-30, 30)
            image = TF.rotate(image, angle)
            segmentation = TF.rotate(segmentation, angle)
        # more transforms ...
        return image, segmentation
    
    
    def __getitem__(self, index : int) -> Tuple[torch.Tensor]:
        spl = self.data[index]
        rgb_path, depth_path, label_path = spl.rgb, spl.depth, spl.label
        rgb_img, _ = np.asarray(exr_to_jpg(rgb_path)) / 255, np.asarray(exr_to_jpg(depth_path))
        original_mask = cv2.imread(label_path,cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        array_mask = torch.from_numpy(category_label(original_mask[:, :, 0], INPUT_SHAPE, NUM_CLASSES))
        rgb_img = deepcopy(rgb_img).reshape(rgb_img.shape[-1], rgb_img.shape[0], rgb_img.shape[1])
        y_input = deepcopy(array_mask).reshape(array_mask.shape[-1], array_mask.shape[0], array_mask.shape[1])
        x_input = torch.from_numpy(deepcopy(rgb_img))
        if self.transforms or self.mode in ["val", "test"]: # val, test -> resize
            x_input, y_input = self.my_segmentation_transforms(x_input, y_input, mode = self.mode)
        return x_input, y_input


    def __len__(self) -> int:
        return len(self.data)



if __name__ == "__main__":
    from .utils import create_sets
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    
    info_ds = create_sets()
    train_ds = SyntDs(data = info_ds, 
                      mode = "train")
    val_ds = SyntDs(data = info_ds, 
                    mode = "val")
    
    
    train_dl = DataLoader(dataset = train_ds, 
                          batch_size = BATCH_SIZE,
                          shuffle = True, 
                          num_workers = 4)
    
    val_dl = DataLoader(dataset = val_ds, 
                          batch_size = BATCH_SIZE,
                          shuffle = False, 
                          num_workers = 4)
    

    print("Testing train set")
    with tqdm(train_dl, unit="batch") as tepoch:
        for batch in tepoch:
            rgb_img, lbl = batch
            # print("input shape {} label shape:{}".format(rgb_img.shape, lbl.shape))
    
    print("Testing val set")
    with tqdm(val_dl, unit="batch") as tepoch:
        for batch in tepoch:
            rgb_img, lbl = batch
            #print("input shape {} label shape:{}".format(rgb_img.shape, lbl.shape))
    
    if "test" in info_ds.keys():
        test_ds = SyntDs(data = info_ds, 
                        mode = "test")
        test_dl = DataLoader(dataset = test_ds, 
                            batch_size = BATCH_SIZE,
                            shuffle = False, 
                            num_workers = 4)
        print("Testing test set")
        with tqdm(test_dl, unit="batch") as tepoch:
            for batch in tepoch:
                rgb_img, lbl = batch
                #print("input shape {} label shape:{}".format(rgb_img.shape, lbl.shape))
            
    print("Datasets and dataloaders passed!")
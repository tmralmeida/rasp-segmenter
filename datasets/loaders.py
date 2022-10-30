import os
import cv2
from torch.utils.data import Dataset
from typing import List, Tuple
from .utils import exr_to_jpg, category_label
from ..constants import *
import torch
import numpy as np
from copy import deepcopy


os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

class SyntDs(Dataset):
    """_summary_

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, data : List[Tuple], mode : str, transforms = None) -> None:
        super().__init__()
        self.data = data[mode]
        self.transforms = transforms
        
        
    def __getitem__(self, index):
        spl = self.data[index]
        rgb_path, depth_path, label_path = spl.rgb, spl.depth, spl.label
        rgb_img, dep_img = np.asarray(exr_to_jpg(rgb_path)), np.asarray(exr_to_jpg(depth_path))
        original_mask = cv2.imread(label_path,cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        array_mask = torch.from_numpy(category_label(original_mask[:, :, 0], INPUT_SHAPE, NUM_CLASSES))
        rgb_img = deepcopy(rgb_img).reshape(rgb_img.shape[-1], rgb_img.shape[0], rgb_img.shape[1])
        array_mask = deepcopy(array_mask).reshape(array_mask.shape[-1], array_mask.shape[0], array_mask.shape[1])
        
        
        return torch.from_numpy(deepcopy(rgb_img)), array_mask


    def __len__(self):
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
from collections import namedtuple
from random import seed
import ray
import numpy as np
import os 
from ..constants import *
from decimal import *
from typing import List, Tuple
import itertools
from itertools import product
from collections import namedtuple
import imageio.v2 as imageio
import torch
from PIL import Image


DataPoint = namedtuple("DataPoint", ["rgb", "depth", "label"])


def colorize(img):
    z, w, h  = img.shape
    l=torch.zeros((3, w,h))
    for i, j in product(range(w),range(h)):
        if img[0, i,j]==1:
            print("class 0")
            l[0,i,j]=0
            l[1,i,j]=0
            l[2,i,j]=0
        elif img[1, i,j]==1:
            print("class 1")
            l[0,i,j]=255
            l[1,i,j]=0
            l[2,i,j]=0
        elif img[2, i,j]==1:
            print("class 2")
            l[0,i,j]=0
            l[1,i,j]=255
            l[2,i,j]=0
        elif img[3, i,j]==1:
            print("class 3")
            l[0,i,j]=0
            l[1,i,j]=0
            l[2,i,j]=255
        elif img[4, i,j]==1:
            print("class 4")
            l[0,i,j]=238
            l[1,i,j]=197
            l[2,i,j]=145
    return l



def exr_to_jpg(path):
    im = imageio.imread(path)
    im_gamma_correct = np.clip(np.power(im, 0.45), 0, 1)
    im_fixed = Image.fromarray(np.uint8(im_gamma_correct*255))
    return im_fixed


def category_label(labels, dims, n_labels):
    x = np.zeros([dims[0], dims[1], n_labels])
    for i, j in product(range(dims[0]),range(dims[1])):
            f=int(labels[i,j])
            x[i, j, f] = 1
    #x = x.reshape(dims[0] * dims[1], n_labels)
    return x



@ray.remote
def load_from_dir(path : str) -> Tuple[List[np.array], List[np.array], List[np.array]]:
    files = os.listdir(path)
    rgbs, depths, lbls = [], [], []
    for fn in files:
        if fn.endswith(".exr"):
                path_file = os.path.join(path, fn)
                depth_file = os.path.join(RAW_DATA_PATH, "depth_map", "profondeur_map", fn)
                mask_file = os.path.join(RAW_DATA_PATH, "semantic_map", "segmentation2_map", fn)
                rgbs.append(path_file)
                depths.append(depth_file)
                lbls.append(mask_file)
                
    return rgbs, depths, lbls        
                
def create_sets():
    #creat_synt_ds()
    getcontext().prec = 2
    full_paths = []
    for dirs, _, files in os.walk(os.path.join(RAW_DATA_PATH, "rgb_map")):
        if len(files) > 0:
            full_paths.append(dirs)
    print("Found data directories:", full_paths)
    
    
    outputs = ray.get([load_from_dir.remote(fp) for fp in full_paths])
    rgbs = list(itertools.chain.from_iterable(o[0] for o in outputs))
    depths = list(itertools.chain.from_iterable(o[1] for o in outputs))
    labels = list(itertools.chain.from_iterable(o[2] for o in outputs))
    
    assert len(rgbs) == len(depths) == len(labels), "Sth wrong with the data"
    assert Decimal(TRAIN_PT) + Decimal(VAL_PT) + Decimal(TEST_PT) == 1.0, "% of each set must sum up to 1" 
    testing_set = TEST_PT > 0.0
    print("All files visited!")
    liste = np.arange(1,len(rgbs))
    np.random.seed(42)
    np.random.shuffle(liste)

    train_list = liste[0:int(TRAIN_PT * len(rgbs))]
    val_list = liste[int(TRAIN_PT * len(rgbs)):int(TRAIN_PT * len(rgbs)) + int(VAL_PT * len(rgbs))]
    test_list = liste[int(TRAIN_PT * len(rgbs)) + int(VAL_PT * len(rgbs)):]
    
    train_info = [DataPoint(rgbs[idx], depths[idx], labels[idx]) for idx in train_list] 
    val_info = [DataPoint(rgbs[idx], depths[idx], labels[idx]) for idx in val_list] 
    test_info = [DataPoint(rgbs[idx], depths[idx], labels[idx]) for idx in test_list] 
    
    info = {"train" : train_info, 
            "val" : val_info}
    if testing_set:
        info.update({"test" : test_info})
    return info



if __name__ == "__main__":
    create_sets()
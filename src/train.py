import pytorch_lightning as pl
from torch.utils.data import DataLoader

from ..constants import *
from ..datasets.loaders import SyntDs
from ..datasets.utils import create_sets


if DATASET_NAME == "synthetic":
    dataset_info = create_sets()
    train_ds = SyntDs(data = dataset_info, 
                      mode = "train")
    train_dl = DataLoader(dataset = train_ds,
                          batch_size = BATCH_SIZE,
                          shuffle = True,
                          num_workers = 4)
    val_ds = SyntDs(data = dataset_info, 
                    mode = "val")
    val_dl = DataLoader(dataset = val_ds,
                        batch_size = BATCH_SIZE,
                        shuffle = False,
                        num_workers = 4)
    if "test" in dataset_info.keys():
        test_ds = SyntDs(data = dataset_info, 
                        mode = "test")
        test_dl = DataLoader(dataset = test_ds,
                            batch_size = BATCH_SIZE,
                            shuffle = False,
                            num_workers = 4)
else:
    raise NotImplementedError(DATASET_NAME)

#train_ds = SyntDs(mode = "train")

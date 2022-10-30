from re import S
from ..constants import *
from ..datasets.loaders import SyntDs
from ..datasets.utils import create_sets

if DATASET_NAME == "synthetic":
    dataset_info = create_sets()
    train_ds = SyntDs(data = dataset_info, 
                      mode = "train",
                      )
else:
    raise NotImplementedError(DATASET_NAME)

#train_ds = SyntDs(mode = "train")

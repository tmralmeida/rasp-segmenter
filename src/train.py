import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from ..constants import *
from ..datasets.loaders import SyntDs
from ..datasets.utils import create_sets
from ..models.segnet import SegNet
from ..losses.ohem_ce import OHEMCrossEntropyLoss
from .light_hl import LitSegNet


# define the LightningModule
if MODEL_NAME == "segnet":
    model = SegNet(in_channels = INPUT_CHANNELS, 
                   out_channels = NUM_CLASSES)
else:
    raise NotImplementedError(MODEL_NAME)


loss = OHEMCrossEntropyLoss()
ligt_model = LitSegNet(model = model,
                       loss = loss)


# setup data
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


trainer = pl.Trainer(limit_train_batches=100, max_epochs=1, accelerator="gpu")
trainer.fit(model = ligt_model, train_dataloaders = train_dl)

import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from ..constants.config import *
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
NUM_WORKERS, testing = int(os.cpu_count() / 2), False
if DATASET_NAME == "synthetic":
    dataset_info = create_sets()
    train_ds = SyntDs(data = dataset_info, 
                      mode = "train", 
                      transforms = True)
    train_dl = DataLoader(dataset = train_ds,
                          batch_size = BATCH_SIZE,
                          shuffle = True,
                          num_workers = NUM_WORKERS)
    val_ds = SyntDs(data = dataset_info, 
                    mode = "val")
    val_dl = DataLoader(dataset = val_ds,
                        batch_size = BATCH_SIZE,
                        shuffle = False,
                        num_workers = NUM_WORKERS)
    if "test" in dataset_info.keys():
        testing = True
        test_ds = SyntDs(data = dataset_info, 
                        mode = "test")
        test_dl = DataLoader(dataset = test_ds,
                            batch_size = BATCH_SIZE,
                            shuffle = False,
                            num_workers = NUM_WORKERS)
else:
    raise NotImplementedError(DATASET_NAME)


trainer = pl.Trainer(max_epochs = NUM_EPOCHS, accelerator = "gpu", check_val_every_n_epoch = VAL_EVERY, default_root_dir = CHECK_SAVE_PATH)
trainer.fit(model = ligt_model, train_dataloaders = train_dl, val_dataloaders = val_dl)
if testing:
    trainer.test(ckpt_path = "best", dataloaders = test_dl)
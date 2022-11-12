import pytorch_lightning as pl
from torch.optim import Adam
from datasets.utils import colorize
from constants.config import * 
import random
import torchvision
import numpy as np
import torch


class LitSegNet(pl.LightningModule):
    def __init__(self, model, loss) -> None:
        super().__init__()
        self.model =  model
        self.loss = loss
        self.save_hyperparameters()
        self.vals_out = None
        
    def training_step(self, batch):
        x, y = batch
        y_hat = self.model(x.float().to(self.device))
        loss = self.loss(y_hat.to(self.device), y.to(self.device))
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer =  Adam(self.parameters(), lr = LR)
        return optimizer
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x.float().to(self.device))
        loss = self.loss(y_hat.to(self.device), y.to(self.device))
        self.log("val_loss", loss)
        if batch_idx == 0:
            idx_log = random.randint(0, x.size(0) - 2)
            src2log = x[idx_log].clone().squeeze().cpu().numpy() * 255
            seg2log = y_hat[idx_log].clone().squeeze().softmax(dim = 0).cpu().numpy() 
            seg_img = colorize(seg2log)
            stacked_imgs = np.stack([src2log.astype(np.uint8), seg_img.astype(np.uint8)], axis = 0)
            stacked_imgs = torchvision.utils.make_grid(torch.from_numpy(stacked_imgs))
            self.vals_out = stacked_imgs
        
    def on_validation_end(self) -> None:
        self.logger.experiment.add_image("src-vs-seg", self.vals_out, self.current_epoch, dataformats="CHW")
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x.float().to(self.device))
        loss = self.loss(y_hat.to(self.device), y.to(self.device))
        self.log("test_loss", loss)
    
    
        
        
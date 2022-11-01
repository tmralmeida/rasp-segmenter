import pytorch_lightning as pl
from torch.optim import Adam
from ..datasets.utils import colorize
from ..constants import * 
import random


class LitSegNet(pl.LightningModule):
    def __init__(self, model, loss) -> None:
        super().__init__()
        self.model =  model
        self.loss = loss
        
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
        img2_log = y_hat.clone()[random.randint(0, x.size(0) - 2), ...].squeeze()
        seg_img = colorize(img2_log)
        self.logger.experiment.add_image("segmented_images", seg_img, self.current_epoch)
        self.log("val_loss", loss)
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x.float().to(self.device))
        loss = self.loss(y_hat.to(self.device), y.to(self.device))
        self.log("test_loss", loss)
    
    
        
        
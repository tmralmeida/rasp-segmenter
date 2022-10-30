import pytorch_lightning as pl
from torch.optim import Adam
from ..constants import * 

class LitSegNet(pl.LightningModule):
    def __init__(self, model, loss) -> None:
        super().__init__()
        self.model =  model
        self.loss = loss
        
    def training_step(self, batch):
        x, y = batch
        y_hat = self.model(x.float().to(self.device))
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        return loss
    
    
    def configure_optimizers(self):
        optimizer =  Adam(self.parameters(), lr = LR)
        return optimizer
    
    
        
        
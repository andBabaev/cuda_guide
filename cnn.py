import timm
import torch.nn as nn

from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchmetrics.classification import Accuracy, Precision, Recall, F1

from torch import sigmoid
from torch.optim import Adam
from pytorch_lightning import LightningModule


class ClassificationModel(LightningModule):
    def __init__(self, 
                 model_name='resnet18', 
                 n_classes=1, 
                 lr=0.001
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.model = timm.create_model(self.hparams.model_name, 
                                       pretrained=True, 
                                       num_classes=self.hparams.n_classes)
        self.sigmoid = nn.Sigmoid()
        
        
        config = resolve_data_config({}, model=self.model)
        self.preprocessing_fn = create_transform(**config)

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.accuracy = Accuracy()
        self.precision_ = Precision()
        self.recall = Recall()
        self.f1 = F1()
        self.metrics = [
            ("accuracy", self.accuracy),
            # ("precision", self.precision_),
            # ("recall", self.recall),
            ("f1", self.f1),
        ]
        
    def forward(self, x):
        x = self.model(x)
        # x = self.sigmoid(x)
        return x


    def training_step(self, batch, batch_idx):

        if not self.model.training:
            self.model.train()

        x, y = batch
        logits = self.model(x)

        loss = self.loss_fn(logits, y.float().unsqueeze(1))

        log_values = {}
        logits = sigmoid(logits)
        y = y.unsqueeze(1)
        for metric_name, metric_fn in self.metrics:
            log_values['train_{}'.format(metric_name)] = metric_fn(logits, y)
            
        log_values['train_loss'] = loss
        self.log_dict(log_values, prog_bar=True, logger=True, 
                      on_epoch=True, on_step=False)
        

        return loss

    def validation_step(self, batch, batch_idx):

        if self.model.training:
            self.model.eval()

        x, y = batch
        logits = self.model(x)

        loss = self.loss_fn(logits, y.float().unsqueeze(1))

        log_values = {}
        logits = sigmoid(logits)
        y = y.unsqueeze(1)
        for metric_name, metric_fn in self.metrics:
            log_values['val_{}'.format(metric_name)] = metric_fn(logits, y)
        log_values['val_loss'] = loss
        self.log_dict(log_values, prog_bar=True, logger=True, on_epoch=True, on_step=False)

        return loss

    def configure_optimizers(self):
        self.optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        return self.optimizer
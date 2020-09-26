import torch
import torch.nn as nn

from omegaconf import DictConfig
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from pytorch_lightning import LightningModule, EvalResult, TrainResult

from ..metric import get_auc

class AutotaggingRunner(LightningModule):
    def __init__(self, model: nn.Module, runner_config: DictConfig):
        super().__init__()
        self.model = model
        self.criterion = nn.BCELoss()
        self.hparams.update(runner_config.optimizer.params)
        self.hparams.update(runner_config.scheduler.params)
        self.hparams.update(runner_config.trainer.params)
    
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        opt = Adam(params=self.model.parameters(),
                   lr=self.hparams.learning_rate)
        scheduler = ExponentialLR(opt, gamma=self.hparams.gamma)
        return [opt], [scheduler]


    def training_step(self, batch, batch_idx):
        audio, label = batch
        prediction = self.model(audio)
        loss = self.criterion(prediction, label)
        return {
            "loss": loss,
            "progress_bar": {"train_loss": loss},
            "log": {"train_loss": loss},
        }

    def validation_step(self, batch, batch_idx):
        audio, label = batch
        prediction = self.model(audio)
        loss = self.criterion(prediction, label)
        return {'val_loss': loss, 'predictions': prediction, 'labels': label}
    
    def validation_epoch_end(self, outputs):
        val_loss = torch.mean(torch.stack([output["val_loss"] for output in outputs]))
        predctions = torch.stack([output["predictions"] for output in outputs])
        labels = torch.stack([output["labels"] for output in outputs])
        roc_auc, pr_auc = get_auc(predctions, labels)
        return {
            "progress_bar": {'val_loss': val_loss, 'roc_auc': roc_auc, 'pr_auc': pr_auc},
            "log": {'val_loss': val_loss, 'roc_auc': roc_auc, 'pr_auc': pr_auc},
        }

    def test_step(self, batch, batch_idx):
        audio, label = batch
        prediction = self.model(audio)
        loss = self.criterion(prediction, label)
        return {'val_loss': loss, 'predictions': prediction, 'labels': label}

    def test_epoch_end(self, outputs):
        val_loss = torch.mean(torch.stack([output["val_loss"] for output in outputs]))
        predctions = torch.stack([output["predictions"] for output in outputs])
        labels = torch.stack([output["labels"] for output in outputs])
        roc_auc, pr_auc = get_auc(predctions, labels)
        return {
            "log": {'val_loss': val_loss, 'roc_auc': roc_auc, 'pr_auc': pr_auc},
        }
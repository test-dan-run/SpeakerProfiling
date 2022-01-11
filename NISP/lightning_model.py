import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from torchmetrics.regression import MeanAbsoluteError as MAE
from torchmetrics.regression import MeanSquaredError  as MSE
from torchmetrics.classification import Accuracy

from Model.models import Wav2VecLSTM
import pandas as pd

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))

class LightningModel(pl.LightningModule):
    def __init__(self, hidden_size, alpha=1, beta=1, gamma=1, lr=1e-3, csv_path=None):
        super().__init__()
        # HPARAMS
        self.save_hyperparameters()
        self.model = Wav2VecLSTM(hidden_size)

        self.classification_criterion = MSE()
        self.regression_criterion = MSE()
        self.mae_criterion = MAE()
        self.rmse_criterion = RMSELoss()
        self.accuracy = Accuracy()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.lr = lr

        self.csv_path = csv_path
        self.df = pd.read_csv(self.csv_path, sep=' ')
        self.h_mean = self.df['Height'].mean()
        self.h_std = self.df['Height'].std()
        self.a_mean = self.df['Age'].mean()
        self.a_std = self.df['Age'].std()

        print(f"Model Details: #Params = {self.count_total_parameters()}\t#Trainable Params = {self.count_trainable_parameters()}")

    def count_total_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return [optimizer]

    def training_step(self, batch, batch_idx):
        x, y_h, y_a, y_g = batch
        y_hat_h, y_hat_a, y_hat_g = self(x)
        y_h, y_a, y_g = y_h.view(-1).float(), y_a.view(-1).float(), y_g.view(-1).float()
        y_hat_h, y_hat_a, y_hat_g = y_hat_h.view(-1).float(), y_hat_a.view(-1).float(), y_hat_g.view(-1).float()

        height_loss = self.regression_criterion(y_hat_h, y_h)
        age_loss = self.regression_criterion(y_hat_a, y_a)
        gender_loss = self.classification_criterion(y_hat_g, y_g)
        loss = self.alpha * height_loss + self.beta * age_loss + self.gamma * gender_loss

        height_mae = self.mae_criterion(y_hat_h*self.h_std+self.h_mean, y_h*self.h_std+self.h_mean)
        age_mae =self.mae_criterion(y_hat_a*self.a_std+self.a_mean, y_a*self.a_std+self.a_mean)
        gender_acc = self.accuracy((y_hat_g>0.5).long(), y_g.long())

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False)

        return {'loss':loss, 
                'train_height_mae':height_mae.item(),
                'train_age_mae':age_mae.item(),
                'train_gender_acc':gender_acc,
                }
    
    def training_epoch_end(self, outputs):
        n_batch = len(outputs)
        loss = torch.tensor([x['loss'] for x in outputs]).mean()
        height_mae = torch.tensor([x['train_height_mae'] for x in outputs]).sum()/n_batch
        age_mae = torch.tensor([x['train_age_mae'] for x in outputs]).sum()/n_batch
        gender_acc = torch.tensor([x['train_gender_acc'] for x in outputs]).mean()

        self.log('epoch_loss' , loss, prog_bar=True)
        self.log('h_mae',height_mae.item(), prog_bar=True)
        self.log('a_mae',age_mae.item(), prog_bar=True)
        self.log('g_acc',gender_acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y_h, y_a, y_g = batch
        y_hat_h, y_hat_a, y_hat_g = self(x)
        y_h, y_a, y_g = y_h.view(-1).float(), y_a.view(-1).float(), y_g.view(-1).float()
        y_hat_h, y_hat_a, y_hat_g = y_hat_h.view(-1).float(), y_hat_a.view(-1).float(), y_hat_g.view(-1).float()

        height_loss = self.regression_criterion(y_hat_h, y_h)
        age_loss = self.regression_criterion(y_hat_a, y_a)
        gender_loss = self.classification_criterion(y_hat_g, y_g)
        loss = self.alpha * height_loss + self.beta * age_loss + self.gamma * gender_loss

        height_mae = self.mae_criterion(y_hat_h*self.h_std+self.h_mean, y_h*self.h_std+self.h_mean)
        age_mae = self.mae_criterion(y_hat_a*self.a_std+self.a_mean, y_a*self.a_std+self.a_mean)
        gender_acc = self.accuracy((y_hat_g>0.5).long(), y_g.long())

        return {'val_loss':loss, 
                'val_height_mae':height_mae.item(),
                'val_age_mae':age_mae.item(),
                'val_gender_acc':gender_acc}

    def validation_epoch_end(self, outputs):
        n_batch = len(outputs)
        val_loss = torch.tensor([x['val_loss'] for x in outputs]).mean()
        height_mae = torch.tensor([x['val_height_mae'] for x in outputs]).sum()/n_batch
        age_mae = torch.tensor([x['val_age_mae'] for x in outputs]).sum()/n_batch
        gender_acc = torch.tensor([x['val_gender_acc'] for x in outputs]).mean()
        
        self.log('v_loss' , val_loss, prog_bar=True)
        self.log('v_h_mae',height_mae.item(), prog_bar=True)
        self.log('v_a_mae',age_mae.item(), prog_bar=True)
        self.log('v_g_acc',gender_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y_h, y_a, y_g = batch
        y_hat_h, y_hat_a, y_hat_g = self(x)
        y_h, y_a, y_g = y_h.view(-1).float(), y_a.view(-1).float(), y_g.view(-1).float()
        y_hat_h, y_hat_a, y_hat_g = y_hat_h.view(-1).float(), y_hat_a.view(-1).float(), y_hat_g.view(-1).float()

        gender_acc = self.accuracy((y_hat_g>0.5).long(), y_g.long())

        idx = y_g.view(-1).long()
        female_idx = torch.nonzero(idx).view(-1)
        male_idx = torch.nonzero(1-idx).view(-1)

        male_height_mae = self.mae_criterion(y_hat_h[male_idx]*self.h_std+self.h_mean, y_h[male_idx]*self.h_std+self.h_mean)
        male_age_mae = self.mae_criterion(y_hat_a[male_idx]*self.a_std+self.a_mean, y_a[male_idx]*self.a_std+self.a_mean)

        femal_height_mae = self.mae_criterion(y_hat_h[female_idx]*self.h_std+self.h_mean, y_h[female_idx]*self.h_std+self.h_mean)
        female_age_mae = self.mae_criterion(y_hat_a[female_idx]*self.a_std+self.a_mean, y_a[female_idx]*self.a_std+self.a_mean)

        male_height_rmse = self.rmse_criterion(y_hat_h[male_idx]*self.h_std+self.h_mean, y_h[male_idx]*self.h_std+self.h_mean)
        male_age_rmse = self.rmse_criterion(y_hat_a[male_idx]*self.a_std+self.a_mean, y_a[male_idx]*self.a_std+self.a_mean)

        femal_height_rmse = self.rmse_criterion(y_hat_h[female_idx]*self.h_std+self.h_mean, y_h[female_idx]*self.h_std+self.h_mean)
        female_age_rmse = self.rmse_criterion(y_hat_a[female_idx]*self.a_std+self.a_mean, y_a[female_idx]*self.a_std+self.a_mean)

        return {
                'male_height_mae':male_height_mae.item(),
                'male_age_mae':male_age_mae.item(),
                'female_height_mae':femal_height_mae.item(),
                'female_age_mae':female_age_mae.item(),
                'male_height_rmse':male_height_rmse.item(),
                'male_age_rmse':male_age_rmse.item(),
                'femal_height_rmse':femal_height_rmse.item(),
                'female_age_rmse':female_age_rmse.item(),
                'test_gender_acc':gender_acc}

    def test_epoch_end(self, outputs):
        n_batch = len(outputs)
        male_height_mae = torch.tensor([x['male_height_mae'] for x in outputs]).mean()
        male_age_mae = torch.tensor([x['male_age_mae'] for x in outputs]).mean()
        female_height_mae = torch.tensor([x['female_height_mae'] for x in outputs]).mean()
        female_age_mae = torch.tensor([x['female_age_mae'] for x in outputs]).mean()

        male_height_rmse = torch.tensor([x['male_height_rmse'] for x in outputs]).mean()
        male_age_rmse = torch.tensor([x['male_age_rmse'] for x in outputs]).mean()
        femal_height_rmse = torch.tensor([x['femal_height_rmse'] for x in outputs]).mean()
        female_age_rmse = torch.tensor([x['female_age_rmse'] for x in outputs]).mean()

        gender_acc = torch.tensor([x['test_gender_acc'] for x in outputs]).mean()

        pbar = {'male_height_mae' : male_height_mae.item(),
                'male_age_mae':male_age_mae.item(),
                'female_height_mae':female_height_mae.item(),
                'female_age_mae': female_age_mae.item(),
                'male_height_rmse' : male_height_rmse.item(),
                'male_age_rmse':male_age_rmse.item(),
                'femal_height_rmse':femal_height_rmse.item(),
                'female_age_rmse': female_age_rmse.item(),
                'test_gender_acc':gender_acc.item()}
        self.logger.log_hyperparams(pbar)
        self.log_dict(pbar)

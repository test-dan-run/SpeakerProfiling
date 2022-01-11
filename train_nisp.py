import os

from NISP.dataset import NISPDataset
from NISP.lightning_model import LightningModel 

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from config import NISPConfig
import torch.utils.data as data

if __name__ == "__main__":

    cfg = NISPConfig()

    print(f'Training Model on NISP Dataset\n#Cores = {cfg.n_workers}\t#GPU = {cfg.gpu}')

    # hyperparameters and details about the model 
    HPARAMS = {
        'data_path' : cfg.data_path,
        'speaker_csv_path' : cfg.speaker_csv_path,
        'data_wav_len' : cfg.wav_len,
        'data_batch_size' : cfg.batch_size,
        'data_wav_augmentation' : cfg.wav_augmentation,
        'data_label_scale' : cfg.label_scale,

        'training_optimizer' : cfg.optimizer,
        'training_lr' : cfg.lr,
        'training_lr_scheduler' : cfg.lr_scheduler,

        'model_hidden_size' : cfg.hidden_size,
        'model_alpha' : cfg.alpha,
        'model_beta' : cfg.beta,
        'model_gamma' : cfg.gamma,
        'model_architecture' : cfg.architecture,
    }

    # Training, Validation and Testing Dataset
    ## Training Dataset
    train_set = NISPDataset(
        wav_folder = os.path.join(cfg.data_path, 'TRAIN'),
        csv_file = cfg.speaker_csv_path,
        wav_len = cfg.speaker_csv_path,
        noise_dataset_path = cfg.noise_dataset_path
    )
    ## Training DataLoader
    trainloader = data.DataLoader(
        train_set, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        num_workers=cfg.n_workers
    )

    ## Validation Dataset
    valid_set = NISPDataset(
        wav_folder = os.path.join(cfg.data_path, 'VAL'),
        csv_file = cfg.speaker_csv_path,
        wav_len = cfg.wav_len,
        is_train=False
    )
    ## Validation Dataloader
    valloader = data.DataLoader(
        valid_set, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        num_workers=cfg.n_workers
    )

    ## Testing Dataset
    test_set = NISPDataset(
        wav_folder = os.path.join(cfg.data_path, 'TEST'),
        csv_file = cfg.speaker_csv_path,
        wav_len = cfg.speaker_csv_path,
        noise_dataset_path = cfg.noise_dataset_path
    )
    ## Testing DataLoader
    testloader = data.DataLoader(
        test_set, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        num_workers=cfg.n_workers
    )

    print('Dataset Split (Train, Validation, Test)=', len(train_set), len(valid_set), len(test_set))

    #Training the Model
    logger = TensorBoardLogger('NISP_logs', name='')

    model = LightningModel(cfg.hidden_size, cfg.alpha, cfg.beta, cfg.gamma, cfg.lr, cfg.speaker_csv_path)

    checkpoint_callback = ModelCheckpoint(
        monitor='v_loss', 
        mode='min',
        verbose=1)

    trainer = pl.Trainer(fast_dev_run=False, 
                        gpus=cfg.gpu, 
                        max_epochs=cfg.epochs, 
                        checkpoint_callback=checkpoint_callback,
                        callbacks=[
                            EarlyStopping(
                                monitor='v_loss',
                                min_delta=0.00,
                                patience=10,
                                verbose=True,
                                mode='min'
                                )
                        ],
                        logger=logger,
                        resume_from_checkpoint=cfg.model_checkpoint,
                        accelerator='ddp'
                        )

    trainer.fit(model, train_dataloader=trainloader, val_dataloaders=valloader)

    
    print('\n\nCompleted Training...\nTesting the model with checkpoint -', checkpoint_callback.best_model_path)
    model = LightningModel.load_from_checkpoint(checkpoint_callback.best_model_path)
    trainer.test(model, test_dataloaders=testloader)
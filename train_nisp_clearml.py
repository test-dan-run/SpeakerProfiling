from clearml import Task, StorageManager, Dataset

task = Task.init(project_name='NISP', task_name='height_classifier-augmented', output_uri='s3://experiment-logging/storage')
task.set_base_docker('dleongsh/pytorch-1.7.1-audio:public')
task.execute_remotely(queue_name='compute2', clone=False, exit_process=True)

import os

from NISP.dataset import NISPDataset
from NISP.lightning_model import LightningModel 

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from config import ClearMLNISPConfig
import torch.utils.data as data

if __name__ == "__main__":

    cfg = ClearMLNISPConfig()

    print('Retrieving data from S3')
    dataset = Dataset.get(dataset_project=cfg.dataset_project, dataset_name=cfg.dataset_name)
    dataset_path = dataset.get_local_copy()
    os.listdir(dataset_path)

    noise_dataset = Dataset.get(dataset_project=cfg.noise_dataset_project, dataset_name=cfg.noise_dataset_name)
    noise_dataset_path = noise_dataset.get_local_copy()

    print(f'Training Model on NISP Dataset\n#Cores = {cfg.n_workers}\t#GPU = {cfg.gpu}')

    # Training, Validation and Testing Dataset
    ## Training Dataset
    train_set = NISPDataset(
        wav_folder = os.path.join(dataset_path, 'TRAIN'),
        csv_file = os.path.join(dataset_path, 'total_spkrinfo.list'),
        wav_len = cfg.wav_len,
        noise_dataset_path = noise_dataset_path
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
        wav_folder = os.path.join(dataset_path, 'VAL'),
        csv_file = os.path.join(dataset_path, 'total_spkrinfo.list'),
        wav_len = cfg.wav_len,
        noise_dataset_path = noise_dataset_path,
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
        wav_folder = os.path.join(dataset_path, 'TEST'),
        csv_file = os.path.join(dataset_path, 'total_spkrinfo.list'),
        wav_len = cfg.wav_len,
        is_train=False
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

    model = LightningModel(cfg.hidden_size, cfg.alpha, cfg.beta, cfg.gamma, cfg.lr, os.path.join(dataset_path, 'total_spkrinfo.list'))

    checkpoint_callback = ModelCheckpoint(
        filename='best',
        monitor='v_loss', 
        mode='min',
        verbose=1,
        save_top_k=1)

    trainer = pl.Trainer(fast_dev_run=False, 
                        gpus=cfg.gpu, 
                        max_epochs=cfg.epochs, 
                        callbacks=[
                            checkpoint_callback,
                            # EarlyStopping(
                            #     monitor='v_loss',
                            #     min_delta=0.00,
                            #     patience=10,
                            #     verbose=True,
                            #     mode='min'
                            #     )
                        ],
                        logger=logger,
                        resume_from_checkpoint=cfg.model_checkpoint,
                        accelerator='ddp'
                        )

    trainer.fit(model, train_dataloader=trainloader, val_dataloaders=valloader)

    
    print('\n\nCompleted Training...\nTesting the model with checkpoint -', checkpoint_callback.best_model_path)
    model = LightningModel.load_from_checkpoint(checkpoint_callback.best_model_path)
    trainer.test(model, test_dataloaders=testloader)
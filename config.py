import os

class TIMITConfig(object):
    # path to the unzuipped TIMIT data folder
    data_path = '/home/shangeth/DATASET/TIMIT/wav_data'

    # path to csv file containing age, heights of timit speakers
    speaker_csv_path = os.path.join(str(os.getcwd()), 'src/Dataset/data_info_height_age.csv')

    # length of wav files for training and testing
    timit_wav_len = 3 * 16000
    # 16000 * 2

    batch_size = 150
    epochs = 200
    
    # loss = alpha * height_loss + beta * age_loss + gamma * gender_loss
    alpha = 1
    beta = 1
    gamma = 1

    # training type - AHG/H
    training_type = 'H'

    # data type - raw/spectral
    data_type = 'spectral' 

    # model type
    ## AHG 
    # wav2vecLSTMAttn/spectralCNNLSTM/MultiScale
    
    ## H
    # wav2vecLSTMAttn/MultiScale/LSTMAttn
    model_type = 'MultiScale'

    # hidden dimension of LSTM and Dense Layers
    hidden_size = 128

    # No of GPUs for training and no of workers for datalaoders
    gpu = '-1'
    n_workers = 4

    # model checkpoint to continue from
    model_checkpoint = None
    
    # noise dataset for augmentation
    noise_dataset_path = '/home/shangeth/noise_dataset'

    # LR of optimizer
    lr = 1e-3

    run_name = data_type + '_' + training_type + '_' + model_type


class NISPConfig(object):
    # path to the unzuipped TIMIT data folder
    data_path = '/datasets/NISP-Dataset/final_data_16k'

    # path to csv file containing age, heights of timit speakers
    speaker_csv_path = '/datasets/NISP-Dataset/total_spkrinfo.list'

    # length of wav files for training and testing
    wav_len = 16000 * 5

    batch_size = 128
    epochs = 100

    # augmentation
    wav_augmentation = 'Random Crop, Additive Noise'
    label_scale = 'Standardization'

    # optimizer
    optimizer = 'Adam'
    lr = 1e-4
    lr_scheduler = '-'

    # loss = alpha * height_loss + beta * age_loss + gamma * gender_loss
    alpha = 1
    beta = 0
    gamma = 0

    # model architecture
    architecture = 'wav2vec + soft-attention'

    # hidden dimension of LSTM and Dense Layers
    hidden_size = 128

    # No of GPUs for training and no of workers for datalaoders
    gpu = 1
    n_workers = 4

    # model checkpoint to continue from
    model_checkpoint = None

    # noise dataset for augmentation
    noise_dataset_path = '/datasets/wham_noise/tr'

class ClearMLNISPConfig(object):
    
    # clearml dataset IDs
    dataset_project = 'datasets/NISP'
    dataset_name = 'processed'

    # augmentation
    wav_augmentation = 'Random Crop, Additive Noise'
    label_scale = 'Standardization'

    noise_dataset_project = 'datasets/wham_noise'
    noise_dataset_name = 'wham_train'

    # length of wav files for training and testing
    wav_len = 16000 * 5

    batch_size = 8
    epochs = 100

    # optimizer
    optimizer = 'Adam'
    lr = 1e-3
    lr_scheduler = '-'

    # loss = alpha * height_loss + beta * age_loss + gamma * gender_loss
    alpha = 1
    beta = 1
    gamma = 1

    # model architecture
    architecture = 'wav2vec + soft-attention'

    # hidden dimension of LSTM and Dense Layers
    hidden_size = 128

    # No of GPUs for training and no of workers for datalaoders
    gpu = 1
    n_workers = 4

    # model checkpoint to continue from
    model_checkpoint = None

class TestNISPConfig(object):

    # audio recording configs
    channels = 1
    sample_rate = 16000
    record_seconds = 10
    wav_output_filename = 'out1.wav'

    # model configs
    slice_seconds = 5
    slice_window = 1
    
    model_checkpoint = './NISP/hclf-best.ckpt'
    csv_path = './NISP/total_spkrinfo.list'
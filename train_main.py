import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import time
from tensorboardX import SummaryWriter
from tqdm import tqdm
from data_input import *
from Network_structure import *
from loss_function import *
import utils
import os
# Author: Kecheng Chen
# Here is the main part of the denoising neurl network, We can adjust all the parameter in the user-defined area.
# user-defined

epochs = 10000    # training epoch
batch_size = 1000    # training batch size
train_num = 4000   # how many trails for train
validation_num = 100     # how many trails for validation 
combin_num = 10    # combin EEG and noise ? times
denoise_network = 'DeT' #(our model)    #   fcNN   &  simple_CNN_pro   & Simple_CNN   &    Complex_CNN    &    RNN_lstm
result_location = 'your path/benchmark_networks/NN_result/' #  Where to export network results

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


# train method & step
def train(denoiseNN, noiseEEG_train, EEG_train, noiseEEG_test, EEG_test, epochs):
    best_test_loss = 1000
    writer = SummaryWriter(log_dir=result_location, flush_secs=60)
    # data
    data_train = TensorDataset(noiseEEG_train, EEG_train)
    data_loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=False)
    data_test = TensorDataset(noiseEEG_test, EEG_test)
    data_loader_test = DataLoader(data_test, batch_size=batch_size, shuffle=False)

    # model
    denoiseNN = denoiseNN.cuda()

    for epoch in range(epochs):
        start = time.time()

        # initialize  loss value for every epoch
        train_loss, test_loss = 0, 0

        with tqdm(total=N_batch, position=0, leave=True) as pbar:
            denoiseNN.train()
            for _, (inputs, labels) in enumerate(data_loader_train):

                inputs = inputs.cuda()
                labels = labels.cuda()

                outputs = denoiseNN(inputs)

                m_loss = denoise_loss_mse(outputs, labels)
                train_loss += m_loss.item()

                denoise_optimizer.zero_grad()
                m_loss.backward()  # backward
                denoise_optimizer.step()  # optimizer

                pbar.update()
            pbar.close()
        train_loss = train_loss / float(N_batch)
        writer.add_scalar('data/train_loss', train_loss, epoch)

        # calculate mse loss for test set

        with tqdm(total=int(EEG_test.shape[0] / batch_size), position=0, leave=True) as pbar:
            denoiseNN.eval()
            for _, (inputs, labels) in enumerate(data_loader_test):

                inputs = inputs.cuda()
                labels = labels.cuda()

                outputs = denoiseNN(inputs)

                m_loss = denoise_loss_mse(outputs, labels)
                test_loss += m_loss.item()

                pbar.update()
            pbar.close()
        test_loss = test_loss / float(EEG_test.shape[0] / batch_size)
        writer.add_scalar('data/test_loss', test_loss, epoch)

        if best_test_loss > test_loss:
            torch.save(denoiseNN.state_dict(), "your path/EEG_Trans/" + "best" + ".pth")
            best_test_loss = test_loss
        if epoch % 10 == 9:
            torch.save(denoiseNN.state_dict(), "your path/EEG_Trans/" + "EPOCH" + str(epoch) + ".pth")
        print('\nEpoch #: {}/{}, Time taken: {} secs,\n  Losses: train_mse= {},test_mse={}'\
                 .format(epoch+1, epochs, time.time()-start, train_loss,  test_loss))


if __name__ == '__main__':

    # Import data


    EEG_all = np.load('your path/data/EEG_all_epochs.npy')
    noise_all = np.load('your path/data/EMG_all_epochs.npy')
    noiseEEG_train, EEG_train, noiseEEG_test, EEG_test, test_std_VALUE = data_prepare(EEG_all, noise_all, combin_num,
                                                                                      train_num, validation_num)
    datanum = int(EEG_all.shape[1])
    N_batch = int(EEG_train.shape[0] / batch_size)

    noiseEEG_train = torch.tensor(noiseEEG_train)
    EEG_train = torch.tensor(EEG_train)
    noiseEEG_test = torch.tensor(noiseEEG_test)
    EEG_test = torch.tensor(EEG_test)

    # Import network

    if denoise_network == 'DeT':
        denoiseNN = DeT(seq_len=512, patch_len=64, depth=6, heads=1)

    elif denoise_network == 'Simple_CNN':
         denoiseNN = simple_CNN(batch_size, datanum)

    elif denoise_network == 'simple_CNN_pro':
         denoiseNN = simple_CNN_pro(batch_size, datanum)

    elif denoise_network == 'complex_CNN':
         denoiseNN = complex_CNN(batch_size, datanum)

    elif denoise_network == 'RNN_lstm':
         denoiseNN = RNN_lstm(batch_size, datanum)

    else:
        print('NN name arror')

    # optimizer adjust parameter

    # rmsp = torch.optim.RMSprop(denoiseNN.parameters(), lr=0.0001, alpha=0.9)
    adam = torch.optim.Adam(denoiseNN.parameters(), lr=0.00005, betas=(0.5, 0.9), eps=1e-08)
    # sgd = torch.optim.SGD(denoiseNN.parameters(), lr=0.0002, momentum=0.9, weight_decay=0, nesterov=False)

    denoise_optimizer = adam

    torch.set_default_tensor_type('torch.DoubleTensor')
    denoiseNN = denoiseNN.double()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    denoiseNN = nn.DataParallel(denoiseNN)
    torch.cuda.empty_cache()
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = True

    print('---------- Networks initialized -------------')
    utils.print_network(denoiseNN)
    print('-----------------------------------------------')

    train(denoiseNN, noiseEEG_train, EEG_train, noiseEEG_test, EEG_test, epochs)

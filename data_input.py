import numpy as np
import math
"""
Author: wolider wong
Date: 2024-1-14
Description: EEG, EOG, EMG data preprocessing to form the dataset for training
cite: EEGDnet: Fusing non-local and local self-similarity for EEG signal denoising with transformer
"""

# Author: woldier wong
# The code here not only include data importing, but also data standardization and the generation of analog noise signals

def RMS(x: np.ndarray):
    """
    Root Mean Squared (RMS)

    :param x: input
    :return:
    """
    x2 = x ** 2  # x^2
    sum_s2 = np.sum(x2, axis=-1, keepdims=True)  # sum
    return (sum_s2 / x.shape[-1]) ** 0.5


def compute_noise_signal(x: np.ndarray, n: np.ndarray, snr: np.ndarray):
    """
    λ is a hyperparameter to control the signal-to-noise ratio (SNR) in the contaminated EEG signal y
    SNR = 10 log( RMS(x) / RMS(λ · n) )

    SNR = 10 log ( RMS(x) / ( λ · RMS(n) )  )

    (SNR / 10 ) ** 10 = RMS(x) / ( λ · RMS(n) )

    y = x + λ · n
    :param x: noise-free signal
    :param n: noise signal
    :param snr:
    :return:
    """
    lamda = RMS(x) / ((10 ** (snr / 10)) * RMS(n))
    return x + lamda * n


def normalize(x: np.ndarray, y: np.ndarray, mean_norm=False):
    """
    In order to facilitate the learning procedure, we normalized the input contaminated EEG segment and the ground-truth
    EEG segment by dividing the standard deviation of contaminated EEG segment according to
    x_bar = x / std(y)
    y_bar = y / std(y)
    :param x: noise-free signal
    :param y: contaminated signal
    :param mean_norm: bool , default false  . If true, will norm mean to 0
    :return:
    """
    mean = y.mean() if mean_norm else 0
    std = y.std(axis=-1, keepdims=True)
    x_bar = (x - mean) / std
    y_bar = (y - mean) / std
    return x_bar, y_bar, std


def data_prepare(EEG_all: np.ndarray, noise_all: np.ndarray, combin_num: int, train_num: int, test_num: int):
    # The code here not only include data importing,
    # but also data standardization and the generation of analog noise signals

    # First of all, if we just divide the data into training set and test set according to train_num,test_num,
    # then the coverage of the samples in the training set and test set may not be comprehensive,
    # because we should do a disruptive operation before dividing the data.

    # a random seed to 109(this number can be chosen at random, the realization of the choice of 109 just to have a good feeling about the number),
    # to ensure that each time the random result is the same
    np.random.seed(109)
    # disruptive element
    # disorder the elements of an array
    np.random.shuffle(EEG_all)
    np.random.shuffle(noise_all)

    # Get x, and n for the training and test sets
    eeg_train, eeg_test = EEG_all[0:train_num, :], EEG_all[train_num:train_num + test_num, :]
    noise_train, noise_test = noise_all[0:train_num, :], noise_all[train_num:train_num + test_num, :]

    # Repeat the dataset combin_num times to accumulate noise of different intensities.
    # shape [train_num * combin_num, L] , [test_num * combin_num, L]
    eeg_train, eeg_test = np.repeat(eeg_train, combin_num, axis=0), np.repeat(eeg_test, combin_num, axis=0)
    noise_train, noise_test = np.repeat(noise_train, combin_num, axis=0), np.repeat(noise_test, combin_num, axis=0)

    #################################  simulate noise signal of training set  ##############################

    # create random number between -7dB ~ 2dB
    snr_table = np.linspace(-7, -7 + combin_num, combin_num)  # a shape of [combin_num]
    snr_table = snr_table.reshape((1, -1))  # reshape to [1, combin_num]
    num_table = np.zeros((train_num, 1))  # reshape to [train_num, 1]
    snr_table = snr_table + num_table  # broadcast to [train_num, combin_num]
    snr_table = snr_table.reshape((-1, 1))  # match samples [train_num * combin_num, 1]
    eeg_train_y = compute_noise_signal(eeg_train, noise_train, snr_table)

    # normalize
    EEG_train_end_standard, noiseEEG_train_end_standard, EEG_trian_std = normalize(eeg_train, eeg_train_y)

    #################################  simulate noise signal of test  ##############################
    snr_table = np.linspace(-7, -7 + combin_num, combin_num)  # a shape of [combin_num]
    snr_table = snr_table.reshape((1, -1))  # reshape to [1, combin_num]
    num_table = np.zeros((test_num, 1))  # reshape to [test_num, 1]
    snr_table = snr_table + num_table  # broadcast to [test_num, combin_num]
    snr_table = snr_table.reshape((-1, 1))  # match samples [test_num * combin_num, 1]
    eeg_test_y = compute_noise_signal(eeg_test, noise_test, snr_table)
    EEG_test_end_standard, noiseEEG_test_end_standard, std_VALUE = normalize(eeg_test, eeg_test_y)

    return noiseEEG_train_end_standard, EEG_train_end_standard, noiseEEG_test_end_standard, EEG_test_end_standard, std_VALUE


if __name__ == "__main__":
    EEG_all = np.load('./data/EEG_all_epochs.npy')
    noise_all = np.load('./data/EMG_all_epochs.npy')
    noiseEEG_train, EEG_train, noiseEEG_test, EEG_test, test_std_VALUE = data_prepare(EEG_all, noise_all,
                                                                                      10,
                                                                                      4000, 100)

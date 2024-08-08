""" file for small helper functions """

import string
from functools import partial
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import pickle5 as pickle
from termcolor import colored
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import time
import json
import random

from nameEthnicityDataset import NameEthnicityDataset

torch.manual_seed(0)
random.seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def custom_collate(batch):
    """ adds custom dataloader feature: batch padding for the sample-batch (the batch containing the one-hot-enc. names)

    :param batch: three batches -> non-padded sample-batch, target-batch, non-padded sample-batch (again)
    :return torch.Tensor: padded sample-batch, target-batch, non-padded sample-batch
    """

    batch_size = len(batch)

    sample_batch, target_batch, non_padded_batch = [], [], []
    for sample, target, non_padded_sample in batch:

        sample_batch.append(sample)
        target_batch.append(target)

        # non_padded_batch is the original batch, which is not getting padded so it can be converted back to string
        non_padded_batch.append(non_padded_sample)

    padded_batch = pad_sequence(sample_batch, batch_first=True)
    padded_to = list(padded_batch.size())[1]
    padded_batch = padded_batch.reshape(len(sample_batch), padded_to, 1) # change 200 to 1 if not using char2vec        

    return padded_batch, torch.cat(target_batch, dim=0).reshape(len(sample_batch), target_batch[0].size(0)), non_padded_batch


def create_dataloader(dataset_path: str="", test_size: float=0.01, val_size: float=0.01, batch_size: int=32, class_amount: int=10, \
                                                                            augmentation: float=0.0):
    """ create three dataloader (train, test, validation)

    :param str dataset_path: path to dataset
    :param float test_size/val_size: test-/validation-percentage of dataset
    :param int batch_size: batch-size
    :return torch.Dataloader: train-, test- and val-dataloader
    """

    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)
    print(len(dataset))
    test_size = int(np.round(len(dataset)*test_size))
    val_size = int(np.round(len(dataset)*val_size))

    train_set, test_set, validation_set = dataset[(test_size+val_size):], dataset[:test_size], dataset[test_size:(test_size+val_size)]

    train_set = NameEthnicityDataset(dataset=train_set, class_amount=class_amount, augmentation=augmentation)
    test_set = NameEthnicityDataset(dataset=test_set, class_amount=class_amount, augmentation=0.0)
    val_set = NameEthnicityDataset(dataset=validation_set, class_amount=class_amount, augmentation=0.0)

    train_dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True,
        collate_fn=custom_collate
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_set,
        batch_size=int(batch_size),
        num_workers=0,
        shuffle=True,
        collate_fn=custom_collate

    )
    test_dataloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=int(batch_size),
        num_workers=0,
        shuffle=True,
        collate_fn=custom_collate
    )

    return train_dataloader, val_dataloader, test_dataloader


def show_progress(epochs: int, epoch: int, train_loss: float, train_accuracy: float, val_loss: float, val_accuracy: float, colored: bool=True):
    """ print training stats
    
    :param int epochs: amount of total epochs
    :param int epoch: current epoch
    :param float train_loss/train_accuracy: train-loss, train-accuracy
    :param float val_loss/val_accuracy: validation accuracy/loss
    :return None
    """
    if colored:
        epochs = colored(epoch, "cyan", attrs=["bold"]) + colored("/", "cyan", attrs=["bold"]) + colored(epochs, "cyan", attrs=["bold"])
        train_accuracy = colored(round(train_accuracy, 4), "cyan", attrs=["bold"]) + colored("%", "cyan", attrs=["bold"])
        train_loss = colored(round(train_loss, 6), "cyan", attrs=["bold"])
        val_accuracy = colored(round(val_accuracy, 4), "cyan", attrs=["bold"]) + colored("%", "cyan", attrs=["bold"])
        val_loss = colored(round(val_loss, 6), "cyan", attrs=["bold"])
    else:
        epochs = str(epoch) + "/" + str(epochs)
        train_accuracy = str(round(train_accuracy, 4)) + "%"
        train_loss = str(round(train_loss, 6))
        val_accuracy = str(round(val_accuracy, 4)) + "%"
        val_loss = str(round(val_loss, 6))

    print("epoch {} train_loss: {} - train_acc: {} - val_loss: {} - val_acc: {}".format(epochs, train_loss, train_accuracy, val_loss, val_accuracy), "\n")


def lr_scheduler(optimizer: torch.optim, current_iteration: int=0, warmup_iterations: int=0, lr_end: float=0.001, decay_rate: float=0.99, decay_intervall: int=100) -> None:
    current_iteration += 1
    current_lr = optimizer.param_groups[0]["lr"]

    if current_iteration <= warmup_iterations:
        optimizer.param_groups[0]["lr"] = (current_iteration * lr_end) / warmup_iterations
        # print(" WARMUP", optimizer.param_groups[0]["lr"])

    elif current_iteration > warmup_iterations and current_iteration % decay_intervall == 0:
        optimizer.param_groups[0]["lr"] = current_lr * decay_rate
        # print(" DECAY", optimizer.param_groups[0]["lr"])
    else:
        pass


def onehot_to_string(one_hot_name: list=[]) -> str:
    """ convert one-hot encoded name back to string

    :param list one_hot_name: one-hot enc. name
    :return str: original string-type name
    """

    alphabet = string.ascii_lowercase.strip()

    name = ""
    for one_hot_char in one_hot_name:
        idx = list(one_hot_char).index(1)

        if idx == 26:
            name += " "
        elif idx == 27:
            name += "-"
        else:
            name += alphabet[idx]

    return name


def string_to_onehot(string_name: str="") -> list:
    """ create one-hot encoded name

    :param str name: name to encode
    :return list: list of all one-hot encoded letters of name
    """

    alphabet = list(string.ascii_lowercase.strip()) + [" ", "-"]

    full_name_onehot = []
    for char in string_name:
        char_idx = alphabet.index(char)

        one_hot_char = np.zeros((28))
        one_hot_char[char_idx] = 1

        full_name_onehot.append(one_hot_char)
    
    return full_name_onehot


def char_indices_to_string(char_indices: list=[str]) -> str:
    """ takes a list with indices from 0 - 27 (alphabet + " " + "-") and converts them to a string

        :param str char_indices: list containing the indices of the chars
        :return str: decoded name
    """

    alphabet = list(string.ascii_lowercase.strip()) + [" ", "-"]
    name = ""
    for idx in char_indices:
        if int(idx) == 0:
            pass
        else:
            name += alphabet[int(idx) - 1]
    
    return name


def init_xavier_weights(m):
    """ initializes model parameters with xavier-initialization

    :param m: model parameters
    """
    if isinstance(m, nn.RNN):
        nn.init.xavier_uniform_(m.weight_hh_l0.data)


def load_json(file_path: str) -> dict:
    with open(file_path, "r") as f:
        return json.load(f)


def write_json(file_path: str, content: dict) -> None:
    with open(file_path, "w") as f:
            json.dump(content, f, indent=4)


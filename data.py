"""
LOAD DATA from file.
"""

# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
import os
import torch
import numpy as np
import torchvision.datasets as datasets
from mnist import MNIST
from mnist import FashionMNIST
from torchvision.datasets import CIFAR10
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import cv2

##
def load_data(opt):
    """ Load Data

    Args:
        opt ([type]): Argument Parser

    Raises:
        IOError: Cannot Load Dataset

    Returns:
        [type]: dataloader
    """

    ##
    # LOAD DATA SET
    if opt.dataroot == '':
        opt.dataroot = 'data/{}'.format(opt.dataset)

    if opt.dataset in ['mnist']:
        opt.anomaly_class = int(opt.anomaly_class)

        splits = ['train', 'test']
        drop_last_batch = {'train': True, 'test': True}
        shuffle = {'train': True, 'test': True}
        batch_size = {'train':opt.train_batch,'test':opt.test_batch}

        transform = transforms.Compose(
            [
                transforms.Resize(opt.isize),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]
        )

        dataset = {}
        dataset['train'] = MNIST(root='./data', train=True, download=True, transform=transform)
        dataset['test'] = MNIST(root='./data', train=False, download=True, transform=transform)
        # print(dataset['train'])
        a,b,c,d=get_mnist_anomaly_dataset(dataset['train'].train_data,
                                  dataset['train'].train_labels,
                                  dataset['test'].test_data,
                                  dataset['test'].test_labels,
                                  opt.anomaly_class, -1)
        # print(a)
        dataset['train'].train_data=a
        dataset['train'].train_labels=b
        dataset['test'].test_data=c
        dataset['test'].test_labels = d

        dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                     batch_size=opt.train_batch,
                                                     shuffle=shuffle[x],
                                                     num_workers=opt.workers,
                                                     drop_last=drop_last_batch[x]) for x in splits}
        return dataloader

def get_mnist_anomaly_dataset(trn_img, trn_lbl, tst_img, tst_lbl, abn_cls_idx=8, manualseed=-1):
    """[summary]

    Arguments:
        trn_img {np.array} -- Training images
        trn_lbl {np.array} -- Training labels
        tst_img {np.array} -- Test     images
        tst_lbl {np.array} -- Test     labels

    Keyword Arguments:
        abn_cls_idx {int} -- Anomalous class index (default: {0})

    Returns:
        [np.array] -- New training-test images and labels.
    """
    # --
    # Find normal abnormal indexes.
    nrm_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() != abn_cls_idx)[0])
    abn_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() == abn_cls_idx)[0])
    nrm_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() != abn_cls_idx)[0])
    abn_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() == abn_cls_idx)[0])
    # --
    # Find normal and abnormal images
    nrm_trn_img = trn_img[nrm_trn_idx]    # Normal training images
    abn_trn_img = trn_img[abn_trn_idx]    # Abnormal training images.
    nrm_tst_img = tst_img[nrm_tst_idx]    # Normal training images
    abn_tst_img = tst_img[abn_tst_idx]    # Abnormal training images.

    # --
    # Find normal and abnormal labels.
    nrm_trn_lbl = trn_lbl[nrm_trn_idx]    # Normal training labels
    abn_trn_lbl = trn_lbl[abn_trn_idx]    # Abnormal training labels.
    nrm_tst_lbl = tst_lbl[nrm_tst_idx]    # Normal training labels
    abn_tst_lbl = tst_lbl[abn_tst_idx]    # Abnormal training labels.

    # --
    # Assign labels to normal (0) and abnormals (1)
    nrm_trn_lbl[:] = 0
    nrm_tst_lbl[:] = 0
    abn_trn_lbl[:] = 1
    abn_tst_lbl[:] = 1

    abn_trn_img = abn_trn_img
    abn_trn_lbl = abn_trn_lbl
    train_idx = np.arange(len(abn_trn_lbl))
    nrm_trn_len = int(len(train_idx))

    nrm_tst_img = torch.cat((nrm_tst_img, abn_tst_img), dim=0)
    abn_tst_lbl = torch.cat((nrm_tst_lbl, abn_tst_lbl), dim=0)
    test_idx = np.arange(len(abn_tst_lbl))
    test_idx = np.random.permutation(test_idx)
    nrm_tst_idx = test_idx[:nrm_trn_len]
    # print('nrm_tst_idx:',nrm_tst_idx)

    
    nrm_tst_img = nrm_tst_img[nrm_tst_idx]
    nrm_tst_lbl = abn_tst_lbl[nrm_tst_idx]
    # --
    if manualseed != -1:
        # Random seed.
        # Concatenate the original train and test sets.
        nrm_img = torch.cat((nrm_trn_img, nrm_tst_img), dim=0)
        nrm_lbl = torch.cat((nrm_trn_lbl, nrm_tst_lbl), dim=0)
        abn_img = torch.cat((abn_trn_img, abn_tst_img), dim=0)
        abn_lbl = torch.cat((abn_trn_lbl, abn_tst_lbl), dim=0)

        # Split the normal data into the new train and tests.
        idx = np.arange(len(nrm_lbl))
        np.random.seed(manualseed)
        np.random.shuffle(idx)

        nrm_trn_len = int(len(idx) * 0.80)
        nrm_trn_idx = idx[:nrm_trn_len]
        nrm_tst_idx = idx[nrm_trn_len:]

        nrm_trn_img = nrm_img[nrm_trn_idx]
        nrm_trn_lbl = nrm_lbl[nrm_trn_idx]
        nrm_tst_img = nrm_img[nrm_tst_idx]
        nrm_tst_lbl = nrm_lbl[nrm_tst_idx]

    new_trn_img = abn_trn_img.clone()
    new_trn_lbl = abn_trn_lbl.clone()
    new_tst_img = nrm_tst_img.clone()
    new_tst_lbl = nrm_tst_lbl.clone()

    return new_trn_img, new_trn_lbl, new_tst_img, new_tst_lbl
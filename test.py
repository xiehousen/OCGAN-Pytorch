'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random
from sklearn.metrics import roc_auc_score
import torch

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torchvision

from utils import Bar, Logger, AverageMeter, mkdir_p, savefig
import math

from PIL import Image
import matplotlib.pyplot as plt
from torch.autograd import Variable

from ocgan.networks import *
 
from torchvision.utils import save_image

from data import load_data




parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='mnist', type=str)
parser.add_argument('--dataroot', default='./data', type=str)
parser.add_argument('--anomaly_class', default='9', type=int)
parser.add_argument('--isize', default='28', type=int)

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train_batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test_batch', default=64, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[40, 100],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}



# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy



dataloader = load_data(args)
num_classes = 2

Tensor = torch.cuda.FloatTensor
enc = get_encoder().cuda()
dec = get_decoder().cuda()

checkpoint_en = torch.load('./checkpoint/enc_model.pth.tar')
enc.load_state_dict(checkpoint_en['state_dict'])

checkpoint_de = torch.load('./checkpoint/dec_model.pth.tar')
dec.load_state_dict(checkpoint_de['state_dict'])

if not os.path.exists('./result/0009/test_dc_fake-9'):
    os.mkdir('./result/0009/test_dc_fake-9')
if not os.path.exists('./result/0009/test_dc_real-9'):
    os.mkdir('./result/0009/test_dc_real-9')


for batch_idx, (inputs, targets) in enumerate(dataloader['test']):
     
    if use_cuda:
        inputs, targets = inputs.cuda(), targets.cuda()
        # with torch.no_grad():
    inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
       
    #update class
    '''
    imput_show = inputs[1,...]
    imput_show = imput_show[0,...]
    # label_show = targets[1,...]

    print('mmmmm',imput_show.squeeze().shape,targets.shape)
    plt.figure(1)
    plt.imshow(imput_show.squeeze().cpu().detach().numpy())
    plt.show()
    '''
  
    # compute output
    recon = dec(enc(inputs))
    '''
    recon_show = recon[1,...]
    recon_show = recon_show[0,...]   
    plt.figure(2)  
    plt.imshow(recon_show.squeeze().cpu().detach().numpy())
    plt.show()
    '''
    scores = torch.mean(torch.pow((inputs - recon), 2),dim=[1,2,3])
    prec1 = roc_auc_score(targets.cpu().detach().numpy(), -scores.cpu().detach().numpy())
    print('\nBatch: {0:d} =====  auc:{1:.2e}' .format(batch_idx,prec1))
    pic = recon.cpu().data
    img = inputs.cpu().data
    save_image(pic, './result/0009/test_dc_fake-9/fake_0{}.png'.format(batch_idx))
    save_image(img, './result/0009/test_dc_real-9/real_0{}.png'.format(batch_idx))
print('Saving pic... ')



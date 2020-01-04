'''
Copyright (c) Housen Xie, 2019
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
parser.add_argument('--anomaly_class', default='1', type=int)
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
parser.add_argument('--test_batch', default=128, type=int, metavar='N',
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

parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
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

best_acc = 0

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data
    print('==> Preparing dataset' )
    dataloader = load_data(args)
    Tensor = torch.cuda.FloatTensor

    print("==> creating model")
    title = 'Pytorch-OCGAN'

    enc = get_encoder().cuda()
    dec = get_decoder().cuda()
    disc_v = get_disc_visual().cuda()
    disc_l = get_disc_latent().cuda() 
    cl = get_classifier().cuda()

    #load origianal weights
    disc_v.apply(weights_init)
    cl.apply(weights_init)
    enc.apply(weights_init)
    dec.apply(weights_init)
    disc_l.apply(weights_init)




    model = torch.nn.DataParallel(enc).cuda()
    cudnn.benchmark = True
    print('  enc     Total params: %.2fM' % (sum(p.numel() for p in enc.parameters())/1000000.0))
    print('  dec     Total params: %.2fM' % (sum(p.numel() for p in dec.parameters())/1000000.0))
    print('  disc_v  Total params: %.2fM' % (sum(p.numel() for p in disc_v.parameters())/1000000.0))
    print('  disc_l  Total params: %.2fM' % (sum(p.numel() for p in disc_l.parameters())/1000000.0))
    print('  cl      Total params: %.2fM' % (sum(p.numel() for p in cl.parameters())/1000000.0))

#Loss Loss Loss Loss Loss Loss Loss
    print("==> creating optimizer")

    criterion_ce = torch.nn.BCELoss(size_average=True).cuda()
    criterion_ae = nn.MSELoss(size_average=True).cuda()

    l2_int=torch.empty(size=(args.train_batch, 288,1,1), dtype=torch.float32)

    optimizer_en = optim.Adam(enc.parameters(), lr=args.lr, betas=(0.9, 0.99))
    optimizer_de = optim.Adam(dec.parameters(), lr=args.lr, betas=(0.9, 0.99))
    optimizer_dl = optim.Adam(disc_l.parameters(), lr=args.lr, betas=(0.9, 0.99))
    optimizer_dv = optim.Adam(disc_v.parameters(), lr=args.lr, betas=(0.9, 0.99))
    optimizer_c   = optim.Adam(cl.parameters(), lr=args.lr, betas=(0.9, 0.99))
    optimizer_l2 = optim.Adam([{'params':l2_int}], lr=args.lr, betas=(0.9, 0.99))


    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)

        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Acc.'])



    # Train and val
    for epoch in range(start_epoch, args.epochs):
        # adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        # model = optimize_fore()
        if epoch < 20:
        
            train_loss_ae = train_ae(args,dataloader['train'], enc, dec, optimizer_en, optimizer_de,criterion_ae, epoch, use_cuda)
            test_acc = test(args,dataloader['test'],  enc, dec,cl,disc_l,disc_v, epoch, use_cuda)
        else:
        
            train_loss = train(args,dataloader['train'], enc, dec,cl,disc_l,disc_v,
                                            optimizer_en, optimizer_de,optimizer_c,optimizer_dl,optimizer_dv,optimizer_l2,
                                            criterion_ae, criterion_ce, 
                                            Tensor,epoch, use_cuda
                                           )
            test_acc = test(args,dataloader['test'],  enc, dec,cl,disc_l,disc_v, epoch, use_cuda)

        # append logger file

            logger.append([state['lr'], train_loss,test_acc])
        

            # save model
            is_best = train_loss < best_acc
            best_acc = min(train_loss, best_acc)
            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': enc.state_dict(),
                    'loss': train_loss,
                    'best_loss': best_acc,
                }, is_best, checkpoint=args.checkpoint,filename='enc_model.pth.tar')
            
            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': dec.state_dict(),
                    'loss': train_loss,
                    'best_loss': best_acc,
                }, is_best, checkpoint=args.checkpoint,filename='dec_model.pth.tar')
            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': cl.state_dict(),
                    'loss': train_loss,
                    'best_loss': best_acc,
                }, is_best, checkpoint=args.checkpoint,filename='cl_model.pth.tar')
            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': disc_l.state_dict(),
                    'loss': train_loss,
                    'best_loss': best_acc,
                }, is_best, checkpoint=args.checkpoint,filename='disc_l_model.pth.tar')
            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': disc_v.state_dict(),
                    'loss': train_loss,
                    'best_loss': best_acc,
                }, is_best, checkpoint=args.checkpoint,filename='disc_v_model.pth.tar')                

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)
def train_ae(args,trainloader, enc, dec, optimizer_en, optimizer_de, criterion, epoch, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()


    end = time.time()
    

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        
        u = np.random.uniform(-1, 1, (args.train_batch, 288, 1, 1))   
        l2 = torch.from_numpy(u).float()

        n = torch.randn(args.train_batch, 1, 28, 28).cuda()
        l1 = enc(inputs + n)
        del1 = dec(l1)
  

        loss = criterion(del1,inputs)


        losses.update(loss.item(), inputs.size(0))


        enc.zero_grad()
        dec.zero_grad()

        loss.backward()

        optimizer_en.step()
        optimizer_de.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f}  '.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    )
        bar.next()
    bar.finish()
    return losses.avg

def train(args,trainloader,enc, dec,cl,disc_l,disc_v,
                    optimizer_en, optimizer_de,optimizer_c,optimizer_dl,optimizer_dv,optimizer_l2,
                    criterion_ae, criterion_ce,Tensor, epoch, use_cuda):
    # switch to train mode
    enc.train()
    dec.train()
    cl.train()
    disc_l.train()
    disc_v.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()


    end = time.time()
    

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)


        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        
        #update class
        '''
        imput_show = inputs[1,...]
        imput_show = imput_show[0,...]
        label_show = targets[1,...]

        print('mmmmm',inputs.shape,targets.shape)
        plt.figure()
        plt.imshow(imput_show.cpu())
        plt.show()
        '''
        u = np.random.uniform(-1, 1, (args.train_batch, 288, 1, 1))   
        l2 = torch.from_numpy(u).float().cuda()

        dec_l2 = dec(l2)
        n = torch.randn(args.train_batch, 1, 28, 28).cuda()
        l1 = enc(inputs + n)
        logits_C_l1 = cl(dec(l1))
        logits_C_l2 = cl(dec_l2)

        valid_logits_C_l1 = Variable(Tensor(logits_C_l1.shape[0], 1).fill_(1.0), requires_grad=False)
        fake_logits_C_l2 = Variable(Tensor(logits_C_l2.shape[0], 1).fill_(0.0), requires_grad=False)

        loss_cl_l1 = criterion_ce(logits_C_l1,valid_logits_C_l1)
        loss_cl_l2 = criterion_ce(logits_C_l2,fake_logits_C_l2)

        loss_cl = (loss_cl_l1 + loss_cl_l2 ) / 2

        cl.zero_grad()
        loss_cl.backward(retain_graph=True)
        optimizer_c.step()


        disc_l_l1 = l1.view(l1.size(0),32,3,3)
        disc_l.zero_grad()
        logits_Dl_l1 = disc_l(disc_l_l1)
        logits_Dl_l2 = disc_l(l2)
        dl_logits_DL_l1 = Variable(Tensor(logits_Dl_l1.shape[0], 1).fill_(0.0), requires_grad=False)
        dl_logits_DL_l2 = Variable(Tensor(logits_Dl_l2.shape[0], 1).fill_(1.0), requires_grad=False)
 
        loss_dl_1 = criterion_ce(logits_Dl_l1 , dl_logits_DL_l1)
        loss_dl_2 = criterion_ce(logits_Dl_l2 , dl_logits_DL_l2)
        loss_dl = (loss_dl_1 + loss_dl_2) / 2

        
        loss_dl.backward(retain_graph=True)
        optimizer_dl.step()


        logits_Dv_X = disc_v(inputs)
        logits_Dv_l2 = disc_v(dec(l2))


        dv_logits_Dv_X = Variable(Tensor(logits_Dv_X.shape[0], 1).fill_(1.0), requires_grad=False)
        dv_logits_Dv_l2 = Variable(Tensor(logits_Dv_l2.shape[0], 1).fill_(0.0), requires_grad=False)
        
        loss_dv_1 = criterion_ce(logits_Dv_X,dv_logits_Dv_X)
        loss_dv_2 = criterion_ce(logits_Dv_l2,dv_logits_Dv_l2)
        loss_dv = (loss_dv_1 + loss_dv_2) / 2

        disc_v.zero_grad()
        loss_dv.backward()
        optimizer_dv.step()



        for i in range(5):
            logits_C_l2_mine = cl(dec(l2))
            zeros_logits_C_l2_mine = Variable(Tensor(logits_C_l2_mine.shape[0], 1).fill_(0.0), requires_grad=False)
            loss_C_l2_mine = criterion_ce(logits_C_l2_mine,zeros_logits_C_l2_mine)
            optimizer_l2.zero_grad()
            loss_C_l2_mine.backward()
            optimizer_l2.step()

        ######  update ae 
        out_gv1 = disc_v(dec(l2))
        Xh = dec(l1)
        loss_mse = criterion_ae(Xh,inputs)
        ones_logits_Dl_l1 = Variable(Tensor(logits_Dl_l1.shape[0], 1).fill_(1.0), requires_grad=False)
 
        loss_AE_l = criterion_ce(logits_Dl_l1,ones_logits_Dl_l1)

        logits_Dv_l2_mine = disc_v(dec_l2)
        ones_logits_Dv_l2_mine = Variable(Tensor(logits_Dv_l2_mine.shape[0], 1).fill_(1.0), requires_grad=False)
        loss_ae_v = criterion_ce(logits_Dv_l2_mine,ones_logits_Dv_l2_mine)

        loss_ae_all = 10*loss_mse + loss_ae_v + loss_AE_l

        enc.zero_grad()
        dec.zero_grad()
        loss_ae_all.backward()
        optimizer_en.step()
        optimizer_de.step()

        losses.update(loss_ae_all.item(), inputs.size(0))



       
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f}  '.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,

                    )
        bar.next()
    bar.finish()
    
    #save images during training time
    if epoch % 5 == 0:
        recon = dec(enc(inputs))
        recon = recon.cpu().data
        inputs = inputs.cpu().data
        if not os.path.exists('./result/0000/train_dc_fake-1'):
            os.mkdir('./result/0000/train_dc_fake-1')
        if not os.path.exists('./result/0000/train_dc_real-1'):
            os.mkdir('./result/0000/train_dc_real-1')
        save_image(recon, './result/0000/train_dc_fake-1/fake_0{}.png'.format(epoch))
        save_image(inputs, './result/0000/train_dc_real-1/real_0{}.png'.format(epoch))  
    return losses.avg

def test(args,testloader, enc, dec,cl,disc_l,disc_v, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()


    # switch to evaluate mode
    enc.eval()
    dec.eval()
    cl.eval()
    disc_l.eval()
    disc_v.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))

    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        # with torch.no_grad():
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        # compute output
        recon = dec(enc(inputs))       
        scores = torch.mean(torch.pow((inputs - recon), 2),dim=[1,2,3])
        prec1 = roc_auc_score(targets.cpu().detach().numpy(), -scores.cpu().detach().numpy())

        top1.update(prec1, inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | top1: {top1: .4f} '.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    top1=top1.avg,
                    )
        bar.next()
    bar.finish()
    return top1.avg

def save_checkpoint(state, is_best, checkpoint,filename):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()

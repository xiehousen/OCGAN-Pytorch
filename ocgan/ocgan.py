import numpy as np
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
from collections import OrderedDict
import os
import time
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils

import torch.nn.functional as F
def weights_init(mod):
    """
    Custom weights initialization called on netG, netD and netE
    :param m:
    :return:
    """
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        mod.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)
class get_encoder(nn.Module):
    """
    DCGAN ENCODER NETWORK
    """

    def __init__(self):
        super(get_encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=3//2)
        # self.activation = nn.Tanh()
        self.conv2 = nn.Conv2d(32, 32, 3, padding=3//2)
        self.batch_norm_1 = nn.BatchNorm2d(32)
        self.maxpooling_1 = nn.MaxPool2d(3, stride=2,padding=3//2)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=3//2)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=3//2)
        self.batch_norm_2 = nn.BatchNorm2d(64)
        self.maxpooling_2 = nn.MaxPool2d(3, stride=2,padding=3//2)

        self.conv5 = nn.Conv2d(64, 32, 3)
        self.conv6 = nn.Conv2d(32, 32, 3)

    def forward(self, input): 
        # print('input:',input.shape)  
        output = F.tanh(self.conv1(input))  
        # print('output1:',output.shape)
        output = F.tanh(self.conv2(output))
        output = self.batch_norm_1(output)
        output = self.maxpooling_1(output)
        # print('output2:',output.shape)
        output = F.tanh(self.conv3(output))
        # print('output3:',output.shape)
        output = F.tanh(self.conv4(output))
        output = self.batch_norm_2(output)
        output = self.maxpooling_2(output)
        # print('output4:',output.shape)

        output = F.tanh(self.conv5(output))
        # print('output5:',output.shape)
        output = F.tanh(self.conv6(output)) 
        # print('output6:',output.shape)
        # output = output.permute((0, 2, 3, 1))
        # output = output.contiguous().view(-1, 4 * 4 * 64) 
        output = output.view(output.size(0),-1)
        # print('output11:',output.shape)

        return output

##


class get_decoder(nn.Module):
    """
    DCGAN DECODER NETWORK
    """
    def __init__(self):
        super(get_decoder, self).__init__()
        self.conv1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = nn.ConvTranspose2d(32, 32, 3, padding=3//2)
        self.activation = nn.Tanh()
        self.conv3 = nn.ConvTranspose2d(32, 32, 3, padding=3//2)
        self.batch_norm_1 = nn.BatchNorm2d(32)  

        self.conv4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv5 = nn.ConvTranspose2d(32, 64, 3, padding=3//2)
        self.conv6 = nn.ConvTranspose2d(64, 64, 3, padding=3//2)
        self.batch_norm_2 = nn.BatchNorm2d(64)    
        
        self.conv7 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv8 = nn.ConvTranspose2d(64, 64, 3)
        self.conv9 = nn.ConvTranspose2d(64, 64, 3)
        self.batch_norm_3 = nn.BatchNorm2d(64)    

        self.conv10 = nn.ConvTranspose2d(64, 1, 3, padding=3//2)    

    def forward(self, input):
        # print('l2_input:',input.size(0))
        output = input.view(input.size(0),32,3,3)
        # print('l2_reshape:',output.shape)
        output = self.conv1(output) 
        # print('l2_output1:',output.shape)
        output = F.tanh(self.conv2(output))
        # print('l2_output2:',output.shape)
        output = F.tanh(self.conv3(output))
        output = self.batch_norm_1(output)
        # print('l2_output3:',output.shape)

        output = self.conv4(output)
        # print('l2_output4:',output.shape)
        output = F.tanh(self.conv5(output))
        # print('l2_output5:'/,output.shape)
        output = F.tanh(self.conv6(output))
        output = self.batch_norm_2(output)
        # print('l2_output6:',output.shape)

        output = self.conv7(output)
        # print('l2_output7:',output.shape)
        output = F.tanh(self.conv8(output)) 
        # print('l2_output8:',output.shape)
        output = F.tanh(self.conv9(output))
        output = self.batch_norm_3(output)
        # print('l2_outpu/t9:',output.shape)
        # output = output.permute((0, 2, 3, 1))
        # output = output.contiguous().view(-1, 4 * 4 * 64) 
        output = F.sigmoid(self.conv10(output))   
        # print('l2_output10:',output.shape)
        return output


class get_disc_latent(nn.Module):
    """
    DISCRIMINATOR latent NETWORK
    """

    def __init__(self):
        super(get_disc_latent, self).__init__()
        self.dense_1 = nn.Linear(288, 128)
        self.batch_norm_1 = nn.BatchNorm2d(128)

        self.dense_2 = nn.Linear(128, 64)
        self.batch_norm_2 = nn.BatchNorm2d(64)

        self.dense_3 = nn.Linear(64, 32)
        self.batch_norm_3 = nn.BatchNorm2d(32)

        self.dense_4 = nn.Linear(32, 16)
        self.batch_norm_4 = nn.BatchNorm2d(16)

        self.dense_5 = nn.Linear(16, 1)
    

    def forward(self, input):
        output = self.batch_norm_1(self.dense_1(input))
        output = F.relu(output)

        output = self.batch_norm_2(self.dense_2(output))
        output = F.relu(output)

        output = self.batch_norm_3(self.dense_3(output))
        output = F.relu(output)

        output = self.batch_norm_4(self.dense_4(output))
        output = F.relu(output)

        output = self.dense_5(output)

        return output

class get_disc_visual(nn.Module):
    """
    DISCRIMINATOR vision  NETWORK
    """

    def __init__(self):
        super(get_disc_visual, self).__init__()
        self.conv1 = nn.Conv2d(1,16,5,stride=2,padding=5//2)
        self.batch_norm_1 = nn.BatchNorm2d(16)
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(16,16,5,stride=2,padding=5//2)
        self.batch_norm_2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16,16,5,stride=2,padding=5//2)
        self.batch_norm_3 = nn.BatchNorm2d(16)
                
        self.conv4 = nn.Conv2d(16,1,5,stride=2,padding=5//2)
        self.conv5 = nn.AdaptiveAvgPool2d((None,1))

    def forward(self, input):

        output = self.batch_norm_1(self.conv1(input))
        output = self.leaky_relu(output)

        output = self.batch_norm_2(self.conv2(output))
        output = self.leaky_relu(output)

        output = self.batch_norm_3(self.conv3(output))
        output = self.leaky_relu(output)

        output = self.conv4(output)
        output = self.conv5(output)

        return output


class get_classifier(nn.Module):
    """
    Classfier NETWORK
    """

    def __init__(self):
        super(get_classifier, self).__init__()
        self.conv1 = nn.Conv2d(1,32,5,stride=2,padding=5//2)
        self.batch_norm_1 = nn.BatchNorm2d(16)
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(32,64,5,stride=2,padding=5//2)
        self.batch_norm_2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(64,64,5,stride=2,padding=5//2)
        self.batch_norm_3 = nn.BatchNorm2d(16)
                
        self.conv4 = nn.Conv2d(64,1,5,stride=2,padding=5//2)
        self.conv5 = nn.AdaptiveAvgPool2d((None,1))

    def forward(self, input):

        output = self.batch_norm_1(self.conv1(input))
        output = self.leaky_relu(output)

        output = self.batch_norm_2(self.conv2(output))
        output = self.leaky_relu(output)

        output = self.batch_norm_3(self.conv3(output))
        output = self.leaky_relu(output)

        output = self.conv4(output)
        output = self.conv5(output)

        return output
class OCGAN:
    def __init__(self, opt):
        super(OCGAN, self).__init__()
        self.batchsize = 32
        latent_shape = [288]
        self.enc = get_encoder()
        self.dec = get_decoder()
        self.disc_v = get_disc_visual()
        self.disc_l = get_disc_latent() #original shape 
        self.cl = get_classifier()

        self.disc_v.apply(weights_init)
        self.cl.apply(weights_init)
        self.enc.apply(weights_init)
        self.dec.apply(weights_init)
        self.disc_l.apply(weights_init)

        self.optimizer_en = optim.Adam(self.enc.parameters(), lr=opt.lr, betas=(0.9, 0.99))
        self.optimizer_de = optim.Adam(self.dec.parameters(), lr=opt.lr, betas=(0.9, 0.99))
        self.optimizer_dl = optim.Adam(self.disc_l.parameters(), lr=opt.lr, betas=(0.9, 0.99))
        self.optimizer_dv = optim.Adam(self.disc_v.parameters(), lr=opt.lr, betas=(0.9, 0.99))
        self.optimizer_c   = optim.Adam(self.cl.parameters(), lr=opt.lr, betas=(0.9, 0.99))
        self.optimizer_l2 = optim.Adam([{'params':self.l2}], lr=opt.lr, betas=(0.9, 0.99))

        self.enc.train().cuda()
        self.dec.train().cuda()
        self.disc_v.train().cuda()
        self.disc_l.train().cuda()
        self.cl.train().cuda()

                # Loss Functions
        self.bce_criterion = nn.BCELoss().cuda()
        self.l1l_criterion = nn.L1Loss().cuda()
        self.l2l_criterion = nn.MSELoss().cuda()

        print('Initialize input tensors.')
        self.input = torch.empty(size=(self.batchsize, 1, 28, 28), dtype=torch.float32)
        self.label = torch.empty(size=(self.batchsize,), dtype=torch.float32)
        self.labelf = torch.empty(size=(self.batchsize,), dtype=torch.float32)
        self.gt    = torch.empty(size=(self.batchsize,), dtype=torch.long)
        self.fixed_input = torch.empty(size=(self.batchsize, 1, 28, 28), dtype=torch.float32)
        self.real_label = 1
        self.fake_label = 0
        self.u=None
        self.n=None
        self.l1=None
        self.l2=torch.empty(size=(self.batchsize, 288,1,1), dtype=torch.float32)
        self.del1=None
        self.del2=None
        print('end')

    def update_netd(self):
        """
        Update D network: Ladv = |f(real) - f(fake)|_2
        """

        self.disc_v.zero_grad()
        self.disc_l.zero_grad()
        # --
        self.out_dv0 = self.disc_v(self.del2.detach())
        self.out_dv1 = self.disc_v(self.input)
        self.out_dv = torch.cat([self.out_dv0, self.out_dv1], 0)

        self.labelf.data.resize_(self.batchsize).fill_(self.fake_label)
        self.label.data.resize_(self.batchsize).fill_(self.real_label)
        self.label_dv = torch.cat([self.labelf, self.label], 0)

        self.err_dv_bce = self.bce_criterion(self.out_dv, self.label_dv)

        self.out_dl0 = self.disc_l(self.l1.detach())
        self.out_dl1 = self.disc_l(self.l2)
        self.out_dl = torch.cat([self.out_dl0, self.out_dl1], 0)

        self.labelf.data.resize_(self.batchsize).fill_(self.fake_label)
        self.label.data.resize_(self.batchsize).fill_(self.real_label)
        self.label_dl = torch.cat([self.labelf, self.label], 0)

        self.err_dl_bce = self.bce_criterion(self.out_dl, self.label_dl)

        self.err_d=self.err_dl_bce+self.err_dv_bce
        self.err_d.backward(retain_graph=True)
        self.optimizer_dv.step()
        self.optimizer_dl.step()

    def update_netg(self):
        """
        # ============================================================ #
        # (2) Update G network: log(D(G(x)))  + ||G(x) - x||           #
        # ============================================================ #

        """
        self.enc.zero_grad()
        self.dec.zero_grad()

        self.out_gv1 = self.disc_v(self.del2)


        self.label.data.resize_(self.batchsize).fill_(self.real_label)

        self.err_gv_bce = self.bce_criterion(self.out_gv1, self.label)


        self.out_gl1 = self.disc_l(self.l1)


        self.label.data.resize_(self.batchsize).fill_(self.real_label)


        self.err_gl_bce = self.bce_criterion(self.out_gl1, self.label)

        self.err_g_mse=self.l2l_criterion(self.input, self.del1)

        self.err_g = self.err_gl_bce + self.err_gv_bce + self.err_g_mse
        self.err_g.backward(retain_graph=True)
        self.optimizer_en.step()
        self.optimizer_de.step()


    def update_netc(self):
        self.cl.zero_grad()
        self.out_c0=self.cl(self.del2.detach())
        self.out_c1 = self.cl(self.input)
        # self.out_c1=self.netc(self.del1.detach())
        self.out_c = torch.cat([self.out_c0,self.out_c1],0)

        self.labelf.data.resize_(self.batchsize).fill_(self.fake_label)
        self.label.data.resize_(self.batchsize).fill_(self.real_label)
        self.label_c=torch.cat([self.labelf,self.label],0)

        self.err_c_bce = self.bce_criterion(self.out_c, self.label_c)
        self.err_c_bce.backward()
        self.optimizer_c.step()

    def update_l2(self):
        for i in range(5):
            self.optimizer_l2.zero_grad()
            self.out_m=self.cl(self.del2)
            self.labelf.data.resize_(self.batchsize).fill_(self.fake_label)
            self.err_m_bce = self.bce_criterion(self.out_m, self.labelf)
            self.err_m_bce.backward()
            self.optimizer_l2.step()
            self.del2=self.dec(self.l2)

    def optimize(self):
        """ 
        Optimize netD and netG  networks.
        """
        self.u = np.random.uniform(-1, 1, (self.batchsize, 288, 1, 1))
        self.l2 = torch.from_numpy(self.u).float()
        self.n = torch.randn(self.batchsize, 1, 28, 28)
        self.l1 = self.enc(self.input + self.n)
        print(self.l1.shape,99999999999999999999999999999999999)
        self.del1=self.dec(self.l1)
        self.del2=self.dec(self.l2)
        self.update_netc()
        self.update_netd()

        self.update_l2()
        self.update_netg()

    def optimize_fore(self):
        """ 
        Optimize netD and netG  networks.
        """
        self.u = np.random.uniform(-1, 1, (32, 288, 1, 1))
        
        self.l2 = torch.from_numpy(self.u).float()
        print('self u shape',self.l2.shape)
        self.n = torch.randn(32, 1, 28, 28)
        self.l1 = self.enc(self.input + self.n)
        self.del1=self.dec(self.l1)
        self.del2=self.dec(self.l2)
        self.update_netc()
        self.update_netd()

        self.update_l2()
        self.update_netg()
        print('sssssssssssssssssssssss')



   



    
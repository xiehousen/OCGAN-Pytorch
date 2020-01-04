import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import math
##
def weights_init(mod):

    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        mod.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)
# target output size of 5x7

# ###
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
        output = torch.tanh(self.conv1(input))  
        # print('output1:',output.shape)
        output = torch.tanh(self.conv2(output))
        output = self.batch_norm_1(output)
        output = self.maxpooling_1(output)
        # print('output2:',output.shape)
        output = torch.tanh(self.conv3(output))
        # print('output3:',output.shape)
        output = torch.tanh(self.conv4(output))
        output = self.batch_norm_2(output)
        output = self.maxpooling_2(output)
        # print('output4:',output.shape)

        output = torch.tanh(self.conv5(output))
        # print('output5:',output.shape)
        output = torch.tanh(self.conv6(output)) 

        output = output.view(output.size(0),-1)
        # print('wwwwwwwwwwwwwwwwwwwwwwwww:',output.shape)

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
        # self.activation = nn.Tanh()
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
        output = torch.tanh(self.conv2(output))
        # print('l2_output2:',output.shape)
        output = torch.tanh(self.conv3(output))
        output = self.batch_norm_1(output)
        # print('l2_output3:',output.shape)

        output = self.conv4(output)
        # print('l2_output4:',output.shape)
        output = torch.tanh(self.conv5(output))
        # print('l2_output5:'/,output.shape)
        output = torch.tanh(self.conv6(output))
        output = self.batch_norm_2(output)
        # print('l2_output6:',output.shape)

        output = self.conv7(output)
        # print('l2_output7:',output.shape)
        output = torch.tanh(self.conv8(output)) 
        # print('l2_output8:',output.shape)
        output = torch.tanh(self.conv9(output))
        output = self.batch_norm_3(output)
        # print('l2_outpu/t9:',output.shape)
        # output = output.permute((0, 2, 3, 1))
        # output = output.contiguous().view(-1, 4 * 4 * 64) 
        output = torch.sigmoid(self.conv10(output))   
        # print('l2_output10:',output.shape)
        return output


class get_disc_latent(nn.Module):
    """
    DISCRIMINATOR latent NETWORK
    """

    def __init__(self):
        super(get_disc_latent, self).__init__()
        self.dense_1 = nn.Linear(288, 128)
        self.batch_norm_1 = nn.BatchNorm1d(128)

        self.dense_2 = nn.Linear(128, 64)
        self.batch_norm_2 = nn.BatchNorm1d(64)

        self.dense_3 = nn.Linear(64, 32)
        self.batch_norm_3 = nn.BatchNorm1d(32)

        self.dense_4 = nn.Linear(32, 16)
        self.batch_norm_4 = nn.BatchNorm1d(16)

        self.dense_5 = nn.Linear(16, 1)
    

    def forward(self, input):
        # print(self.dense_1(input).shape,22222222222222222222222222222222222)
        output = input.view(input.size(0),-1)

        output = self.batch_norm_1(self.dense_1(output))
        output = F.relu(output)
        
        output = self.batch_norm_2(self.dense_2(output))
        output = F.relu(output)

        output = self.batch_norm_3(self.dense_3(output))
        output = F.relu(output)

        output = self.batch_norm_4(self.dense_4(output))
        output = F.relu(output)

        output = self.dense_5(output)
        # print(output.shape,11111111111111111111111111111)
        output = torch.sigmoid(output)

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
        self.conv5 = nn.AdaptiveAvgPool2d(output_size=1)
        # self.sigmoid = torch.sigmoid()

    def forward(self, input):

        output = self.batch_norm_1(self.conv1(input))
        output = self.leaky_relu(output)

        output = self.batch_norm_2(self.conv2(output))
        output = self.leaky_relu(output)

        output = self.batch_norm_3(self.conv3(output))
        output = self.leaky_relu(output)

        output = self.conv4(output)
        output = torch.sigmoid(self.conv5(output))
        output = output.view(output.size(0), -1)

        return output


class get_classifier(nn.Module):
    """
    Classfier NETWORK
    """

    def __init__(self):
        super(get_classifier, self).__init__()
        self.conv1 = nn.Conv2d(1,32,5,stride=1,padding=5//2)
        self.batch_norm_1 = nn.BatchNorm2d(32)
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(32,64,5,stride=1,padding=5//2)
        self.batch_norm_2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64,64,5,stride=1,padding=5//2)
        self.batch_norm_3 = nn.BatchNorm2d(64)
                
        self.conv4 = nn.Conv2d(64,1,5,stride=1,padding=5//2)
        self.conv5 = nn.AdaptiveAvgPool2d(output_size=1)
        # self.sigmoid = torch.sigmoid()
        


    def forward(self, input):
        output = self.batch_norm_1(self.conv1(input))
        output = self.leaky_relu(output)

        output = self.batch_norm_2(self.conv2(output))
        output = self.leaky_relu(output)

        output = self.batch_norm_3(self.conv3(output))
        output = self.leaky_relu(output)

        output = self.conv4(output)

        output = torch.sigmoid(self.conv5(output))
        output = output.view(output.size(0), -1)


        return output
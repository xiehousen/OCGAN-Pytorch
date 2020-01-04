# OCGAN-Pytorch
OCGAN - UnOfficial PyTorch Implementation

This is the official PyTorch implementation of OCGAN. The code accompanies the paper "[OCGAN: One-class Novelty Detection Using GANs with Constrained Latent Representations](http://openaccess.thecvf.com/content_CVPR_2019/papers/Perera_OCGAN_One-Class_Novelty_Detection_Using_GANs_With_Constrained_Latent_Representations_CVPR_2019_paper.pdf)".

The author's implementation of *OCGAN* in MXNet is at [here](https://github.com/PramuPerera/OCGAN).

## Features
* Unified interface for different network architectures
* Multi-GPU support
* Training progress bar with rich info
* Training log and training curve visualization code (see `./utils/logger.py`)

## Installation
This code is written in `Python 3.7` and tested with `Pytorch 1.1.0`.
* Install [PyTorch](http://pytorch.org/)
* Clone recursively
  ```
  git clone --recursive https://github.com/xiehousen/OCGAN-Pytorch.git
  ```

## Training
  ```
  python train.py
  ```
  During training, every five epochs will store the training input and output pictures in the result folder.

  <p align="center">
  <img src="https://github.com/xiehousen/OCGAN-Pytorch/blob/master/result/0002/train_dc_fake-2/fake_095.png">      <img src="https://github.com/xiehousen/OCGAN-Pytorch/blob/master/result/0002/train_dc_real-2/real_095.png">
</p>  
  
The picture on the left is the generated image of the autoencoder, and the right is the real training image
  
  
  ## Test Results
  ```
  python test.py
  ```
  The test results are also saved in the result folder.

  <p align="center">
  <img src="/result/0002/test_dc_fake-2/fake_01.png">      <img src="/result/0002/test_dc_real-2/real_01.png">
</p>  
  
The picture on the left is the generated image of the test data set after passing through the automatic encoder, and the right is the real test image.  

During the test, the AUC was also calculated, and the calculation results were stored in the log file of the result folder.


  ##  At Last
  **When training 100 epochs, the auc did not achieve the results in the paper. We continue to improve the code to achieve the results in the paper.**
  
  ## Acknowledgements
This code borrows part from
+ [nuclearboy95/Anomaly-Detection-OCGAN-tensorflow](https://github.com/nuclearboy95/Anomaly-Detection-OCGAN-tensorflow)
+ [TobinZuo/OCGAN](https://github.com/TobinZuo/OCGAN)

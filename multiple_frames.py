import pandas as pd
import matplotlib.pyplot as plt
import dlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import supplemental_code.supplemental_code as sc
from torch.autograd import Variable
import numpy as np
import pca 
import h5py
import latent_parameters_estimation as lpe
import texturize as tx



if __name__=='__main__':

    BFM_PATH = "models_landmarks/model2017-1_face12_nomouth.h5"
    bfm = h5py.File(BFM_PATH , 'r' )
    model=None
    for i in range(4):

        IMAGE_PATH = 'images/image'+str(i)+'.png'

        img = dlib.load_rgb_image(IMAGE_PATH)
        model = lpe.train(bfm, img, model, lr=0.6, iters=2000)

        tx.texturize(bfm, img, model=model, save_ob_path='images/multipleframes_'+str(i)+'_pointcloud.OBJ')


import h5py
import numpy as np
from supplemental_code.supplemental_code import save_obj
from scipy.spatial.transform import Rotation as R
import csv
import dlib
import torch
import matplotlib.pyplot as plt
from latent_parameters_estimation import train
from texturize import texturize
from supplemental_code.supplemental_code import render
from PIL import Image


if __name__ == '__main__':


    BFM_PATH = "models_landmarks/model2017-1_face12_nomouth.h5"

    bfm = h5py.File(BFM_PATH , 'r' )
    triangle_top = np.asarray(bfm['shape/representer/cells'], int).T
    # IMAGE_PATH = 'images/artemis.png'


    IMAGE_PATH = 'images/sander.jpeg'
    bg = dlib.load_rgb_image(IMAGE_PATH)
    # m = train(bfm, img, lr=10, iters=10)
    uvz, color = texturize(bfm, bg, save_ob_path='images/swap.OBJ')
    w = int(bg.shape[0])
    h = int(bg.shape[1])


    IMAGE_PATH = 'images/artemis.png'
    fg = dlib.load_rgb_image(IMAGE_PATH)
    bfm = h5py.File("models_landmarks/model2017-1_face12_nomouth.h5" , 'r' )

    uvz2, color2 = texturize(bfm, fg, save_ob_path='images/swap.OBJ')
    # print(uvz.shape)

    image = render(uvz2, color, triangle_top, H=h, W=w)
    save_obj('images/swap.OBJ', uvz2, color, triangle_top)
    plt.imshow(image)
    plt.show()



    # plt.plot(image)

    #
    # print(color2.shape)
    # h,w=1000
    # print(,w)
    # h, w  = uvz[0].shape

    # h = int(uvz2[:, 0].max()  - uvz2[:, 0].min())
    # w = int(uvz2[:, 1].max() - uvz2[:, 0].min())

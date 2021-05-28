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
    IMAGE_PATH = 'images/sander.jpeg'
    # IMAGE_PATH = 'images/artemis.png'
    img = dlib.load_rgb_image(IMAGE_PATH)
    bfm = h5py.File(BFM_PATH , 'r' )

    # m = train(bfm, img, lr=10, iters=10)
    uvz, color = texturize(bfm, img, save_ob_path='images/swap.OBJ')

    # print('Alpha = ', m.alpha)
    # print('Delta = ', m.delta)
    # print('omega = ', m.w)
    # print('T = ', m.t)

    IMAGE_PATH = 'images/artemis.png'
    bfm = h5py.File("models_landmarks/model2017-1_face12_nomouth.h5" , 'r' )
    triangle_top = np.asarray(bfm['shape/representer/cells'], int).T

    img = dlib.load_rgb_image(IMAGE_PATH)
    print(uvz.shape)
    uvz2, color2= texturize(bfm, img, save_ob_path='images/swap.OBJ')
    print(color2.shape)
    # h,w=1000
    # print(,w)
    # h, w  = uvz[0].shape
    h = int(uvz[:, 0].max())
    w = int(uvz[:, 1].max())
    image = render(uvz2, color, triangle_top, H=h, W=w)
    # print("hallo")
    im = Image.fromarray(image, 'RGB')
    # print(image.shape)
    plt.imshow(im)

    plt.show()
    # plt.plot(image)


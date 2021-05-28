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
import cv2


if __name__ == '__main__':


    BFM_PATH = "models_landmarks/model2017-1_face12_nomouth.h5"

    bfm = h5py.File(BFM_PATH , 'r' )
    triangle_top = np.asarray(bfm['shape/representer/cells'], int).T


    IMAGE_PATH = 'images/sander.jpeg'
    bg = dlib.load_rgb_image(IMAGE_PATH)
    bg = cv2.imread(IMAGE_PATH)
    bg = cv2.resize(bg, (400, 400))

    cv2.imwrite(IMAGE_PATH, bg)

    uvz, color = texturize(bfm, bg, save_ob_path='images/swap.OBJ')
    w = int(bg.shape[0])
    h = int(bg.shape[1])



    IMAGE_PATH = 'images/artemis.png'
    fg = cv2.imread(IMAGE_PATH)
    fg = cv2.resize(fg, (400, 400))

    cv2.imwrite(IMAGE_PATH, fg)
    # fg = dlib.load_rgb_image(IMAGE_PATH)

    uvz2, color2 = texturize(bfm, fg, save_ob_path='images/swap.OBJ')
    # print(uvz.shape)

    image = render(uvz2, color, triangle_top, H=h, W=w)
    save_obj('images/swap.OBJ', uvz, color2, triangle_top)
    plt.imshow(image)
    plt.show()




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


if __name__='__main__':

    bfm = h5py.File("models_landmarks/model2017-1_face12_nomouth.h5" , 'r' )
	t = np.asarray([0,0,-500])
	degree = np.array([0, 10, 0])

	G, triangle_top, vertex_color = morphable_model(bfm)
	rotate(G, triangle_top, vertex_color, np.array([0,10, 0]), t, translation=False)
	rotate(G, triangle_top, vertex_color, np.array([0,-10, 0]), t, translation=False)
	rotate(G, triangle_top, vertex_color, np.array([0,10, 0]), t, translation=True)

	landmark_points(G, triangle_top, vertex_color, degree, t)


    BFM_PATH = "models_landmarks/model2017-1_face12_nomouth.h5"
    IMAGE_PATH = 'images/koning2.png'
    img = dlib.load_rgb_image(IMAGE_PATH)
    bfm = h5py.File(BFM_PATH , 'r' )

    m = train(bfm, img, lr=0.5, iters=1000, visualise=True)

    print('Alpha = ', m.alpha.min(), m.alpha.max())
    print('Delta = ', m.delta.min(), m.delta.max())
    print('omega = ', m.w)
    print('T = ', m.t)


    img = dlib.load_rgb_image(IMAGE_PATH)
    bfm = h5py.File(BFM_PATH , 'r' )

    G, colors = texturize(bfm, img, save_ob_path='images/pointcloud_texturize_mean.OBJ')

    image = sc.render(G, colors, np.asarray(bfm['shape/representer/cells'], int).T)
    image = image.astype(np.int64)

    plt.imshow(image)

    plt.show()


    BFM_PATH = "models_landmarks/model2017-1_face12_nomouth.h5"
    bfm = h5py.File(BFM_PATH , 'r' )
    model=None
    for i in range(4):

        IMAGE_PATH = 'images/image'+str(i)+'.png'

        img = dlib.load_rgb_image(IMAGE_PATH)
        model = lpe.train(bfm, img, model, lr=0.6, iters=2000)

        tx.texturize(bfm, img, model=model, save_ob_path='images/multipleframes_'+str(i)+'_pointcloud.OBJ')


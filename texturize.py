
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



def texturize(bfm, img, model=None, save_ob_path=None):

    if model==None:
        model = lpe.train(bfm, img, model=None, lr=10, iters=1000)
    
    w, t = model.w, model.t
    G = pca.morphable_model(model.bfm, model.alpha, model.delta, model.device)

    G = pca.landmark_points_rotation(G, w, t).detach().numpy()

    colors = np.zeros((G.shape[0], 3))

    G[:, 0] = np.clip(G[:, 0], 0, img.shape[1] - 1)
    G[:, 1] = np.clip(G[:, 1], 0, img.shape[0] - 1)

    for i, point in enumerate(G):


        x = point[1]
        y = point[0]

        x1 = np.floor(x).astype(int)
        x2 = np.ceil(x).astype(int)

        y1 = np.floor(y).astype(int)
        y2 = np.ceil(y).astype(int)

        if (x1 - x2 == 0) or (y1 - y2 == 0):
            continue

        first = 1/((x2-x1)*(y2-y1))

        second = np.asarray([x2-x, x-x1])
        third = np.asarray([[img[x1,y1],img[x1,y2]],
                            [img[x2,y1], img[x2,y2]]])
        fourth = np.asarray([y2-y, y-y1])

        f_xy =  fourth.dot((first*second).dot(third))

        if any(np.isnan(f_xy)):
            continue

        colors[i] = f_xy

    if save_ob_path is not None:
        save(bfm, colors, model, save_ob_path)

def save(bfm, colors, model, save_ob_path):

    id = 30
    exp = 20

    vertex_color = colors
    triangle_top = np.asarray(bfm['shape/representer/cells'], int).T
    mu_id = np.asarray(bfm['shape/model/mean'], float).reshape(-1,3)
    mu_exp = np.asarray(bfm['expression/model/mean'], float).reshape(-1,3)

    E_id = np.asarray(bfm['shape/model/pcaBasis'], float)[:,:id].reshape(-1,3, id)
    E_exp = np.asarray(bfm['shape/model/pcaBasis'], float)[:,:exp].reshape(-1,3, exp)

    sigma_id =  np.sqrt(np.asarray(bfm['shape/model/pcaVariance'], float)[:id])
    sigma_exp = np.sqrt(np.asarray(bfm['expression/model/pcaVariance'], float)[:exp])

    alpha = model.alpha.detach().numpy()
    delta = model.delta.detach().numpy()

    G = mu_id + E_id @ (alpha * sigma_id) + mu_exp + E_exp @ (delta * sigma_exp)

    sc.save_obj(save_ob_path, G, vertex_color, triangle_top)


if __name__=='__main__':      

    BFM_PATH = "models_landmarks/model2017-1_face12_nomouth.h5"
    IMAGE_PATH = 'images/koning.png'

    img = dlib.load_rgb_image(IMAGE_PATH)
    bfm = h5py.File(BFM_PATH , 'r' )

    texturize(bfm, img, save_ob_path='images/pointcloud_texturize.OBJ')

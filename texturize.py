
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



def texturize(bfm, img, save_ob=True):
    m = lpe.train(bfm, img, iters=200)
    
    w, t = m.w, m.t
    G = pca.morphable_model(m.bfm, m.alpha, m.delta, m.device)

    G = pca.landmark_points_rotation(G, w, t).detach().numpy()

    colors = np.zeros((G.shape[0], 3))

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

    if save_ob:
        save(bfm, colors, m)

def save(bfm, colors, m):

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

    alpha = m.alpha.detach().numpy()
    delta = m.delta.detach().numpy()

    G = mu_id + E_id @ (alpha * sigma_id) + mu_exp + E_exp @ (delta * sigma_exp)

    sc.save_obj('images/pointcloudyes.OBJ', G, vertex_color, triangle_top)


if __name__=='__main__':      

    BFM_PATH = "models_landmarks/model2017-1_face12_nomouth.h5"
    IMAGE_PATH = 'images/koning.png'

    img = dlib.load_rgb_image(IMAGE_PATH)
    bfm = h5py.File(BFM_PATH , 'r' )

    texturize(bfm, img)

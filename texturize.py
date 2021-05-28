
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
from supplemental_code.supplemental_code import render
import cv2
def bilinear_sampler(img, x, y):
    # prepare useful params
    B = tf.shape(img)[0]
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    C = tf.shape(img)[3]

    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')

    # grab 4 nearest corner points for each (x_i, y_i)
    # i.e. we need a rectangle around the point of interest
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1
    zero = tf.zeros_like(x0)

    # clip to range [0, H/W] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = _get_pixel_value(img, x0, y0)
    Ib = _get_pixel_value(img, x0, y1)
    Ic = _get_pixel_value(img, x1, y0)
    Id = _get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=2)
    wb = tf.expand_dims(wb, axis=2)
    wc = tf.expand_dims(wc, axis=2)
    wd = tf.expand_dims(wd, axis=2)

    # compute output
    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])

    return out

def texturize(bfm, img, model=None, save_ob_path=None):

    if model==None:
        model = lpe.train(bfm, img, model=None, lr=0.3, iters=600)

    w, t = model.w, model.t
    G = pca.morphable_model(model.bfm, model.alpha, model.delta, model.device)

    G = pca.landmark_points_rotation(G, w, t).detach().numpy()

    colors = np.zeros((G.shape[0], 3))

    width, height = img.shape[1], img.shape[0]

    G[:, 0] = np.clip(G[:, 0], 0, height-1)

    G[:, 1] = np.clip(G[:, 1], 0, width-1)

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
        G_new = save(bfm, colors, model, save_ob_path)
    return G_new, colors


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
    return G


# if __name__=='__main__':

#     BFM_PATH = "models_landmarks/model2017-1_face12_nomouth.h5"
#     IMAGE_PATH = 'images/koning2.png'

#     img = dlib.load_rgb_image(IMAGE_PATH)
#     bfm = h5py.File(BFM_PATH , 'r' )
#     triangle_top = np.asarray(bfm['shape/representer/cells'], int).T
#     w = int(img.shape[0])
#     h = int(img.shape[1])

#     uvz, color = texturize(bfm, img, save_ob_path='images/pointcloud_texturize_mean.OBJ')
#     image = render(uvz, color, triangle_top, H=h, W=w)

#     plt.imshow(image)
#     plt.show()

if __name__=='__main__':

    BFM_PATH = "models_landmarks/model2017-1_face12_nomouth.h5"
    IMAGE_PATH = 'images/koning2.png'

    img = dlib.load_rgb_image(IMAGE_PATH)
    # img = cv2.imread(IMAGE_PATH)
    bfm = h5py.File(BFM_PATH , 'r' )

    G, colors = texturize(bfm, img, save_ob_path='images/pointcloud_texturize_mean.OBJ')
    print(G.shape)
    print (colors.shape)
    image = sc.render(G, colors, np.asarray(bfm['shape/representer/cells'], int).T, H=img.shape[0], W=img.shape[1])

    image = image.astype(np.int64)

    plt.imshow(image)

    plt.show()
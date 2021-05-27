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



class energy_model(torch.nn.Module):

    def __init__(self, device, bfm, landmarks, alpha=30, delta=20, w=[0, 10, 0], t=[0, 0, -500] ):
        super().__init__()

        self.landmarks = landmarks
        self.alpha = nn.Parameter(torch.zeros(alpha), requires_grad=True).to(device)
        self.delta = nn.Parameter(torch.zeros(delta), requires_grad=True).to(device)
        self.w = nn.Parameter(torch.FloatTensor(w), requires_grad=True).to(device)
        self.t = nn.Parameter(torch.FloatTensor(t), requires_grad=True).to(device)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.bfm = bfm

    def forward(self):
        
        G = pca.morphable_model(self.bfm, self.alpha, self.delta, self.device)
        project_G = pca.landmark_points_rotation(G, self.w, self.t, self.device)

        return project_G[self.landmarks, :2]



def L_lan(p_landmarks, gt_landmarks):
    return torch.mean((p_landmarks-gt_landmarks)**2)

def L_reg(lambda_a, lambda_d, alpha, delta):
    return lambda_a * torch.mean(alpha**2) + lambda_d * torch.mean(delta**2)


def loss(p_landmarks, gt_landmarks, lambda_a, lambda_d, alpha, delta):
    return L_lan(p_landmarks, gt_landmarks) + L_reg(lambda_a, lambda_d, alpha, delta)




def train(bfm, img, model=None, lr=10, iters=1000):

    LEARNING_RATE = lr 
    NUM_ITERS = iters 
    OPTIMIZER_CONSTRUCTOR = optim.Adam 

    p_landmarks = torch.LongTensor(np.loadtxt("supplemental_code/Landmarks68_model2017-1_face12_nomouth.anl", dtype=np.int32))
    gt_landmarks = torch.LongTensor(sc.detect_landmark(img))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log_name = f"{LEARNING_RATE}, {OPTIMIZER_CONSTRUCTOR.__name__}"
    writer = SummaryWriter(log_dir=f"runs/{log_name}")
    print("To see tensorboard, run: tensorboard --logdir=runs/")

    if model==None:
        model = energy_model(device, bfm, p_landmarks)
    else:
        model.w = nn.Parameter(torch.FloatTensor([0, 10, 0]), requires_grad=True).to(device)
        model.t = nn.Parameter(torch.FloatTensor([0, 0, -500]), requires_grad=True).to(device)


    ### Training the model ###
    loss_min = 10e10
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    lambda_a, lambda_d = 1., 1.
    with torch.autograd.set_detect_anomaly(True):

        for t in range(NUM_ITERS):

            optimizer.zero_grad()

            landmarks_new = model.forward()

            current_loss = loss(landmarks_new, gt_landmarks, lambda_a, lambda_d, model.alpha, model.delta)
            print(current_loss)

            current_loss.backward()

            if current_loss<loss_min:
                return_model = model

            optimizer.step()

            writer.add_scalar('loss', current_loss, global_step=t)

    writer.close()

    return return_model

if __name__ == '__main__':


    BFM_PATH = "models_landmarks/model2017-1_face12_nomouth.h5"
    IMAGE_PATH = 'images/koning.png'
    img = dlib.load_rgb_image(IMAGE_PATH)
    bfm = h5py.File(BFM_PATH , 'r' )

    m = train(bfm, img, lr=10, iters=1000)

    print('Alpha = ', m.alpha)
    print('Delta = ', m.delta)
    print('omega = ', m.w)
    print('T = ', m.t)

    
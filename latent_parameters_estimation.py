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

    def __init__(self, device, bfm, landmarks, alpha=30, delta=20, w=[0, 10, 0], t=[ 0, 0, -500]):
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
    return torch.sum(torch.nn.PairwiseDistance(p=2)(p_landmarks, gt_landmarks)**2)

def L_reg(lambda_a, lambda_d, alpha, delta):
    return lambda_a * torch.sum(alpha**2) + lambda_d * torch.sum(delta**2)


def loss(p_landmarks, gt_landmarks, lambda_a, lambda_d, alpha, delta):
    return L_lan(p_landmarks, gt_landmarks) + L_reg(lambda_a, lambda_d, alpha, delta)


def visualize_landmarks(img, landmarks, radius=2):
  new_img = np.copy(img)
  h, w, _ = new_img.shape
  for x, y in landmarks:
    x = int(x)
    y = int(y)
    new_img[max(0,y-radius):min(h-1,y+radius),max(0,x-radius):min(w-1,x+radius)] = (255, 0, 0)
  plt.imshow(new_img)
  plt.show()

def train(bfm, img, model=None, lr=1, iters=1000, visualise=False):

    LEARNING_RATE = lr 
    NUM_ITERS = iters 
    OPTIMIZER_CONSTRUCTOR = optim.Adam 

    p_landmarks = torch.LongTensor(np.loadtxt("supplemental_code/Landmarks68_model2017-1_face12_nomouth.anl", dtype=np.int32))
    gt_landmarks = torch.LongTensor(sc.detect_landmark(img))

    if visualise:
        visualize_landmarks(img, gt_landmarks.numpy(), radius=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log_name = f"{LEARNING_RATE}, {OPTIMIZER_CONSTRUCTOR.__name__}"
    writer = SummaryWriter(log_dir=f"runs/{log_name}")
    print("To see tensorboard, run: tensorboard --logdir=runs/")

    
    if model==None:
        model = energy_model(device, bfm, p_landmarks)
    else:
        model.w = nn.Parameter(torch.FloatTensor([0, 10, 0]), requires_grad=True).to(device)
        model.t = nn.Parameter(torch.FloatTensor([ 137.5618, -186.9494, -300]), requires_grad=True).to(device)


    ### Training the model ###
    loss_min = 10e10
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    lambda_a, lambda_d = 1., 1.
    optim_iter = 0
    with torch.autograd.set_detect_anomaly(True):

        for t in range(NUM_ITERS):

            optimizer.zero_grad()

            landmarks_new = model.forward()

            if t == 0:
                if visualise:
                    visualize_landmarks(img, landmarks_new.detach().numpy(), radius=2)
            current_loss = loss(landmarks_new, gt_landmarks, lambda_a, lambda_d, model.alpha, model.delta)
            print(current_loss)
        
            current_loss.backward()

            if current_loss<loss_min:
                optim_iter = t
                loss_min = current_loss
                return_model = model

            optimizer.step()

            writer.add_scalar('loss', current_loss, global_step=t)

    writer.close()

    print(loss_min)
    print(optim_iter)
    landmarks_final = return_model.forward()
    if visualise:
        visualize_landmarks(img, landmarks_final.detach().numpy(), radius=2)
    return return_model

if __name__ == '__main__':


    BFM_PATH = "models_landmarks/model2017-1_face12_nomouth.h5"
    IMAGE_PATH = 'images/koning2.png'
    img = dlib.load_rgb_image(IMAGE_PATH)
    bfm = h5py.File(BFM_PATH , 'r' )

    m = train(bfm, img, lr=0.5, iters=1000, visualise=True)

    print('Alpha = ', m.alpha.min(), m.alpha.max())
    print('Delta = ', m.delta.min(), m.delta.max())
    print('omega = ', m.w)
    print('T = ', m.t)

    
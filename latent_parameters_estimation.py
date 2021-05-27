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


BFM_PATH = "models_landmarks/model2017-1_face12_nomouth.h5"
IMAGE_PATH = 'images/koning.png'

# Then we define the prediction model
torch.autograd.set_detect_anomaly(True)
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



### TensorBoard Writer Setup ###
def train(bfm, img, lr=10, iters=1000):



    ### Hyperparameters ###

    LEARNING_RATE = lr 
    NUM_ITERS = iters 
    OPTIMIZER_CONSTRUCTOR = optim.Adam 



    p_landmarks = torch.LongTensor(np.loadtxt("supplemental_code/Landmarks68_model2017-1_face12_nomouth.anl", dtype=np.int32))
    gt_landmarks = torch.LongTensor(sc.detect_landmark(img))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log_name = f"{LEARNING_RATE}, {OPTIMIZER_CONSTRUCTOR.__name__}"
    writer = SummaryWriter(log_dir=f"runs/{log_name}")
    print("To see tensorboard, run: tensorboard --logdir=runs/")

    model = energy_model(device, bfm, p_landmarks)


    ### Training the model ###

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    lambda_a, lambda_d = 1., 1.
    with torch.autograd.set_detect_anomaly(True):

        for t in range(NUM_ITERS):

            optimizer.zero_grad()

            landmarks_new = model.forward()

            current_loss = loss(landmarks_new, gt_landmarks, lambda_a, lambda_d, model.alpha, model.delta)
            print(current_loss)

            current_loss.backward()

            optimizer.step()

            writer.add_scalar('loss', current_loss, global_step=t)

    writer.close()

    return model





def texturize(bfm, img):
    m = train(bfm, img, iters=400)
    
    w, t = m.w, m.t
    G = pca.morphable_model(m.bfm, m.alpha, m.delta, m.device)

    G = pca.landmark_points_rotation(G, w, t).detach().numpy()

    colors = np.zeros((G.shape[0], 3))
    
    for i, point in enumerate(G):
        x1 = point[1]
        x2 = point[0]

        x2_ceil = np.ceil(x2).astype(int)
        x2_floor = np.floor(x2).astype(int)
        x1_ceil = np.ceil(x1).astype(int)
        x1_floor = np.floor(x1).astype(int)

        if (x1_ceil - x1_floor == 0) or (x2_ceil - x2_floor == 0):
            continue

        # Interp over columns
        lower = (x2_ceil - x2) / (x2_ceil - x2_floor) * img[x1_floor, x2_floor, :] \
                + (x2 - x2_floor) / (x2_ceil - x2_floor) * img[x1_floor, x2_ceil, :]

        # Interp over next row and then columns
        upper = (x2_ceil - x2) / (x2_ceil - x2_floor) * img[x1_ceil, x2_floor, :] \
                + (x2 - x2_floor) / (x2_ceil - x2_floor) * img[x1_ceil, x2_ceil, :]

        intensity = (x1_ceil - x1) / (x1_ceil - x1_floor) * lower \
                    + (x1 - x1_floor) / (x1_ceil - x1_floor) * upper

        if any(np.isnan(intensity)):
            continue

        colors[i] = intensity
	
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
    # print(G.shape, vertex_color.shape, triangle_top.shape)
    # sc.save_obj('images/pointcloudyes.OBJ', G, vertex_color, triangle_top)

        


img = dlib.load_rgb_image(IMAGE_PATH)
bfm = h5py.File(BFM_PATH , 'r' )

texturize(bfm, img)

# def morphable_model(bfm):

# 	id = 30
# 	exp = 20

# 	triangle_top = np.asarray(bfm['shape/representer/cells'], int).T
# 	vertex_color = np.asarray(bfm['color/model/mean'], float).reshape(-1,3)

# 	mu_id = np.asarray(bfm['shape/model/mean'], float).reshape(-1,3)
# 	mu_exp = np.asarray(bfm['expression/model/mean'], float).reshape(-1,3)

# 	E_id = np.asarray(bfm['shape/model/pcaBasis'], float)[:,:id].reshape(-1,3, id)
# 	E_exp = np.asarray(bfm['shape/model/pcaBasis'], float)[:,:exp].reshape(-1,3, exp)

# 	sigma_id =  np.sqrt(np.asarray(bfm['shape/model/pcaVariance'], float)[:id])
# 	sigma_exp = np.sqrt(np.asarray(bfm['expression/model/pcaVariance'], float)[:exp])

# 	alpha = model.alpha.detach().numpy()
# 	delta = model.delta.detach().numpy()


# 	G = mu_id + E_id @ (alpha * sigma_id) + mu_exp + E_exp @ (delta * sigma_exp)

# 	sc.save_obj('images/pointcloudyes.OBJ', G, vertex_color, triangle_top)

# 	return G, triangle_top, vertex_color


# morphable_model(bfm)
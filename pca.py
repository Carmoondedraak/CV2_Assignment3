import h5py
import numpy as np
from supplemental_code.supplemental_code import save_obj
from scipy.spatial.transform import Rotation as R
import csv
import matplotlib.pyplot as plt
import torch
import math

# bfm = h5py.File("model2017-1_face12_nomouth.h5" , 'r' )

def morphable_model(bfm, alpha, delta, device):

	id = 30
	exp = 20

	mu_id = torch.FloatTensor(np.asarray(bfm['shape/model/mean'], float).reshape(-1,3))
	mu_exp = torch.FloatTensor(np.asarray(bfm['expression/model/mean'], float).reshape(-1,3))

	E_id = torch.FloatTensor(np.asarray(bfm['shape/model/pcaBasis'], float)[:,:id].reshape(-1,3, id))
	E_exp = torch.FloatTensor(np.asarray(bfm['shape/model/pcaBasis'], float)[:,:exp].reshape(-1,3, exp))

	sigma_id =  torch.FloatTensor(np.sqrt(np.asarray(bfm['shape/model/pcaVariance'], float)[:id]))
	sigma_exp = torch.FloatTensor(np.sqrt(np.asarray(bfm['expression/model/pcaVariance'], float)[:exp]))


	G = mu_id + E_id @ (alpha * sigma_id) + mu_exp + E_exp @ (delta * sigma_exp)

	return G.to(device)


def rotation(degree, device):
	omega = degree * (math.pi / 180)

	R_x = torch.FloatTensor(
			[[1, 0, 0],
			[0, torch.cos(omega[0]), -1 * torch.sin(omega[0])],
			[0, torch.sin(omega[0]), torch.cos(omega[0])]
			]).to(device)


	R_y = torch.FloatTensor(
			[[torch.cos(omega[1]), 0, torch.sin(omega[1])],
			[0, 1, 0],
			[-1 * torch.sin(omega[1]), 0, torch.cos(omega[1])]
			]).to(device)


	R_z = torch.FloatTensor(
			[[torch.cos(omega[2]), -1 * torch.sin(omega[2]), 0],
			[torch.sin(omega[2]), torch.cos(omega[2]), 0],
			[0, 0, 1]
			]).to(device)

	r = R_z @ R_y @ R_x

	return r

def translation(R, t, device):

	T = torch.cat((R, t.reshape(3, 1)), dim=1)

	# initialize last row of matrix
	l_r = torch.FloatTensor([0, 0, 0, 1]).to(device).view(1, 4)

	# concatenate last row to the rest
	T = torch.cat((T, l_r), dim=0)

	return T


def viewport(device, imageWidth, imageHeight):
	middle_x = imageWidth
	middle_y = imageHeight

	V_p = torch.FloatTensor([
		[middle_x,		0, 				0, 		middle_x],
		[0,				-middle_y,		0,		middle_y],
		[0,					0,				0.5,			0.5],
		[0,					0,				0,				1  ]
	]).to(device)
	return V_p

def projection(landmarks, device, imageWidth, imageHeight):

	angleOfView = 0.5
	n = 300
	f = 2000
	scale = np.tan(angleOfView * 0.5 ) * n
	imageAspectRatio = imageWidth / imageHeight
	r = imageAspectRatio * scale
	l = -r
	t = scale
	b = -t


	P = torch.FloatTensor([	[(2*n)/(r-l),	0,				(r+l)/(r-l),	0],
							[0,				(2*n)/(t-b),	(t+b)/(t-b), 	0],
							[0,				0,				-(f+n)/(f-n),	-(2*f*n)/(f-n)],
							[0, 			0,				-1,				0]]).to(device)
	return P

def landmark_points_rotation(G, w, t, device='cpu'):
	
	R = rotation(w, device)
	T = translation(R, t, device)

	imageWidth = G[:, 0].max() - G[:, 0].min()
	imageHeight = G[:, 1].max() - G[:, 1].min()
	V_p = viewport(device, imageWidth, imageHeight)
	P = projection(G, device, imageWidth, imageHeight)
	Pi = V_p @ P
	points = G.shape[0]

	G_extended = torch.cat((G, torch.ones(points).reshape(points, 1)), axis=1)

	Rotated = T @ G_extended.T

	G = (Pi @ Rotated).T


	G_new = torch.empty((G.shape[0], 0))

	for i in range(G.shape[1]-1):
		G_new = torch.cat((G_new, (G[:, i]/G[:, -1]).view(G_new.shape[0], 1)), axis=1)

	return G_new



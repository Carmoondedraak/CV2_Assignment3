import h5py
import numpy as np
from supplemental_code.supplemental_code import save_obj, render, detect_landmark, visualize_landmarks

from scipy.spatial.transform import Rotation as R
import csv
import matplotlib.pyplot as plt

from copy import deepcopy
from PIL import Image

def morphable_model(bfm):

	id = 30
	exp = 20

	triangle_top = np.asarray(bfm['shape/representer/cells'], int).T
	vertex_color = np.asarray(bfm['color/model/mean'], float).reshape(-1,3)

	mu_id = np.asarray(bfm['shape/model/mean'], float).reshape(-1,3)
	mu_exp = np.asarray(bfm['expression/model/mean'], float).reshape(-1,3)

	E_id = np.asarray(bfm['shape/model/pcaBasis'], float)[:,:id].reshape(-1,3, id)
	E_exp = np.asarray(bfm['shape/model/pcaBasis'], float)[:,:exp].reshape(-1,3, exp)

	sigma_id =  np.sqrt(np.asarray(bfm['shape/model/pcaVariance'], float)[:id])
	sigma_exp = np.sqrt(np.asarray(bfm['expression/model/pcaVariance'], float)[:exp])

	alpha = np.random.uniform(-1, 1, size=(id))
	delta = np.random.uniform(-1, 1, size=(exp))


	G = mu_id + E_id @ (alpha * sigma_id) + mu_exp + E_exp @ (delta * sigma_exp)

	save_obj('images/pointcloudyes.OBJ', G, vertex_color, triangle_top)

	return G, triangle_top, vertex_color


# Inspiration from https://www.meccanismocomplesso.org/en/3d-rotations-and-euler-angles-in-python/
def rotate(G, triangle_top, vertex_color, degree, t, translation=False):

	o_x, o_y, o_z = np.radians(degree)

	R_x = 	np.array(
			[[1, 0, 0],
			[0, np.cos(o_x), -1 * np.sin(o_x)],
			[0, np.sin(o_x), np.cos(o_x)]])

	R_y = np.array(
			[[np.cos(o_y), 0, np.sin(o_y)],
			[0, 1, 0],
			[-1 * np.sin(o_y), 0, np.cos(o_y)]])

	R_z = np.array(
			[[np.cos(o_z), -1 * np.sin(o_z), 0],
			[np.sin(o_z), np.cos(o_z), 0],
			[0, 0, 1]])

	r =  R_x @ R_y @ R_z

	rotated_matrix = G @ r.T
	if translation:
		rotated_matrix = translate(rotated_matrix, t)

	save_obj('images/rotated_matrix_{}{}.OBJ'.format(degree,translation),rotated_matrix, vertex_color, triangle_top)

	return rotated_matrix

def translate(m, t):
	translated = m + t
	return translated


landmarks = np.loadtxt("models_landmarks/Landmarks68_model2017-1_face12_nomouth.anl", dtype=np.int32)

def viewport(landmarks):
	h = landmarks[:, 0].max() - landmarks[:, 0].min()
	w = landmarks[:, 1].max() - landmarks[:, 1].min()

	V_p = np.array([
	[w,		0,		0,		w],
	[0,		h,		0,		h],
	[0,		0,		0.5,	0.5],
	[0,		0,		0,		1]
	])

	return V_p

# Inspiration from https://www.scratchapixel.com/code.php?id=4&origin=/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix
def projection(landmarks):
	imageWidth, imageHeight = landmarks.shape
	angleOfView = 90
	n = 0.1
	f = 100
	scale = np.tan(angleOfView * 0.5 ) * n
	imageAspectRatio = imageWidth / imageHeight
	r = imageAspectRatio * scale
	l = -r
	t = scale
	b = -t


	P = np.array([	[(2*n)/(r-l),	0,				(r+l)/(r-l),	0],
					[0,				(2*n)/(t-b),	(t+b)/(t-b), 	0],
					[0,				0,				-(f+n)/(f-n),	-(2*f*n)/(f-n)],
					[0, 			0,				-1,				0]])
	return P

def landmark_points(G, triangle_top, vertex_color, degree, t):
	rotation = rotate(G, triangle_top, vertex_color, degree, t, translation=True)
	# rotation = rotate(G, triangle_top, vertex_color, 'y', 10, t, translation = True)
	landmarks = np.loadtxt("models_landmarks/Landmarks68_model2017-1_face12_nomouth.anl", dtype=np.int32)
	coords= rotation[landmarks]
	coords = np.hstack((coords, np.ones((coords.shape[0], 1))))
	V_p = viewport(coords)
	P = projection(coords)
	Pi =  V_p @ P


	coords_3D = (coords @ Pi.T)
	x = coords_3D[:, 0]
	y = coords_3D[:, 1]
	hom = coords_3D[:, 3]
	plt.scatter(x/hom, y/hom)
	plt.savefig("images/landmark_points.png")
	plt.show()
	return x, y, hom


if __name__ == '__main__':
	bfm = h5py.File("models_landmarks/model2017-1_face12_nomouth.h5" , 'r' )
	t = np.asarray([0,0,-500])
	degree = np.array([0, 10, 0])

	G, triangle_top, vertex_color = morphable_model(bfm)
	rotate(G, triangle_top, vertex_color, np.array([0,10, 0]), t, translation=False)
	rotate(G, triangle_top, vertex_color, np.array([0,-10, 0]), t, translation=False)
	rotate(G, triangle_top, vertex_color, np.array([0,10, 0]), t, translation=True)


	x,y,hom = landmark_points(G, triangle_top, vertex_color, degree, t)

	# Compare to landmarks from supplemental code:
	# img = Image.open('images/10.png')
	# grey = np.array(img.convert("L"))
	# plt.imshow(img)
	# og = detect_landmark(grey)
	# plt.scatter(og[:, 0], og[:, 1])

	# plt.savefig("images/landmarks.png")
	# plt.show()






import h5py
import numpy as np
from supplemental_code.supplemental_code import save_obj
from scipy.spatial.transform import Rotation as R
import csv
import matplotlib.pyplot as plt

# bfm = h5py.File("models_landmarks/model2017-1_face12_nomouth.h5" , 'r' )

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
	# alpha = (2 * (np.random.random(id) - 1))
	# delta = (2 * (np.random.random(exp) - 1))


	G = mu_id + E_id @ (alpha * sigma_id) + mu_exp + E_exp @ (delta * sigma_exp)

	save_obj('images/pointcloudyes.OBJ', G, vertex_color, triangle_top)

	return G, triangle_top, vertex_color



def rotate(G, triangle_top, vertex_color,axis,degree,t, translation=False):
	r = R.from_euler(axis, degree, degrees=True)
	rotated_matrix = G @ r.as_matrix()
	if translation:
		rotated_matrix = translate(rotated_matrix, t)
	save_obj('images/rotated_matrix_{}{}{}.OBJ'.format(axis,degree,translation),rotated_matrix, vertex_color, triangle_top)
	# save_obj('images/rotated_matrix_{}_{}_{}.OBJ'.format(axis,degree,translation),rotated_matrix,  triangle_top[:28588], vertex_color)
	return rotated_matrix

def translate(m, t):
	translated = m + t
	return translated


landmarks = np.loadtxt("models_landmarks/Landmarks68_model2017-1_face12_nomouth.anl", dtype=np.int32)

def viewport(landmarks):
	v_r = 1
	v_t = 1
	v_l = -1
	v_b = -1
	# width / 2, height / 2
	V_p = np.array([
		[(v_r-v_l)/2,		0, 				0, 		(v_r + v_l)/2],
		[0,				(v_t - v_b)/2,		0,		(v_t + v_b)/2],
		[0,					0,				0.5,			0.5],
		[0,					0,				0,				1  ]
	])
	return V_p

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

def landmark_points(G, triangle_top, vertex_color, t):
	rotation = rotate(G, triangle_top, vertex_color, 'y', 10, t, translation = True)
	landmarks = np.loadtxt("models_landmarks/Landmarks68_model2017-1_face12_nomouth.anl", dtype=np.int32)
	coords= rotation[landmarks]

	V_p = viewport(landmarks)
	P = projection(coords)
	Pi = V_p @ P

	coords = np.hstack((coords, np.ones((coords.shape[0], 1))))
	coords_3D = coords @ Pi.T
	x = coords_3D[:, 0]
	y = coords_3D[:, 1]
	hom = coords_3D[:, 3]
	plt.scatter(x/hom, y/hom)
	plt.show()

if __name__=='__main__':
	
	bfm = h5py.File("models_landmarks/model2017-1_face12_nomouth.h5" , 'r' )
	# morphable_model(bfm)
	t = np.array([0,0,-500])
	G, triangle_top, vertex_color = morphable_model(bfm)
	rotate(G, triangle_top, vertex_color, 'y', 10, t)
	rotate(G, triangle_top, vertex_color, 'y', -10, t)
	rotate(G, triangle_top, vertex_color, 'y', 10, t, translation = True)

	landmark_points(G, triangle_top, vertex_color, t)



import numpy as np
from tensorOperations import naive_relu

"Creamos nuestros tensores."
#Escalar
tensor0d = np.array(0)

#Vector
tensor1d = np.array([0,1,2,3,4,5,6])

#Array
tensor2d = np.array([[0,1,2,3],
					 [4,5,6,7],
					 [8,9,10,11]])

#Tensor 3D, una image 3x4 RGB. 1x(3x4x3)
tensor3d = np.array([[[0, 1, 2,  3],
					  [4, 5, 6,  7],
					  [8, 9, 10, 11]],
					  [[0, 1, 2, 3],
					  [4, 5, 6,  7],
					  [8, 9, 10, 11]],
					  [[0, 1, 2, 3],
					  [4, 5, 6,  7],
					  [8, 9, 10, 11]]])

"Operaciones con tensores"
print(tensor2d)
print(naive_relu(tensor2d))
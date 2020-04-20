# simpleLinearClassifier.py
import numpy as np
import cv2

# Nuestras 3 clases.
labels = ['dog', 'cat', 'panda']

# Random parameters
np.random.seed(1) #Semilla para que lo coeficientes y bias se ajuste a los perros. 
Weights = np.random.randn(3, 3072) # Vector de 3x3072 coefifientes aleatorio
bias = np.random.randn(3) # Vector 3x1 bias.

# Cargamos nuestra imagen y la mostramos.
originalImage = cv2.imread('beagle.jpg')

#Convertimos la imagen en 32x32x3 y luego en un vector 3072x1
imageVector = cv2.resize(originalImage, (32, 32)).flatten() 

# Calculamos la salida de la neurona
scores = Weights.dot(imageVector) + bias

# Creamos una tuple con las etiquetas y los resultados obtenidos.
for (label, score) in zip(labels, scores):
	print("[INFO] {}: {:.3f}".format(label, score)) # Formateamos 

# Nuestra prediccion sera el valor más alto obtenido.
print(np.argmax(scores))

# Dibujamos la prediccion en nuestra imagen
cv2.putText(originalImage, "Label: {}".format(labels[np.argmax(scores)]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Mostramos nuestra imagen con la prediccion
cv2.imshow("Image", originalImage)
cv2.waitKey(0)
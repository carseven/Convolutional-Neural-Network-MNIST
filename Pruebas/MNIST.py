import tensorflow as tf
from keras import models, layers, Sequential, __version__
from keras.utils import to_categorical
import matplotlib.pyplot as plt
print(__version__)

"Cargamos el dataset de MNIST."
(train_images, train_labels),(test_images, test_labels) = tf.keras.datasets.mnist.load_data()
print("Cargamos el dataset de MNIST")


print('')
print('/**********************************************/')
print("		 OBSERVAMOS EL DATASET MNIST")
print('/**********************************************/')
print('Shape del tensor:')
print(train_images.shape)
print('Numero de imagenes:')
print(len(train_labels))
print('Tipo de dato del 3D tensor de las imagenes:')
print(train_images.dtype)
print('1D tensor son las etiquetas:')
print(train_labels)
print('Tipo de dato del 1D tensor de las etiquetas')
print(train_labels.dtype)
print('/**********************************************/')
print('')


"Representamos graficamente una muestra del Dataset"
digit = train_images[4]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

"Preparamos las imagenes para introducirlas"
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255


"Codificamos  categoricamente las etiquetas"
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


"Nuestra red neuronal"
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])


"Entrenamos el modelo"
network.fit(train_images, train_labels, epochs=5, batch_size=128)


"Evaluamos nuestra red entrenada con nuestro imagenes de test"
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('Precision del test: ', test_acc)
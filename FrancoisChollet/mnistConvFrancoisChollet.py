from keras import layers
from keras import models
from keras.utils import to_categorical
from keras.datasets import mnist
import matplotlib.pyplot as plt

"Cargamos el dataset de MNIST."
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


"Preparamos las imagenes para introducirlas"
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255  # Normalizamos

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255  # Normalizamos

"Codificamos  categoricamente las etiquetas"
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


"Convoluciones 2D"
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3),
                        activation='relu',
                        input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))


"Classificador"
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

"Configuramos la funcion de perdidas y el tipo de optimizacion"
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

"Resumen de nuestro modelo en forma de tabla"
model.summary()


"Entrenamos el modelo"
history = model.fit(train_images,
                    train_labels,
                    epochs=5,
                    batch_size=64,
                    validation_data=(test_images, test_labels))


"Evaluamos nuestra red entrenada con nuestro imagenes de test"
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Precision del test: ', test_acc)


"Dibujamos la función de perdidas de los datos del test y validación"
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

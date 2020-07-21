from keras import layers
from keras import models
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import random

#  LOAD MNIST DATASET

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print('============ Dataset Split ============')
print('Training set:', train_images.shape)
print('Validation set:', test_images.shape)
print('=======================================')
print('')

#  Save labels. Need it later to create the confusion matrix
raw_test_labels = test_labels

#  Training set, N samples of each class

num_of_samples = []

cols = 5
num_of_classes = 10

fig, axs = plt.subplots(nrows=num_of_classes, ncols=cols,
                        figsize=(5, 10))
fig.tight_layout()

for i in range(cols):
    for j in range(num_of_classes):
        x_selected = train_images[train_labels == j]
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected - 1)),
                                    :, :],
                         cmap=plt.get_cmap('gray'))
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title(str(j))
            num_of_samples.append(len(x_selected))


print('======= Training Class Samples ========')
contador = 0
for num in num_of_samples:
    print('Class:', contador, 'has', num, 'samples.')
    contador = contador + 1
print('=======================================')
print('')
plt.savefig('training_samples.png')
#  Training set, N samples of each class

plt.figure(figsize=(10, 6))
plt.bar(range(0, num_of_classes), num_of_samples, width=0.8, color="blue")

for i in range(0, num_of_classes):
    plt.text(i, num_of_samples[i], str(num_of_samples[i]),
             horizontalalignment='center', fontsize=14)
plt.tick_params(labelsize=14)
plt.xticks(range(0, num_of_classes))
plt.xlabel("Clases", fontsize=16)
plt.ylabel("Nº de muestras", fontsize=16)
plt.title("Frecuencia del conjunto entrenamiento", fontsize=20)


# N Muestras de cada clase (Validacion)

num_of_samples = []

cols = 5
num_of_classes = 10

fig, axs = plt.subplots(nrows=num_of_classes, ncols=cols,
                        figsize=(5, 10))
fig.tight_layout()

for i in range(cols):
    for j in range(num_of_classes):
        x_selected = test_images[test_labels == j]
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected - 1)),
                                    :, :],
                         cmap=plt.get_cmap('gray'))
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title(str(j))
            num_of_samples.append(len(x_selected))

print('======= Validation Class Samples ======')
contador = 0
for num in num_of_samples:
    print('Class:', contador, 'has', num, 'samples.')
    contador = contador + 1
print('=======================================')
print('')


#  Training set, N samples of each class

plt.figure(figsize=(10, 6))
plt.bar(range(0, num_of_classes), num_of_samples, width=0.8, color="blue")
for i in range(0, num_of_classes):
    plt.text(i, num_of_samples[i], str(num_of_samples[i]),
             horizontalalignment='center', fontsize=14)
plt.tick_params(labelsize=14)
plt.xticks(range(0, num_of_classes))
plt.xlabel("Clases", fontsize=16)
plt.ylabel("Nº de muestras", fontsize=16)
plt.title("Frecuencia del conjunto validación", fontsize=20)
plt.savefig('digit_frequency_val.png')
plt.show()

#  NORMALIZATION AND ENCODING

#  Sample normalization and resize

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

#  Encode labels to categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


#  DATA AUGMENTATION

# Normal (Gaussian) distribution
def ruido(img):
    VARIABILITY = 0.1
    deviation = VARIABILITY * random.random()
    noise = np.random.normal(0, deviation, img.shape)
    img += noise
    np.clip(img, 0., 255.)
    return img


datagen = ImageDataGenerator(featurewise_center=False,
                             samplewise_center=False,
                             featurewise_std_normalization=False,
                             samplewise_std_normalization=False,
                             zca_whitening=False,
                             zoom_range=0.1,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             horizontal_flip=False,
                             vertical_flip=False,
                             preprocessing_function=ruido)
datagen.fit(train_images)

#  Results of the data augmentation

for X_batch, y_batch in datagen.flow(train_images, train_labels, batch_size=9):
    for i in range(0, 9):
        plt.subplot(330 + 1 + i)
        plt.imshow(X_batch[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
    plt.show()
    break

#  MODEL

model = models.Sequential()

#  Block 1 CNN
model.add(layers.Conv2D(32, (3, 3),
                        activation='relu',
                        padding='same',
                        use_bias=True,
                        input_shape=(28, 28, 1)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

#  Block 2 CNN
model.add(layers.Conv2D(64, (3, 3),
                        activation='relu',
                        padding='same',
                        use_bias=True))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

# Block 3 CNN
model.add(layers.Conv2D(64, (3, 3),
                        activation='relu',
                        padding='same',
                        use_bias=True))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.25))

#  Block 4 FC
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(10, activation='softmax'))

#  LEARNING ALGORITHM AND LOSS FUNCTION

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()


#  TRAINING THE MODEL

history = model.fit(datagen.flow(train_images, train_labels,
                                 batch_size=256),
                    steps_per_epoch=int(train_images.shape[0] / 256) + 1,
                    epochs=20,
                    validation_data=(test_images, test_labels))

#  TEST

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

#  LOSS and ACCURACY graphs

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'g', label='Conjunto entrenamiento')
plt.plot(epochs, val_loss, 'b', label='Conjunto validación')
plt.title('Curva de perdidas', fontsize=18)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.grid(alpha=0.3)
plt.legend(loc='best', shadow=True)

plt.savefig('MNIST_1_loss.png')
plt.show()

plt.clf()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'g', label='Conjunto entrenamiento')
plt.plot(epochs, val_acc, 'b', label='Conjunto validación')
plt.title('Curva de aciertos', fontsize=18)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.grid(alpha=0.3)
plt.legend(loc='best', shadow=True)

plt.savefig('MNIST_1_val.png')
plt.show()

#  CONFUSION MATRIX

val_p = np.argmax(model.predict(test_images), axis=1)

error = 0
confusion_matrix = np.zeros([10, 10])
for i in range(test_images.shape[0]):
    confusion_matrix[raw_test_labels[i], val_p[i]] += 1
    if raw_test_labels[i] != val_p[i]:
        error += 1


print("\nErrors in validation set: ", error)
print("\nError Persentage : ", (error * 100) / val_p.shape[0])
print("\nAccuracy : ", 100 - (error * 100) / val_p.shape[0])

#  Plot consufion matrix

f = plt.figure(figsize=(10, 8.5))
f.add_subplot(111)

plt.imshow(np.log2(confusion_matrix + 1), cmap="Reds")
plt.colorbar()
plt.tick_params(size=5, color="white")
plt.xticks(np.arange(0, 10), np.arange(0, 10))
plt.yticks(np.arange(0, 10), np.arange(0, 10))

threshold = confusion_matrix.max() / 2

for i in range(10):
    for j in range(10):
        plt.text(j, i, int(confusion_matrix[i, j]), horizontalalignment="center",
                 color="white" if confusion_matrix[i, j] > threshold else "black")

plt.xlabel("Predecido")
plt.ylabel("Actual")
plt.title("Matriz de confusión")
plt.savefig("MNIST_1_ConfusionMatrix.png")
plt.show()

# Test error samples plot

rows = 9
cols = 9

f = plt.figure(figsize=(2 * cols, 2 * rows))
sub_plot = 1
for i in range(test_images.shape[0]):
    if raw_test_labels[i] != val_p[i]:
        f.add_subplot(rows, cols, sub_plot)
        sub_plot += 1
        plt.imshow(test_images[i].reshape([28, 28]), cmap="Blues")
        plt.axis("off")
        plt.title(
            "T: " + str(raw_test_labels[i]) + " P:" + str(val_p[i]), y=-0.15, color="Red")
plt.savefig("MNIST_1_error_plots.png")
plt.show()

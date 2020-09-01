# -------------------------------------------------------------------------
# Implementation of deep neural network structure for handwritten digits
# -------------------------------------------------------------------------
# YEAR 2019-2020               TFG - GIET - ETSE
# -------------------------------------------------------------------------
# AUTHOR: Carles Serra Vendrell
# -------------------------------------------------------------------------

from keras import layers
from keras import models
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import random

#  CARGA DEL DATASET MNIST
#  Se utliza la función propia de keras para cargar los datos del dataset MNIST
#  Esta función nos devuelve una tupla con el conjunto de entrenamiento y el
#  connjunto de validación y test.

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Se muestra la división del dataset
print('============ Dataset Split ============')
print('Training set:', train_images.shape)
print('Test and Validation set:', test_images.shape)
print('=======================================')
print('')


#  Se guardan las etiquetas sin codificar para utilizarlas en graficar la matriz
#  de confusión.
raw_test_labels = test_labels


#  GRÁFICOS DE MUESTRAS ALEATORIAS DE LAS CLASES (CONJUNTO ENTRENAMIENTO)
#  Se grafican 5 muestras aleatorias por clase del conjunto de entrenamiento.

numero_muestras = []
numero_columnas = 5
numero_clases = 10

#  Configuración de las dimensiones del gráfico
figura, axis = plt.subplots(nrows=numero_clases, ncols=numero_columnas,
                            figsize=(6, 11))
figura.tight_layout()

#  Bucle aninado, que recorre las columnas y las clases
for i in range(numero_columnas):
    for j in range(numero_clases):

        #  Se eliguen las imagenes que pertenecen a una clase
        x = train_images[train_labels == j]

        #  Se representa una imagen aleatoriamente.
        axis[j][i].imshow(x[random.randint(0, len(x - 1)),
                            :, :],
                          cmap=plt.get_cmap('gray'))

        # Desactivar cuadricula
        axis[j][i].axis("off")

        #  Titulo de la clase en la 2 columna y se guarda el número de muestras
        #  de cada clase.
        if i == 2:
            axis[j][i].set_title(str(j))
            numero_muestras.append(len(x))

#  Se guarda el grafico en formato .png
plt.savefig('training_samples.png')

#  GRÁFICOS DE FRECUENCIA DE LAS CLASES (CONJUNTO ENTRENAMIENTO)
#  Gráfico de barras representando el número de muestras por clase del conjunto
#  entrenamiento.

#  Configuración de las dimensiones del gráfico
plt.figure(figsize=(10, 6))
plt.bar(range(0, numero_clases), numero_muestras, width=0.7, color="blue")

#  Creación de las barras del grafico, con el número de muestras por clase
for i in range(0, numero_clases):
    plt.text(i, numero_muestras[i], str(numero_muestras[i]),
             horizontalalignment='center', fontsize=14)

#  Configuración de la apariencia del gráfico.
plt.tick_params(labelsize=14)
plt.xticks(range(0, numero_clases))
plt.xlabel("Clases", fontsize=16)
plt.ylabel("Nº de muestras", fontsize=16)
plt.title("Frecuencia del conjunto entrenamiento", fontsize=20)


#  GRÁFICOS DE MUESTRAS ALEATORIAS DE LAS CLASES (CONJUNTO TEST y VALIDACIÓN)
#  Se grafican 5 muestras aleatorias por clase del conjunto de test y validación

numero_muestras = []
numero_columnas = 5
numero_clases = 10

#  Configuración de las dimensiones del gráfico
figura, axis = plt.subplots(nrows=numero_clases, ncols=numero_columnas,
                            figsize=(5, 10))
figura.tight_layout()

#  Bucle aninado, que recorre las columnas y las clases
for i in range(numero_columnas):
    for j in range(numero_clases):

        #  Se eliguen las imagenes que pertenecen a una clase
        x = test_images[test_labels == j]

        #  Se representa una imagen aleatoriamente.
        axis[j][i].imshow(x[random.randint(0, len(x) - 1),
                            :, :],
                          cmap=plt.get_cmap('gray'))

        #  Desactivar cuadricula
        axis[j][i].axis("off")

        #  Titulo de la clase en la 2 columna y se guarda el número de muestras
        #  de cada clase.
        if i == 2:
            axis[j][i].set_title(str(j))
            numero_muestras.append(len(x))

# Se guarda el grafico en formato .png
plt.savefig('validation_samples.png')

#  GRÁFICOS DE FRECUENCIA DE LAS CLASES (CONJUNTO ENTRENAMIENTO)
#  Gráfico de barras representando el número de muestras por clase del conjunto
#  validación y test.

#  Configuración de las dimensiones del gráfico.
plt.figure(figsize=(10, 6))

#  Creación de las barras del grafico, con el número de muestras por clase
plt.bar(range(0, numero_clases), numero_muestras, width=0.8, color="blue")
for i in range(0, numero_clases):
    plt.text(i, numero_muestras[i], str(numero_muestras[i]),
             horizontalalignment='center', fontsize=14)

#  Configuración de la apariencia del gráfico.
plt.tick_params(labelsize=14)
plt.xticks(range(0, numero_clases))
plt.xlabel("Clases", fontsize=16)
plt.ylabel("Nº de muestras", fontsize=16)
plt.title("Frecuencia del conjunto validación y test", fontsize=20)

# Se guarda elgráfico en formato .png
plt.savefig('digit_frequency_val.png')

# Se muestran todos los gráficos.
plt.show()

#  PREPROCESADO DE LOS DATOS

#  Las imagenes se convierten en tensores de 3 dimnsiones para poder ser
#  con las conv2d de keras.
train_images = train_images.reshape((60000, 28, 28, 1))

#  Se normalizan las imagenes en un factor 1/255 y se convierten en tipo float
train_images = train_images.astype('float32') / 255

#  Las imagenes se convierten en tensores de 3 dimnsiones para poder ser
#  con las conv2d de keras.
test_images = test_images.reshape((10000, 28, 28, 1))

#  Se normalizan las imagenes en un factor 1/255 y se convierten en tipo float
test_images = test_images.astype('float32') / 255

#  Se codifican las etiquetas como one-hot enconding
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


#  AUMENTACIÓN DE LOS DATOS

#  Función propia, ruido gaussiano
def ruido(imagen):
    varianza = 0.1
    desviacion = varianza * random.random()
    ruido = np.random.normal(0, desviacion, imagen.shape)
    imagen += ruido
    np.clip(imagen, 0., 255.)
    return imagen


# Configuración del generador de imagenes.
datagen = ImageDataGenerator(zoom_range=0.1,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             preprocessing_function=ruido)

#  Solo utilizamos aumentación en el conjunto de entrenamiento. Se indica al
#  al generador que imagenes tiene que procesar
datagen.fit(train_images)

# Se grafican las primeras 9 muestras generadas por ImageDataGenerator
for x_batch, y_batch in datagen.flow(train_images, train_labels, batch_size=9):
    for i in range(0, 9):
        plt.subplot(330 + 1 + i)
        plt.imshow(x_batch[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
    plt.show()
    break
plt.savefig('dataAugmentation.png')


#  MODELo

#  Se indica que es un modelo secuencial
model = models.Sequential()

#  Se añaden las capas al modelo

#  Bloque 1 CNN
model.add(layers.Conv2D(32, (3, 3),
                        activation='relu',
                        padding='same',
                        use_bias=True,
                        input_shape=(28, 28, 1)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

#  Bloque 2 CNN
model.add(layers.Conv2D(64, (3, 3),
                        activation='relu',
                        padding='same',
                        use_bias=True))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

# Bloque 3 CNN
model.add(layers.Conv2D(64, (3, 3),
                        activation='relu',
                        padding='same',
                        use_bias=True))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.25))

#  Bloque 4 FC
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))

#  Se configura la función de perdidas y el algoritmo de apredizaje.
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#  Visualización de los bloques y parametros del modelo implementado.
model.summary()


#  Se indica que datos alimentan al modelo en la fase de entrenamiento y en la
# de validación. En este caso los datos de entrenamiento viene generador tras
# procesar el conjunto de entrenamiento.
history = model.fit(datagen.flow(train_images, train_labels,
                                 batch_size=256),
                    steps_per_epoch=int(train_images.shape[0] / 256) + 1,
                    epochs=40,
                    validation_data=(test_images, test_labels))

#  TEST
#  Se testea la precisión del modelo en el conjunto de datos de testeo.
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

#  GRÁFICOS DE LA FUNCIÓN DE PERDIDAS Y DE ACIERTOS

#  Se obtiene los datos la función de perdidas calculados por el modelos. Tanto
#  la de entrenamiento como la de validación.
loss = history.history['loss']

#  Se obtiene los datos la función de perdidas calculados por el modelos. Tanto
#  la de entrenamiento como la de validación.
val_loss = history.history['val_loss']

#  Generación del vector de épocas
epochs = range(1, len(loss) + 1)

#  Generamos el grafico de la función de perdidas en entrenamiento y validación
plt.plot(epochs, loss, 'g', label='Conjunto entrenamiento')
plt.plot(epochs, val_loss, 'b', label='Conjunto validación')

#  Se configura la apariencia del gráficos
plt.title('Curva de perdidas', fontsize=18)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.grid(alpha=0.3)
plt.legend(loc='best', shadow=True)

#  Guardar en formato .png y mostrar el gráfico.
plt.savefig('loss.png')
plt.show()

#  Limpiar representación anterior.
plt.clf()


#  Se obtiene los datos la función de aciertos calculados por el modelos. Tanto
#  la de entrenamiento como la de validación.
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

#  Generamos el grafico de la función de perdidas en
#  entrenamiento y validación.
plt.plot(epochs, acc, 'g', label='Conjunto entrenamiento')
plt.plot(epochs, val_acc, 'b', label='Conjunto validación')
plt.title('Curva de aciertos', fontsize=18)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.grid(alpha=0.3)
plt.legend(loc='best', shadow=True)

#  Guardar en formato .png y mostrar el gráfico.
plt.savefig('val.png')
plt.show()

#  MATRIZ DE CONFUSIÓN

#  Generación de la matriz de confusión.

#  Se obtienen las predicciones del modelo.
valores_predecidos = np.argmax(model.predict(test_images), axis=1)

fallos = 0

#  Rellenamos matriz con ceros.
matriz_confusion = np.zeros([10, 10])

#  Rellenamos la matriz con las predicciones y se cuentan el número de fallos.
for i in range(test_images.shape[0]):
    matriz_confusion[raw_test_labels[i], valores_predecidos[i]] += 1
    if raw_test_labels[i] != valores_predecidos[i]:
        fallos += 1

#  Configuramos las dimensiones de la figura
f = plt.figure(figsize=(11, 9))
f.add_subplot(111)

#  Configuramos el aspecto del gráfico
plt.imshow(np.log2(matriz_confusion + 1), cmap="Reds")
plt.colorbar()
plt.tick_params(size=5, color="white")
plt.xticks(np.arange(0, 10), np.arange(0, 10))
plt.yticks(np.arange(0, 10), np.arange(0, 10))
plt.xlabel("Predecido")
plt.ylabel("Actual")
plt.title("Matriz de confusión")

#  Se establece un umbral para la coloración.
threshold = matriz_confusion.max() / 2

#  Rellenamos el grafico mediante la matriz de confusión
for i in range(10):
    for j in range(10):
        plt.text(j,
                 i,
                 int(matriz_confusion[i, j]),
                 horizontalalignment="center",
                 color="white" if matriz_confusion[i, j] > threshold else "black")

#  Guardamos en formato .png y se muestra el gráfico.
plt.savefig("confusionMatrix.png")
plt.show()

#  FIGURA CON LAS MUESTRAS PREDICHAS DE FORMA INCORRECTA

#  Dimensiones de la figura.
rows = 9
cols = 9
f = plt.figure(figsize=(2 * cols, 2 * rows))

subplot = 1
for i in range(test_images.shape[0]):

    #  Si se ha predecido correctamente.
    if raw_test_labels[i] != valores_predecidos[i]:

        # Generamos subplot
        f.add_subplot(rows, cols, subplot)
        subplot += 1

        #  Mostramos imagen mal predecida
        plt.imshow(test_images[i].reshape([28, 28]), cmap="Blues")

        #  Desactivamos la cuadricula y generamos el titulo.
        plt.axis("off")
        plt.title(
            "T: " + str(raw_test_labels[i]) + " P:" + str(valores_predecidos[i]), y=-0.15, color="Red")

#  Guardamos en formato .png y mosatramos la figura.
plt.savefig("error_plots.png")
plt.show()


#  FIGURAS DE LOS MAPAS DE CARACTERÍTICAS DE CADA CAPA

#  Entrada que utilizaremos para generar los mapas de caracteristicas.
imagen = test_images[0].reshape(1, 28, 28, 1)

#  Obtenemos las capas de salida del modelo
capas_salida = [layer.output for layer in model.layers]

#  Obtenemos los mapas de caracteristicas de la imagen
actiaciones_del_modelo = Model(inputs=model.input, outputs=capas_salida)
mapas = actiaciones_del_modelo.predict(imagen)

#  Nombres de las capas
nombre_capas = []
for layer in model.layers[1:7]:
    nombre_capas.append(layer.name)

mapas_por_columnas = 8
#  Representar el mapa de caracteríticas de cada capa de salida
for nombre_capa, mapa_capa in zip(nombre_capas, mapas):

    # Numero de caracteristicas por mapa de caracteristicas
    numero = mapa_capa.shape[-1]

    # Tamaño del mapa de caracteristicas
    tamaño = mapa_capa.shape[1]

    #  Caracteristicas por columna
    num_columnas = numero // mapas_por_columnas

    #  Generamos matriz con zeros que rellenaremos luego.
    display_grid = np.zeros(
        (tamaño * num_columnas, mapas_por_columnas * tamaño))

    #  Rellenamos matriz con las activaciones porcesadas para visualizarse.
    for col in range(num_columnas):
        for fila in range(mapas_por_columnas):

            # Procesamos cada caracteristica
            canal_imagen = mapa_capa[0,
                                     :, :,
                                     col * mapas_por_columnas + fila]
            canal_imagen -= canal_imagen.mean()
            canal_imagen /= canal_imagen.std()
            canal_imagen *= 64
            canal_imagen += 128
            canal_imagen = np.clip(canal_imagen, 0, 255).astype('uint8')

            #  Rellenamos las caracteristicas tras ser procesada en la matriz
            display_grid[col * tamaño: (col + 1) * tamaño,
                         fila * tamaño: (fila + 1) * tamaño] = canal_imagen

    #  Configuramos el tamaño de la figura
    escala = 1. / tamaño
    plt.figure(figsize=(escala * display_grid.shape[1],
                        escala * display_grid.shape[0]))

    # Titulo de la figura
    plt.title(nombre_capa)

    #  Desativamos la cuadricula y los ejes
    plt.grid(False)
    plt.axis('off')

    #  Representamos en la figura la matriz generada
    plt.imshow(display_grid, aspect='auto', cmap='Greys')

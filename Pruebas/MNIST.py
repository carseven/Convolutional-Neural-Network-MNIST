import tensorflow as tf
from tensorflow import keras

"Cargamos el dataset de MNIST."
(train_images, train_labels),(test_images, test_labels) = tf.keras.datasets.mnist.load_data()
print("Cargamos el dataset de MNIST")

"Observamos nuestro dataset"
print(train_images.shape)
print(len(train_labels))
print(train_labels)



"modelo"

model = keras.Sequential([
                          keras.layers.Flatten(input_shape=(28,28)),
                          keras.layers.Dense(128, activation=tf.nn.relu),
                          keras.layers.Dense(10, activation=tf.nn.softmax)
])
"""
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

"Entrenamos el modelo"
model.fit(train_images, train_labels, epochs=5)

"Evaluamos"
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Precision del test: ', test_acc)

"Predicciones"
predictions = model.predict(test_images)"""
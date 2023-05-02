import tensorflow as tf
import tensorflow_datasets as tfds

datos,metadatos =tfds.load('fashion_mnist', as_supervised=True,with_info=True)
print(metadatos)

datos_entrenamiento, datos_pruebas= datos['train'], datos['test']

nombre_clases = metadatos.features['label'].names
print(nombre_clases)

def  normalize(imagenes, etiquetas):
    imagenes =tf.cast(imagenes,tf.float32)
    imagenes /=255
    return imagenes,etiquetas

datos_entrenamiento=datos_entrenamiento.map(normalize)
datos_pruebas=datos_pruebas.map(normalize)

datos_entrenamiento=datos_entrenamiento.cache()
datos_pruebas=datos_pruebas.cache()

for imagen, etiqueta in datos_entrenamiento.take(1):
    break
imagen=imagen.numpy().reshape((28,28))

import matplotlib.pyplot as plt

plt.figure()
plt.imshow(imagen, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
##plt.show()

modelo=tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28,1)),
    tf.keras.layers.Dense(50,activation=tf.nn.relu),
    tf.keras.layers.Dense(50,activation=tf.nn.relu),
    tf.keras.layers.Dense(10,activation=tf.nn.softmax)
])

modelo.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

num_ej_entreaniemto=metadatos.splits['train'].num_examples
num_ej_pruebas=metadatos.splits['test'].num_examples

print(num_ej_entreaniemto)
print(num_ej_pruebas)
TAMANO_LOTE=32

datos_entrenamiento=datos_entrenamiento.repeat().shuffle(num_ej_entreaniemto).batch(TAMANO_LOTE)
datos_pruebas=datos_pruebas.batch(TAMANO_LOTE)

import math

historial= modelo.fit(datos_entrenamiento,epochs=5,steps_per_epoch=math.ceil(num_ej_entreaniemto/TAMANO_LOTE))

plt.xlabel("")
plt.ylabel("")
plt.plot(historial.history['loss'])

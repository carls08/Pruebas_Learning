import tensorflow  as tf
import numpy as np
celsius =np.array([-40, -10,0,8,15,22,38],dtype=np.float64)
farenheit = np.array([-40,14,32,46,58,72,100],dtype=np.float64)

##capa=tf.keras.layers.Dense(units=1,input_shape=[1])
##modelo=tf.keras.Sequential([capa])
oculta1=tf.keras.layers.Dense(units=3,input_shape=[1])
oculta2=tf.keras.layers.Dense(units=3)
salida=tf.keras.layers.Dense(units=1)
modelo=tf.keras.Sequential([oculta1,oculta2,salida])
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)


print("comenzar")
comenzar=modelo.fit(celsius,farenheit,epochs=1000,verbose=False)
print("modelo entrenado")

import matplotlib.pyplot as plt
plt.xlabel("Frequency")
plt.ylabel(" ")
plt.plot(comenzar.history["loss"])
plt.show()

resultado=modelo.predict([80.0])
print(resultado)

print("Variables internas del modelo " )

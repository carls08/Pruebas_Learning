import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

#Leemos el archivo .csv, delimitado por ";"
df =pd.read_csv("dataset_Facebook.csv",delimiter=";")
#aplicamos el metodo Dataframe
df=pd.DataFrame(df)

#se define el objeto de LaberEncoder
encoder=LabelEncoder()

#Se crea nueva columna para el dataframe donde escogemos solo los valores de Type y pasarlos a Enteros
df['n_type']=encoder.fit_transform(df.Type.values)

#se reemplaza los valores nan por valores de la moda
df[['Paid']]=df[['Paid']].replace(np.nan,"0.0")
df[['share']]=df[['share']].replace(np.nan,"13.0")
df[['like']]=df[['like']].replace(np.nan,"98.0")

#Variable independientes
X =df[['Paid','share','n_type']]
#Variables independientes   
y=df[['Total Interactions']]




from sklearn.model_selection import train_test_split

#utilizamos solo el 70% para entrenar
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

regresion=linear_model.LinearRegression()

#Entrenamos nuestros datos con X_train y y_train
regresion=regresion.fit(X_train,y_train)

y_pred=regresion.predict(X_test)

#Valores de la pendiente y la intercepcion
print("El valor de la pendiente es: "+str(regresion.coef_))
print("Elvalor de la b es: "+str(regresion.intercept_))

#El score de la la prediccion
r2=regresion.score(X_train,y_train)
print(r2)

#Nuestros valores de la prediccion
print(y_pred)

#omitir por ahora
#print(np.mean(np.absolute(y_pred - y_test)))
#print(np.mean(y_pred- y_test)**2)
#mse=mean_squared_error(y_test, y_pred)
#print("La raiz del error cuadratico es: ",np.sqrt(mse))

#La grafica para ver que nuestra prediccion sea lineal
plt.scatter(y_pred,y_test)
plt.xlabel("independiente")
plt.ylabel("Dependiente")
plt.plot(y_pred, y_pred, color="Red")
plt.show()





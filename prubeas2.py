from sklearn import datasets, linear_model
import pandas as pd
import matplotlib.pyplot as plt

boston = datasets.load_diabetes()
####print(boston)
print()

print('informacion Datasets')
##print(boston.keys())
print()

print('Caractetristicas del dataset: ')
##print(boston.DESCR)


print('cantidade de datos')
print(boston.data.shape)
print()

print('Nombre de columna')
##print(boston.feature_names)

X_multiple = boston.data[:,4:9]
##print(X_multiple)

y_multiple = boston.target
print(y_multiple)

from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X_multiple,y_multiple, test_size=(0.2))

regresion=linear_model.LinearRegression()
regresion.fit(X_train,y_train)

y_pred = regresion.predict(X_test)
print(regresion.score(X_train,y_train))
print(regresion.coef_, regresion.intercept_)


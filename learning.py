import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
from sklearn.metrics import mean_squared_error





##df1=pd.read_csv("dataset.csv",delimiter=",")
##print (df1)
##df1 =pd.read_csv("flare.data2",delimiter=" ")

df =pd.read_csv("dataset_Facebook.csv",delimiter=";")
##print(df)

df=pd.DataFrame(df)

##Reemplazamos las espacios vacios o nan con "0"
df=df.replace(np.nan,"0")

#Cambio de variable categorica a numerica
type_={'Photo':0,'Status':1,'Link':2,'Video':3}
df['Type']=df['Type'].map(type_)


##Filtrado de las variables independientes
filtrados=df[['Type','Paid','share']]


##Variable dependiente
dependientes=df[['Total Interactions']]
##print(dependientes)




from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(filtrados,dependientes,test_size=0.5,random_state=42)

regresion=linear_model.LinearRegression()
regresion=regresion.fit(X_train,y_train)

y_pred=regresion.predict(X_test)


r2=regresion.score(X_train,y_train)
print(r2)
print("Valor de X",regresion.coef_ + regresion.intercept_)

print(np.mean(np.absolute(y_pred - y_test)))
print(np.mean(y_pred- y_test)**2)
mse=mean_squared_error(y_test, y_pred)
print("La raiz del error cuadratico es: ",np.sqrt(mse))

plt.figure(figsize=(10,6))
plt.scatter(y_test,y_pred)
plt.plot(y_pred, y_pred, color="Red")
plt.show()
##print(y_pred)





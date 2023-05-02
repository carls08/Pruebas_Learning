import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np





##df1=pd.read_csv("dataset.csv",delimiter=",")
##print (df1)
##df1 =pd.read_csv("flare.data2",delimiter=" ")

df =pd.read_csv("dataset_Facebook.csv",delimiter=";")
print(df)

df=pd.DataFrame(df)

df=df.replace(np.nan,"0")

x=df[['Post Hour','Post Weekday','Total Interactions']]
print(x)

y=df[['like']]




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
regresion=linear_model.LinearRegression()
regresion=regresion.fit(X_train,y_train)

y_pred=regresion.predict(X_test)


r2=regresion.score(X_train,y_train)
print(r2)
print("Valor de Y",regresion.coef_,"Valor de X",regresion.intercept_)
##print(y_pred)




from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import pandas as pd

np.set_printoptions(precision =2,suppress=True,linewidth =120)
#need to have the csv feature and target files in the same directory as this script
df = pd.read_csv("diamonds.csv")
X = df.drop(columns=["price"]).to_numpy()
y = df["price"].to_numpy()
print("Daten\n",X[:20])

dt=DecisionTreeRegressor()
print("Score auf der gesamten Trainingsmenge",dt.fit(X,y).score(X,y))
print("alle CV-Scores",cross_val_score(dt,X,y,cv=5))
print("Mittelwert:Crossvalscore eines unbeschränkten Decisiontrees",
      cross_val_score(dt,X,y,cv=5).mean())
print("Standardabweichung:Crossvalscore eines unbeschränkten Decisiontrees",
      cross_val_score(dt,X,y,cv=5).std())
dt.fit(X,y)
#print(dt.feature_importances_)
print("alle Spalten mit Bedeutung \n",dt.feature_importances_)
print("alle Spalten nach Bedeutung angeordnet\n",np.argsort(dt.feature_importances_))
#Die 5 Features mit der größten Bedeutung sind 0,7,5,4,12
#Dies ergibt sich , wenn man den Decisiontree mehrfach mit verschiedenen Seeds laufen lässt





X=X[:,[0,4,5,7,12]]
#Die Spalten, die beim allgemeinen DecisionTree besonders gut abgeschnitten haben
#liefert bis jetzt den besten Score
#X=X[:,[1,2,3,4]]
#Die Spalten, die beim allgemeinen LinearRegression besonders gut abgeschnitten hab
#liefert schlechte Scores
dec=DecisionTreeRegressor(max_depth=4,min_samples_split=3)
lin=LinearRegression()
knn=KNeighborsRegressor(weights="distance")
sc=StandardScaler()
#sc=MinMaxScaler()
sc=RobustScaler()

dec.fit(X,y)

X_sc=sc.fit_transform(X)
lin.fit(X_sc,y)

print()
np.set_printoptions(precision=2,linewidth =120)
print("CV-scores:DecisionTree\n",cross_val_score(dec,X,y,cv=4).mean())
print("Feature_importance\n",dec.feature_importances_,"Indizes der Größe nach geordnet",\
      np.argsort(dec.feature_importances_)[::-1])
print("CV-scores:LinearRegression\n",cross_val_score(lin,X_sc,y,cv=4).mean())
print("Steigungsfaktoren", lin.coef_, "Indizes der Größe nach geordnet",np.argsort(abs(lin.coef_))[::-1])

print("CV-scores:KNN\n",cross_val_score(knn,X_sc,y,cv=4).mean())
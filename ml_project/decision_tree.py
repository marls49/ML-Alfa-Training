from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import pandas as pd

np.set_printoptions(precision=2,suppress=True,linewidth=120)
#need to have the csv feature and target files in the same directory as this script
df = pd.read_csv("diamonds.csv")
# FIX: Convert categorical columns to numbers
df['cut'] = df['cut'].astype('category').cat.codes
df['color'] = df['color'].astype('category').cat.codes
df['clarity'] = df['clarity'].astype('category').cat.codes
X = df.drop(columns=["price"]).to_numpy()
y = df["price"].to_numpy()
print("Data\n",X[:2000])

dt=DecisionTreeRegressor()
print("Score on entire training set",dt.fit(X,y).score(X,y))
print("All CV-Scores",cross_val_score(dt,X,y,cv=5))
print("Mean: Crossval score of unrestricted DecisionTree",
      cross_val_score(dt,X,y,cv=5).mean())
print("Std: Crossval score of unrestricted DecisionTree",
      cross_val_score(dt,X,y,cv=5).std())
dt.fit(X,y)
print("All feature importances\n",dt.feature_importances_)
print("All columns sorted by importance\n",np.argsort(dt.feature_importances_))
#The 5 most important features are 0,3,4,5,6
#This results from running the DecisionTree multiple times with different seeds

X=X[:,[0,3,4,5,6]]
#The columns that performed particularly well with unrestricted DecisionTree
#produce the best score so far
dec=DecisionTreeRegressor(max_depth=4,min_samples_split=3)
lin=LinearRegression()
knn=KNeighborsRegressor(weights="distance")
sc=RobustScaler()

dec.fit(X,y)
X_sc=sc.fit_transform(X)
lin.fit(X_sc,y)

print()
np.set_printoptions(precision=2,linewidth=120)
print("CV-scores: DecisionTree\n",cross_val_score(dec,X,y,cv=4).mean())
print("Feature importance\n",dec.feature_importances_,"Indices sorted by size",\
      np.argsort(dec.feature_importances_)[::-1])
print("CV-scores: LinearRegression\n",cross_val_score(lin,X_sc,y,cv=4).mean())
print("Coefficients", lin.coef_, "Indices sorted by size",np.argsort(abs(lin.coef_))[::-1])
print("CV-scores: KNN\n",cross_val_score(knn,X_sc,y,cv=4).mean())


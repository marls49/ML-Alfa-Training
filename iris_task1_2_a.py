from sklearn import load_iris
from sklearn import DecisionTreeClassifier
from sklearn import cross_val_score
import numpy as np

X, y = load_iris(return_X_y=True)
clf = DecisionTreeClassifier(random_state=42)
cv_scores = cross_val_score(clf, X, y, cv=5)
training_scores = clf.fit(X, y).score(X, y)

print("Cross-validation scores:", cv_scores)
print("Training score:", training_scores)


#b) 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Create sample data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
#Decision Tree 

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_test_score = dt.score(X_test, y_test)        
print("Decision Tree Test Score:", dt_test_score)


#Logistic Regression

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
lr = LogisticRegression(random_state=42, max_iter=200)
lr.fit(X_train_scaled, y_train)
lr_test_score = lr.score(X_test_scaled, y_test)
print(f"LR - Train: {lr.score(X_train_scaled, y_train):.4f},CV: {cross_val_score(lr, X_train_scaled, y_train, cv=5).mean():.4f},Test: {lr.score(X_test_scaled, y_test):.4f}")

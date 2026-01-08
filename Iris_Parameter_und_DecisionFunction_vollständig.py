"""
Aufgabe:
Schreibe eine Schleife , in der einige Parameterkombinationen abgearbeitet werden
 und ermittele jeweils den Score auf Trainings- und Testmenge.
 (Regelkonform wäre :Trainingsscore und cv-score)

 Gib das echte y , das vorausgesagte y und die Werte der Decisionfunction aus.
 Überlege, wie sich aus den Decisionfunction werten die Voraussage ergibt.
 Betrachte besonderes die Datensätze, wo Voraussage und echtes y voneinander abweichen

 Entscheide, ob es sich lohnt, den Datensatz zu skalieren
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.datasets import load_iris
from sklearn.svm import SVC


##########################################################################
# Daten besorgen und vorbereiten
X, y = load_iris(return_X_y=True)

#X=X[50:] hier würde der Datensatz nur noch virginica und versicolor enthalten
#y=y[50:]
#X=X[:,2:] wird schlechter wenn man 2 "unwichtige" Spalten entfernt
XTrain,XTest,ytrain,ytest =train_test_split(X,y,random_state=4)

sc = StandardScaler()
#sc = MinMaxScaler()
sc.fit(XTrain)
XTrain = sc.transform(XTrain)
XTest = sc.transform(XTest)

#########################################################################
# verschiedene Parameter ausprobieren: mit logspace !!

for C in np.logspace(-3,3,num=7):
    for kernel in ["linear","poly","rbf"]:
        svm= SVC(C=C,kernel=kernel)
        svm.fit(XTrain,ytrain)

        print ("Kernel= ",svm.kernel," C= ",C)

        print("Trainings_score\t",svm.score(XTrain,ytrain))
        #
        print("Test_score     \t",svm.score( XTest, ytest))

        print("======================================")

#######################################################################
# Daten besorgen und vorbereiten
# Die gesamten Iris-Daten als Trainingsmenge betrachten
y_gesamt =y
X_scaled=sc.transform(X)# alle Irisdaten noch einmal skalieren

########################################################################
# Instanz aufbauen und fit
svm2 =SVC(kernel="linear",C=0.1).fit(X_scaled,y)
svm =SVC(kernel="linear",C=0.1).fit(X,y)

##########################################################################
# Auswertung
# liefert die decision-function die richtigen Hinweise zur Klassifikation ?
ypred_gesamt_scaled = svm2.predict (X_scaled)# Die letzte svm aus der obigen Schleife arbeitet
decision_scaled = svm2.decision_function(X_scaled)

print("die folgende Aufstellung zeigt,",
      "wie sich aus der Decisionfunction die Voraussage ergibt")
tab= np.repeat("   ",y.shape[0])
for i in zip(y_gesamt, ypred_gesamt_scaled,decision_scaled,tab,X):
    print(i, end=" ")
    if i[0] != i[1]:
        print("Fehler")
    print()

######################################################################
# Noch einmal zum Thema skalieren
print("Sollte man den Irisdatensatz skalieren?")

print("Score ohne Skalierung",svm.score(X,y))

print("Score nach Standard-Skalierung",svm2.score(X_scaled,y))



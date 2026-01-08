import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale


x1=np.arange(1,8)
x2=x1**2
x3=x1**3
X=np.c_[x1,x2,x3]
print(X)
#X_neu=scale(X,axis=0,with_mean=True,with_std=False)
#unnötig, die PCA tut das von selbst
pca=PCA()
pca.fit(X)
#zentriert X
#berechnet Kovarianzmatrix von X (= X.T@X)
#berechnet Eigenwerte +Eigenvektoren
#berechnet aus den Eigenwerten singular Values(=Wurzeln)
#pca=PCA(n_components=2)
#pca.fit(X)
X_trans=pca.transform(X)
#multipliziert X mit der Matrix aus den Eigenvektoren
#Ergebnis: Punkte werden in neuem System beschrieben
print("transform",X_trans)

# Ab hier wird die Informationsverteilung untersucht,
# so dass wir beurteilen können, welche neuen Achsen wichtig sind
print("Eigenvektoren",pca.components_)
print("Singulaerwerte",pca.singular_values_)
print("Erklaerungskraft",pca.explained_variance_ratio_)
print("noise",pca.noise_variance_)


print("spannweite",X_trans.max()-X_trans.min())
#print("transformiert,skaliert",scale(X_trans))

#=============================================================

X_trans=pca.transform(X)
exp_var=pca.explained_variance_ratio_
cum_exp_var=np.cumsum(exp_var)
plt.bar(range(0, len(exp_var)), exp_var, alpha=0.5, align='center', label='Individuelle explained variance')
plt.step(range(0, len(cum_exp_var)), cum_exp_var, where='mid', label='Kumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Neue Achsen')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

#====================================================================0
#Hier sind wir überzeugt,
# dass 2 neue Achsen fast dieselbe Information liefern , wie die alten 3 Achsen
# Wir beauftragen die PCA den Datensatz sofort auf 2 neue Achsen zu transformieren
pca=PCA(n_components=2)
pca.fit(X)
X_reduziert=pca.transform(X)
print(X_reduziert.shape)

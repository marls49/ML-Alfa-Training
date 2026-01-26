# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 11:12:47 2025

@author: Bernd Ebenhoch
"""


from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

(x_train, y_train), (x_val, y_val) = keras.datasets.california_housing.load_data()

scaler_x = MinMaxScaler()
scaler_x.fit(x_train)
x_train = scaler_x.transform(x_train)
x_val = scaler_x.transform(x_val)

y_max = y_train.max()
y_train = y_train/y_max
y_val = y_val/y_max

# Sequential
model1 = keras.models.Sequential()
model1.add(keras.layers.Input(x_train.shape[1:]))
model1.add(keras.layers.Dense(300, activation="relu"))  # hidden1
model1.add(keras.layers.Dense(300, activation="relu"))  # hidden2
model1.add(keras.layers.Dense(1))  # outputlayer
model1.summary()

model1.compile(loss="mse",
               optimizer=keras.optimizers.Adam(learning_rate=0.001),
               metrics=['r2_score'])

# Wir geben uns mal die Anzahl der Schichten m Modell aus
print('Number of layers:', len(model1.layers))

# Und die Größe der Gewichtsmatrix der ersten Schicht
print('Weights of layer 1:', model1.layers[0].weights[0].shape)

# Und die Größe der Bias-Werte der ersten Schicht
print('Bias of layer 1:', model1.layers[0].weights[1].shape)

# Das Modell auf die Trainingsdaten fitten
# Die batch_size gibt an, wie viele Datenpnukte auf einmal das Modell durchlaufen

history = model1.fit(x_train, y_train, batch_size=1000,
                     epochs=500, validation_data=(x_val, y_val))

# Direkt nach dem Training können wir aus dem Modell die Lernkurven abholen
history = model1.history.history

# History ist ein Dictionary. Wir geben uns mal die keys aus
print('History Keys:', history.keys())
# Die Länge der History entsprich der anzahl der Epochen

# Wir stellen die Lernkurven grafisch dar
plt.plot(history['loss'], label='Train data')
plt.plot(history['val_loss'], label='Validation data')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Wir stellen die Lernkurven grafisch dar
plt.plot(history['r2_score'], label='Train data')
plt.plot(history['val_r2_score'], label='Validation data')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('r2_score')
plt.show()

# Das Modell auf der Festplatte speichern und wieder laden
model1.save('model1.keras')
model1 = keras.models.load_model('model1.keras')


print('Training r2_score', r2_score(y_train, model1.predict(x_train)))
print('Validation r2_score', r2_score(y_val, model1.predict(x_val)))

# Eine Prediction als Funktion der richtigen Labels darstellen

y_pred = model1.predict(x_val)

plt.scatter(y_val, y_pred, alpha=0.3, label='Predicted')
plt.plot([0, 1], [0, 1], label='Ideal', color='black')
plt.xlabel('y_true')
plt.ylabel('y_pred')
plt.legend()
plt.show()

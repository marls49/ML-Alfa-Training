# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 09:48:48 2022

@author: Bernd Ebenhoch
"""

# Wir wollen in diesem Beispiel erforschen wie sich Overfitting vermeiden
# und dadurch die Validierungs-Metrik verbessern lässt
# Das Beispiel ist primär so designt, dass ein starkes Overfitting vorliegt.


import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from tensorflow import keras


# Zufallszahlengenerator in keras initialisieren
# keras.utils.set_random_seed(0)

# Datensatz erzeugen
x, y = make_moons(noise=0.5, random_state=0, n_samples=1000)
# y = (x[:, 0] < 0.5).astype('int')

# Aufteilung in Trainings- Validation- und Validierungsdaten
x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.5, random_state=0)

x_val, x_test, y_val, y_test = train_test_split(
    x_val, y_val, test_size=0.5, random_state=0)

# Daten visualisieren
plt.scatter(x[:, 0], x[:, 1], c=y, cmap='coolwarm')
plt.title('Original Data')
plt.show()


# Modell designen für Binärklassifikation
model = keras.models.Sequential()
model.add(keras.layers.Input((2,)))
model.add(keras.layers.Dense(300, activation='relu'))
model.add(keras.layers.Dense(300, activation='relu'))
model.add(keras.layers.Dense(300, activation='relu'))
model.add(keras.layers.Dense(300, activation='relu'))
model.add(keras.layers.Dense(300, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))


# Optimizer mit bestimmter Lernrate festlegen
opt = keras.optimizers.Adam(learning_rate=0.002)  # 0.002

# Loss-Funktion, Optimizer und Metrik dem neuronalen Netz zuweisen
model.compile(loss='binary_crossentropy',
              optimizer=opt, metrics=['accuracy'])


# Eine Zusammenfassung es Modells anzeigen
model.summary()


history = model.fit(x_train, y_train, batch_size=500,
                    epochs=500,
                    validation_data=(x_val, y_val))

history = model.history

# Die Lernkurvven plotten
plt.plot(history.history['accuracy'], label='Train data')
plt.plot(history.history['val_accuracy'], label='Validation data')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train data')
plt.plot(history.history['val_loss'], label='Validation data')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Das Ergebnis beurteilen
print()
print('Evaluation from learning curves:')
print('Train_accuracy:', history.history['accuracy'][-1])
print('Validation_accuracy:', history.history['val_accuracy'][-1])

print()
print('Evaluation by evaluate:')
score = model.evaluate(x_train, y_train, verbose=False)
print('Train_accuracy', score[1])

score = model.evaluate(x_val, y_val, verbose=False)
print('Validation_accuracy:', score[1])

# Nach der Optimierung
print()
print('Evaluation after optimisation')
score = model.evaluate(x_test, y_test, verbose=False)
print('Test_accuracy:', score[1])

y_pred = model.predict(x, verbose=False)
plt.scatter(x[:, 0], x[:, 1], c=y_pred, cmap='coolwarm')
plt.show()

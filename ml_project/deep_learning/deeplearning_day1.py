#deeplearning_day1


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Eager execution = no session needed
z = tf.constant(3.0)
x = tf.Variable(2.0)

with tf.GradientTape() as tape:
    y = x**2 + z
dy_dx = tape.gradient(y, x)
print("dy/dx:", dy_dx.numpy())  # Should print 4.0

A = np.array([1,2]) 
B = tf.Variable(A)
C = B.numpy()

print("C:", C)  # Should print [1 2]


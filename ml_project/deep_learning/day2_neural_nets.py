import numpy as np

w = np.array([[1,2], [3,4]])

print("w")
print(w)

x = np.array([[10],[20]])

print()
print("x")
print(x)

#Bias term
b = np.array([[100],[200]])

print()
print("b")
print(b)    

#Neural Network output
y = np.sin(w@x + b)

print()
print("y")
print(y)    
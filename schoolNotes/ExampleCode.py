import autograd
from math import log2
import numpy as np
import matplotlib.pyplot as plt
x11 = []
x12 = []
x21 = []
x22 = []
y1 = []
y2 = []
coordinatesAndLables = [((1,3.4),-1),((1.3,3.1),-1),((1.5,2.9),-1),((3,1.4),-1),((3.4,1),-1),((3,2.6),1),((3.2,2.4),1),((3.5,2.1),1),((3.5,2.1),1),((2.6,3),1),((4,1.6),1)]
for i in coordinatesAndLables:
    t,l = i
    if(l == 1):
        x,y = t
        x11.append(x)
        x12.append(y)
        y1.append(l)
    elif(l == -1):
        x,y = t
        x21.append(x)
        x22.append(y)
        y2.append(l)

c = x11 + (x21)
k =  x12 + x22
xreal = np.array([c,k])
yreal = np.array([y1+y2])
print(x11)
print(x12)
print(x21)
print(x22)
print(y1)
print(y2)



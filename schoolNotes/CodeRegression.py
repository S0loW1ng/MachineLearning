import autograd

import numpy as np
import matplotlib.pyplot as plt
def g(w):
    return w**2

def gradient_decent(g,alpha, max,w,p):
    gradien = autograd.grad(g)
    weight_history = [w]
    cost_history = [g(x)]

    for k in range(max):
        grad_eval = gradien(w)
        w = w- alpha*grad_eval
        if p:
            plt.plot(w,g(w))
        weight_history.append(w)
        cost_history.append(g(w))
        return weight_history,cost_history



x = np.linspace(-5,5,100)
scale = 5
N = 1
w = scale*np.random.rand(N,1)

plt.plot(x,g(x))
plt.plot(w, g(w),'kx')
u = gradient_decent(g,.1,20,w,1)
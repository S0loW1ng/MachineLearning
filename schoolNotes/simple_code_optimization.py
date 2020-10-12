import autograd
import numpy as np
import matplotlib.pyplot as plt

def approx_grad(f,w0,delta):
    N =  np.linalg.norm(w0)


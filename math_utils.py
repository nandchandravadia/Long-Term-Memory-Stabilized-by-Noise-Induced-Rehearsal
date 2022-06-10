#Author: Nand Chandravadia
#email: ndc2136@columbia.edu

import numpy as np
import math
from sympy import *

def average_firing_rate(t1):
    return (1 / t1) * log((1 + exp(t1)) / 2)

def kernel(t1, t2, A, tau):
    return -((A * tau) / abs(t1 - t2)) * ((1 / exp(abs(t1 - t2) / tau)) - 1)


def compute_A_prime(A, learning_rate, g, epsilon, tau):
    return ((learning_rate * (g ** 2) * (epsilon ** 2)) / (2 * tau)) * A


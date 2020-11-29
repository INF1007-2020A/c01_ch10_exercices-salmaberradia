#!/usr/bin/env python
# -*- coding: utf-8 -*-


# TODO: Importez vos modules ici
import math
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.patches import Polygon
from scipy.integrate import quad
from typing import Callable
from cmath import polar

# TODO: Définissez vos fonctions ici (il en manque quelques unes)
def linear_values() -> np.ndarray:

    return np.linspace(-1.3, 2.5, 64)


def coordinate_conversion(cartesian_coordinates: np.ndarray) -> np.ndarray:
    a= np.zeros([len(cartesian_coordinates), 2])

    for i in range(len(cartesian_coordinates)):
        rho = np.sqrt(cartesian_coordinates[i][0] ** 2 + cartesian_coordinates[i][1] ** 2)
        phi = np.arctan2(cartesian_coordinates[i][1], cartesian_coordinates[i][0])
        polar_coordinate = (rho, phi)
        a[i] = polar_coordinate

    return a


def find_closest_index(values: np.ndarray, number: float) -> int:
    
    return np.abs(values - number).argmin()
#########
def exercice_sin() -> None:
    graph_sin(*samples_exercice_sin(sinusoid, -1, 1, 250))


def graph_sin(x, y):
    plt.plot(x, y, 'o', markersize = 2.5)
    plt.legend(['data'], loc='best')
    plt.show


def samples_exercice_sin(func, start, end, nb_samples):
    x = np.linspace(start, end, num=nb_samples, endpoint=True)
    y = np.array([func(x_i) for x_i in x])

    return x, y

def sinusoid(x):
    return x**2 * math.sin(1 / x**2) + x
#########

def definite_integral():
    return quad(integrand, -np.inf, np.inf)

def integrand(x):
    return np.exp(-x**2)    

def draw_integral():
    a, b = -4, 4
    x = np.linspace(a, b, 100)
    y = integrand(x)

    _, ax = plt.subplots()
    ax.plot(x, y, 'r', linewidth=2)
    ax.set_ylim(bottom=0)
    ax.set_xlim((a - 1, b + 1))

    ix = np.linspace(a, b)
    iy = integrand(ix)
    verts = [(a, 0), *zip(ix, iy), (b, 0)]
    poly = Polygon(verts, facecolor='0.9', edgecolor='0.5')
    ax.add_patch(poly)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')

    ax.set_xticks((a, b))
    ax.set_xticklabels((f"${a}$", f"${b}$"))
    ax.set_yticks([])

    plt.show()        

if __name__ == '__main__':
    # TODO: Appelez vos fonctions ici
    exercice_sin()    
    draw_integral()

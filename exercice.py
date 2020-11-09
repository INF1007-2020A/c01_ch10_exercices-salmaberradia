#!/usr/bin/env python
# -*- coding: utf-8 -*-


# TODO: Importez vos modules ici
import numpy as np


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


if __name__ == '__main__':
    # TODO: Appelez vos fonctions ici

    
   

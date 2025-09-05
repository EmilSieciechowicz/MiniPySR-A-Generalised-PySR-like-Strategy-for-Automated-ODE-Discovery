
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from torchdiffeq import odeint # type: ignore

from IPython.display import display
import json
import gzip


def RK4_static_integrator(candidate, evaluation_sols, initial_condition, evaluation_points, GPU_device):
    """
    Classic explicit Runge-Kutta fourth order method (general purpose). Integrates systems of du/dx = f(x). Takes a candidate 
    equation and numerically integrates it given a passed initial condition and evaluation points. evaluation_points determine
    the step size of the integrator
    Inputs:
        candidate: (function) function which returns the equation to be integrated
        evaluation_sols: (torch.tensor) single token sequence which is updated to hold the integration values for the candidate function. 
                                        (initially full of -1s)
        initial_condition: (torch.tensor) the initial condition of the integration (taken from data)
        evaluation_points: (torch.tensor) which values of x to integrate too. Equally spaced as is not an adaptive step size integrator
    Outputs:
        evaluation_sols: (torch.tensor) single token sequence holding the integration values for the candidate function.
                                        (now contains meaningful values)
    """
    # Storing butcher tableau
    a = torch.tensor([
        [0,0,0,0],
        [1/2,0,0,0],
        [0,1/2,0,0],
        [0,0,1,0]
    ], device=GPU_device)

    b = torch.tensor([1/6,1/3,1/3,1/6], device=GPU_device)

    c = torch.tensor([0,1/2,1/2,1], device=GPU_device)
    
    h   = (evaluation_points[1] - evaluation_points[0]).unsqueeze(0)

    # Setting up mesh
    evaluation_sols[0] = initial_condition

    # Calculating stages
    y_n_1 = initial_condition

    for step in range(1,evaluation_points.shape[0]):

        f1 = candidate((evaluation_points[step-1] + c[0]*h).float()).unsqueeze(0)
        f2 = candidate((evaluation_points[step-1] + c[1]*h).float()).unsqueeze(0)
        f3 = candidate((evaluation_points[step-1] + c[2]*h).float()).unsqueeze(0)
        f4 = candidate((evaluation_points[step-1] + c[3]*h).float()).unsqueeze(0)

        fs  = torch.stack([f1,f2,f3,f4])

        # # Computing Stages (not needed since not dynamical system but put anyway incase want to use later on)
        # Y1 = y_n_1 + h*(a[:,0]@fs)
        # Y2 = y_n_1 + h*(a[:,1]@fs)
        # Y3 = y_n_1 + h*(a[:,2]@fs)
        # Y4 = y_n_1 + h*(a[:,3]@fs)

        # Computing
        y_n = torch.clone((y_n_1 + h*(b@fs)))

        evaluation_sols[step] = torch.clone(y_n)

        y_n_1 = torch.clone(y_n)
    
    return evaluation_sols



def RK4_dynamic_integrator(candidate, evaluation_sols, initial_condition, evaluation_points, GPU_device):
    """
    Classic explicit Runge-Kutta fourth order method (general purpose). Integrates systems of du/dx = f(x). Takes a candidate 
    equation and numerically integrates it given a passed initial condition and evaluation points. evaluation_points determine
    the step size of the integrator
    Inputs:
        candidate: (function) function which returns the equation to be integrated
        evaluation_sols: (torch.tensor) single token sequence which is updated to hold the integration values for the candidate function. 
                                        (initially full of -1s)
        initial_condition: (torch.tensor) the initial condition of the integration (taken from data)
        evaluation_points: (torch.tensor) which values of x to integrate too. Equally spaced as is not an adaptive step size integrator
    Outputs:
        evaluation_sols: (torch.tensor) single token sequence holding the integration values for the candidate function.
                                        (now contains meaningful values)
    """
    # Storing butcher tableau
    a = torch.tensor([
        [0,0,0,0],
        [1/2,0,0,0],
        [0,1/2,0,0],
        [0,0,1,0]
    ], device=GPU_device)

    b = torch.tensor([1/6,1/3,1/3,1/6], device=GPU_device)

    c = torch.tensor([0,1/2,1/2,1], device=GPU_device)
    
    h   = (evaluation_points[1] - evaluation_points[0]).unsqueeze(0)

    # Setting up mesh
    evaluation_sols[0] = initial_condition

    # Calculating stages
    y_n_1 = initial_condition

    for step in range(1,evaluation_points.shape[0]):

        f1 = candidate((evaluation_points[step-1] + c[0]*h).float()).unsqueeze(0)
        f2 = candidate((evaluation_points[step-1] + c[1]*h).float()).unsqueeze(0)
        f3 = candidate((evaluation_points[step-1] + c[2]*h).float()).unsqueeze(0)
        f4 = candidate((evaluation_points[step-1] + c[3]*h).float()).unsqueeze(0)

        fs  = torch.stack([f1,f2,f3,f4])

        # # Computing Stages (not needed since not dynamical system but put anyway incase want to use later on)
        Y1 = y_n_1 + h*(a[:,0]@fs)
        Y2 = y_n_1 + h*(a[:,1]@fs)
        Y3 = y_n_1 + h*(a[:,2]@fs)
        Y4 = y_n_1 + h*(a[:,3]@fs)

        # Computing
        y_n = torch.clone((y_n_1 + h*(b@fs)))

        evaluation_sols[step] = torch.clone(y_n)

        y_n_1 = torch.clone(y_n)
    
    return evaluation_sols
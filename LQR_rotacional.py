# Este código foi trata da dinâmica rotacional de um drone quadrirrotor utilizando a técnica LQR (Controle Ótimo Linear)
# ============================================================================================================================= #
# Importando as bibliotecas
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import pandas as ps
from scipy.linalg import solve_continuous_are
from scipy.linalg import solve

# Parâmetros:
g = 9.8        # aceleração da gravedade
m = 1.0428     # massa total do veículo
Ixx = 1e-2     # Momento de inércia em x
Iyy = 1e-2     # Momento de inércia em y
Izz = 1.8e-2   # Momento de inércia em z
Ir = 1e-3;
Omega_r = 0;
# Sistema linear: \dot{x} = Ax + Bu
# Matriz A (Modelo de Voos 2006)
A = np.zeros((6,6))
#A[0, 1] = -Ir*Omega_r/Ixx
#A[1, 0] = Ir*Omega_r/Iyy
#A[3, 0] = 1
#A[4, 1] = 1
#A[5, 2] = 1
# Modelo mais simples
A[0, 3] = 1
A[1, 4] = 1
A[2, 5] = 1
# Matriz B
B = np.zeros((6, 3))
#B[0, 0] = 1/Ixx
#B[1, 1] = 1/Iyy
#B[2, 2] = 1/Izz
# Modelo mais simples
B[3, 0] = 1/Ixx
B[4, 1] = 1/Iyy
B[5, 2] = 1/Izz
# Matriz C
C = np.zeros((3,6))
C[0, 3] = 1
C[1, 4] = 1
C[2, 5] = 1
# Matriz D
D = np.zeros((3,3))
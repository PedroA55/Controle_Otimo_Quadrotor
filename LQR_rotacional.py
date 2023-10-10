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
# ============================================ Controlabilidade ================================================== #
# Verificar a controlabilidade do sistema
n = A.shape[0] #Verifica o número de estados dos sistema
# Matriz de controlabilidade
Co = B
for i in range(1, n):
  Co = np.hstack((Co, np.linalg.matrix_power(A, i) @ B))

#Verifica se a matriz tem posto completo
rank_Co = np.linalg.matrix_rank(Co)
if rank_Co == n:
  print("O sistema é controlável.")
else:
  print("O sistema não é controlável.")
# =========================================== Indice de Desempenho ======================================================= #  
# Índice de desempenho: J(u) = 1/2 H(x_f)* x^2(t_f) + 1/2 int_0^{t_f} [Q(t)x^2(t) + R(t)u^2(t)] dt
# Matriz H
H = np.zeros((6,6))
# Matriz Q
Q = np.zeros((6,6))
Q_elem = 0.5
Q[0, 0] = Q_elem
Q[1, 1] = Q_elem
Q[2, 2] = Q_elem
Q[3, 3] = Q_elem
Q[4, 4] = Q_elem
Q[5, 5] = Q_elem
# Matriz Q
#Q = np.zeros((3,3))
#Q[0, 0] = 5
#Q[1, 1] = 5
#Q[2, 2] = 5
# Matriz R
R = np.zeros((3,3))
R_elem = 0.01
R[0, 0] = R_elem
R[1, 1] = R_elem
R[2, 2] = R_elem
# Matrizes auxiliares
E = np.dot(B,np.dot(np.linalg.inv(R), B.T))
# ============================================= Função Ruggen-Kutta backward ============================================ #
def rk4_back(f,x0,t0,tf,h):
    # Implementa o algoritmo Runge-Kutta de 4ta ordem
    # dotx = f(t,x)
    # x0 = numpy.array([x1,...,xn]),
    # t0 : tempo inicial
    # tf : tempo final
    # h : passo de integração
    # as saídas são:
    # t : o vetor tempo
    # x : o vetor de estados
    from numpy import zeros, absolute, floor
    N = absolute(floor((tf-t0)/h)).astype(int) # Número de passos
    x = zeros((N+1, x0.size)) # Tamanho da matriz da variavel (Nx6 no caso)
    t = zeros(N+1)
    x[0, :] = x0 # Guarda que os últimos elementos são Ktf
    t[N] = tf # Guarda o tempo final da simulação
    for i in range(0, N):
        k1 = f(t[i], x[i])
        k2 = f(t[i]+h/2, x[i]-(h*k1)/2)
        k3 = f(t[i]+h/2, x[i]-(h*k2)/2)
        k4 = f(t[i]+h, x[i]-h*k3)
        x[i+1, :] = x[i, :]-(h/6)*(k1+2*k2+2*k3+k4)
        t[i+1] = t[i]+h
    return t, x
# ====================================== Solucao da Equação Dif. de Ricatti ============================================== #
def solucao_K(t, x):
    k11 = x[0]
    k12 = x[1]
    k13 = x[2]
    k14 = x[3]
    k15 = x[4]
    k16 = x[5]
    k22 = x[6]
    k23 = x[7]
    k24 = x[8]
    k25 = x[9]
    k26 = x[10]
    k33 = x[11]
    k34 = x[12]
    k35 = x[13]
    k36 = x[14]
    k44 = x[15]
    k45 = x[16]
    k46 = x[17]
    k55 = x[18]
    k56 = x[19]
    k66 = x[20]
    K = np.array([[k11,k12,k13,k14,k15,k16],[k12,k22,k23,k24,k25,k26],[k13,k23,k33,k34,k35,k36],
     [k14,k24,k34,k44,k45,k46],[k15,k25,k35,k45,k55,k56],[k16,k26,k36,k46,k56,k66]]).reshape(6,6)
    dk = -np.dot(K,A)-np.dot(np.transpose(A),K)-Q+np.dot(K,np.dot(E,K))
    dk_vec= []
    for i in range(6):
      for j in range(i,6):
        dk_vec.append(dk[i,j])
    dk_vec = np.array(dk_vec)
    dk_vec = dk_vec.reshape(x.shape)
    return dk_vec
# Condição final K(t_f) = 0
Ktf = np.zeros(21) # Valor final da matriz K(t) - É um vetor 1x21
# Tempo de simulação
t0 = 0
tf = 10
h = 1e-3
t_p, P = rk4_back(solucao_K, Ktf, t0, tf, h)
t_inv = t_p[::-1] #Inverti a lista t
# ============================================ Definições da simulação =================================================== #
n_passos = int((tf-t0)/h)
t_sim = np.linspace(t0, tf,n_passos)
t_sim = t_sim.reshape(n_passos,1)
# ============================================= Gráficos dos resultados ================================================== #
#Gráfico da evolução temporal dos elementos da matriz P(t)
plt.figure(figsize=(5, 5))
plt.plot(t_inv, P[:, 0], label='p11')
plt.plot(t_inv, P[:, 1], label='p12')
plt.plot(t_inv, P[:, 2], label='p13')
plt.plot(t_inv, P[:, 3], label='p14')
plt.plot(t_inv, P[:, 4], label='p15')
plt.plot(t_inv, P[:, 5], label='p16')
plt.plot(t_inv, P[:, 6], label='p22')
plt.plot(t_inv, P[:, 7], label='p23')
plt.plot(t_inv, P[:, 8], label='p24')
plt.plot(t_inv, P[:, 9], label='p25')
plt.plot(t_inv, P[:, 10], label='p26')
plt.plot(t_inv, P[:, 11], label='p33')
plt.plot(t_inv, P[:, 12], label='p34')
plt.plot(t_inv, P[:, 13], label='p35')
plt.plot(t_inv, P[:, 14], label='p36')
plt.plot(t_inv, P[:, 15], label='p44')
plt.plot(t_inv, P[:, 16], label='p45')
plt.plot(t_inv, P[:, 17], label='p46')
plt.plot(t_inv, P[:, 18], label='p55')
plt.plot(t_inv, P[:, 19], label='p56')
plt.plot(t_inv, P[:, 20], label='p66')
plt.legend()
plt.grid()
plt.xlabel('tempo')
plt.ylabel('Elementos matriz P(t)')
# Tendo computado todos valores da matriz P(t)
# Tenho que agora computar tudo como uma matriz tridimensional
Pt = np.zeros((len(t_p), 6, 6))
for i in range(len(t_p)):
  k11 = P[i, 0]
  k12 = P[i, 1]
  k13 = P[i, 2]
  k14 = P[i, 3]
  k15 = P[i, 4]
  k16 = P[i, 5]
  k22 = P[i, 6]
  k23 = P[i, 7]
  k24 = P[i, 8]
  k25 = P[i, 9]
  k26 = P[i, 10]
  k33 = P[i, 11]
  k34 = P[i, 12]
  k35 = P[i, 13]
  k36 = P[i, 14]
  k44 = P[i, 15]
  k45 = P[i, 16]
  k46 = P[i, 17]
  k55 = P[i, 18]
  k56 = P[i, 19]
  k66 = P[i, 20]
  Pt[i,:,:] = np.array([[k11,k12,k13,k14,k15,k16],[k12,k22,k23,k24,k25,k26],[k13,k23,k33,k34,k35,k36],
     [k14,k24,k34,k44,k45,k46],[k15,k25,k35,k45,k55,k56],[k16,k26,k36,k46,k56,k66]])
#Vamos fazer uma trajetória de referência
estados_des = np.zeros((6,len(t_p)))

# Opções de trajetórias
def traj_estab_atitude(t, estados_des):
    # Realiza a trajetória de estabilização de atitude
    # t é o tempo de simulação
    # estados é a matriz de estados desejados que deve ser preenchida
    for i in range(len(t)):
      estados_des[0, i] = 0    #phi
      estados_des[1, i] = 0    #theta
      estados_des[2, i] = 0    #psi
      estados_des[3, i] = 0    #dot_phi
      estados_des[4, i] = 0    #dot_theta
      estados_des[5, i] = 0    #dot_psi
    return estados_des
def traj_phi_15graus(t, estados_des):
    # Realiza 15 graus para um lado e para o outro
    # t é o tempo de simulação
    # estados é a matriz de estados desejados que deve ser preenchida
    for i in range(len(t)):
      estados_des[0, i] = ((15*np.pi)/180)*np.sin(2*t[i])    #phi
      estados_des[1, i] = 0    #theta
      estados_des[2, i] = 0    #psi
      estados_des[3, i] = ((15*np.pi)/180)*2*np.cos(2*t[i])   #dot_phi
      estados_des[4, i] = 0    #dot_theta
      estados_des[5, i] = 0    #dot_psi
    return estados_des
def traj_step_phi(t, estados_des):
    # Realiza a trajetória de estabilização de atitude
    # t é o tempo de simulação
    # estados é a matriz de estados desejados que deve ser preenchida
    for i in range(len(t)):
      estados_des[0, i] = ((30*np.pi)/180)  #phi
      estados_des[1, i] = 0    #theta
      estados_des[2, i] = 0    #psi
      estados_des[3, i] = 0    #dot_phi
      estados_des[4, i] = 0    #dot_theta
      estados_des[5, i] = 0    #dot_psi
    return estados_des

# Escolhendo a trajetória de referência
#r = traj_estab_atitude(t_p, estados_des)
#r = traj_phi_15graus(t_p, estados_des)
r = traj_step_phi(t_p, estados_des)

# Condição inicial do sistema
#x0 = np.array([((10*np.pi)/180), ((10*np.pi)/180), ((10*np.pi)/180), 0, 0, 0]) # c.i como no trabalho de Holger Voos 2006
x0 = np.array([0, 0, 0, 0, 0, 0]) # c.i por default
# Solução apra S(t) entendendo que as variáveis precisam ser declaradas
def solucao_s2(t_atual, g, tp, P, A, E, Q):
    s1 = g[0]
    s2 = g[1]
    s3 = g[2]
    s4 = g[3]
    s5 = g[4]
    s6 = g[5]
    s = np.array([s1, s2, s3, s4, s5, s6]).reshape(6,1)
    # Trajetória desejada r(t)
    r1 = ((30*np.pi)/180)  # phi
    r2 = 0 # theta
    r3 = 0 # psi
    r4 = 0 # dot_phi
    r5 = 0 # dot_theta
    r6 = 0 # dot_psi
    r_final = np.array([r1, r2, r3, r4, r5, r6]).reshape(6,1)
    # Interpolando a solução de Ricatti
    k11 = np.interp(t_atual, t_p, P[:,0]) # interpola elem. p11
    k12 = np.interp(t_atual, t_p, P[:,1]) # interpola elem. p12
    k13 = np.interp(t_atual, t_p, P[:,2]) # interpola elem. p13
    k14 = np.interp(t_atual, t_p, P[:,3]) # interpola elem. p14
    k15 = np.interp(t_atual, t_p, P[:,4]) # interpola elem. p15
    k16 = np.interp(t_atual, t_p, P[:,5]) # interpola elem. p16
    k22 = np.interp(t_atual, t_p, P[:,6]) # interpola elem. p22
    k23 = np.interp(t_atual, t_p, P[:,7]) # interpola elem. p23
    k24 = np.interp(t_atual, t_p, P[:,8]) # interpola elem. p24
    k25 = np.interp(t_atual, t_p, P[:,9]) # interpola elem. p25
    k26 = np.interp(t_atual, t_p, P[:,10]) # interpola elem. p26
    k33 = np.interp(t_atual, t_p, P[:,11]) # interpola elem. p33
    k34 = np.interp(t_atual, t_p, P[:,12]) # interpola elem. p34
    k35 = np.interp(t_atual, t_p, P[:,13]) # interpola elem. p35
    k36 = np.interp(t_atual, t_p, P[:,14]) # interpola elem. p36
    k44 = np.interp(t_atual, t_p, P[:,15]) # interpola elem. p44
    k45 = np.interp(t_atual, t_p, P[:,16]) # interpola elem. p45
    k46 = np.interp(t_atual, t_p, P[:,17]) # interpola elem. p46
    k55 = np.interp(t_atual, t_p, P[:,18]) # interpola elem. p55
    k56 = np.interp(t_atual, t_p, P[:,19]) # interpola elem. p56
    k66 = np.interp(t_atual, t_p, P[:,20]) # interpola elem. p66
    P_final = np.array([[k11,k12,k13,k14,k15,k16],[k12,k22,k23,k24,k25,k26],[k13,k23,k33,k34,k35,k36],
     [k14,k24,k34,k44,k45,k46],[k15,k25,k35,k45,k55,k56],[k16,k26,k36,k46,k56,k66]]).reshape(6,6)
    #indices = np.where(t_p == t_atual) # Guardo o índice do instante de tempo
    #n_passo = indices[0][0]
    #print(s)
    ds = -np.dot((np.transpose(A)-np.dot(P_final,E)),s)+ np.dot(Q,r_final)
    #print(ds)
    ds = ds.reshape(g.shape)
    #print(r_final)
    return ds
solucao_S = lambda t_atual, g : solucao_s2(t_atual, g, t_p, P, A, E, Q)
# Condição final s(t_f) = 0
stf = np.zeros(6) # Valor final do vetor s(t) - É um vetor 1x6
t2, S = rk4_back(solucao_S, stf, t0, tf, h)
t_inv2 = t2[::-1] #Inverti a lista t
# ========== Gráficos dos resultados de S ===============#
#Gráfico da evolução temporal dos elementos do vetor S(t)
plt.figure(figsize=(5, 5))
plt.plot(t_inv2, S[:, 0], label='s1')
plt.plot(t_inv2, S[:, 1], label='s2')
plt.plot(t_inv2, S[:, 2], label='s3')
plt.plot(t_inv2, S[:, 3], label='s4')
plt.plot(t_inv2, S[:, 4], label='s5')
plt.plot(t_inv2, S[:, 5], label='s6')

plt.legend()
plt.grid()
plt.xlabel('tempo')
plt.ylabel('Elementos do vetor S(t)')

# Função Ruggen-Kutta forward
def rk4_forward(f,x0,t0,tf,h):
    # Implementa o algoritmo Runge-Kutta de 4ta ordem
    # dotx = f(t,x)
    # x0 = numpy.array([x1,...,xn]),
    # t0 : tempo inicial
    # tf : tempo final
    # h : passo de integração
    # as saídas são:
    # t : o vetor tempo
    # x : o vetor de estados
    from numpy import zeros, absolute, floor
    N = absolute(floor((tf-t0)/h)).astype(int) # Número de passos
    x = zeros((N+1, x0.size)) # Tamanho da matriz da variavel (Nx3 no caso)
    t = zeros(N+1)
    x[0, :] = x0 # Guarda os estados iniciais de x0
    t[0] = t0 # Guarda o tempo inicial da simulação
    for i in range(0, N):
        k1 = f(t[i], x[i])
        k2 = f(t[i]+h/2, x[i]+(h*k1)/2)
        k3 = f(t[i]+h/2, x[i]+(h*k2)/2)
        k4 = f(t[i]+h, x[i]+h*k3)
        x[i+1, :] = x[i, :]+(h/6)*(k1+2*k2+2*k3+k4)
        t[i+1] = t[i]+h
    return t, x

def solucao_x2(t_atual, x_t, t_s, S, t_p, P, A, E):
    x1 = x_t[0]
    x2 = x_t[1]
    x3 = x_t[2]
    x4 = x_t[3]
    x5 = x_t[4]
    x6 = x_t[5]
    x = np.array([x1, x2, x3, x4, x5, x6]).reshape(6,1)
    # Interpolando o vetor s(t)
    s1 = np.interp(t_atual, t_s, S[:,0]) # interpola elem. s1
    s2 = np.interp(t_atual, t_s, S[:,1]) # interpola elem. s2
    s3 = np.interp(t_atual, t_s, S[:,2]) # interpola elem. s3
    s4 = np.interp(t_atual, t_s, S[:,3]) # interpola elem. s4
    s5 = np.interp(t_atual, t_s, S[:,4]) # interpola elem. s5
    s6 = np.interp(t_atual, t_s, S[:,5]) # interpola elem. s6
    s_final = np.array([s1, s2, s3, s4, s5, s6]).reshape(6,1)
    #print(s_final)
    # Interpolando a solução de Ricatti
    k11 = np.interp(t_atual, t_p, P[:,0]) # interpola elem. p11
    k12 = np.interp(t_atual, t_p, P[:,1]) # interpola elem. p12
    k13 = np.interp(t_atual, t_p, P[:,2]) # interpola elem. p13
    k14 = np.interp(t_atual, t_p, P[:,3]) # interpola elem. p14
    k15 = np.interp(t_atual, t_p, P[:,4]) # interpola elem. p15
    k16 = np.interp(t_atual, t_p, P[:,5]) # interpola elem. p16
    k22 = np.interp(t_atual, t_p, P[:,6]) # interpola elem. p22
    k23 = np.interp(t_atual, t_p, P[:,7]) # interpola elem. p23
    k24 = np.interp(t_atual, t_p, P[:,8]) # interpola elem. p24
    k25 = np.interp(t_atual, t_p, P[:,9]) # interpola elem. p25
    k26 = np.interp(t_atual, t_p, P[:,10]) # interpola elem. p26
    k33 = np.interp(t_atual, t_p, P[:,11]) # interpola elem. p33
    k34 = np.interp(t_atual, t_p, P[:,12]) # interpola elem. p34
    k35 = np.interp(t_atual, t_p, P[:,13]) # interpola elem. p35
    k36 = np.interp(t_atual, t_p, P[:,14]) # interpola elem. p36
    k44 = np.interp(t_atual, t_p, P[:,15]) # interpola elem. p44
    k45 = np.interp(t_atual, t_p, P[:,16]) # interpola elem. p45
    k46 = np.interp(t_atual, t_p, P[:,17]) # interpola elem. p46
    k55 = np.interp(t_atual, t_p, P[:,18]) # interpola elem. p55
    k56 = np.interp(t_atual, t_p, P[:,19]) # interpola elem. p56
    k66 = np.interp(t_atual, t_p, P[:,20]) # interpola elem. p66
    P_final = np.array([[k11,k12,k13,k14,k15,k16],[k12,k22,k23,k24,k25,k26],[k13,k23,k33,k34,k35,k36],
     [k14,k24,k34,k44,k45,k46],[k15,k25,k35,k45,k55,k56],[k16,k26,k36,k46,k56,k66]]).reshape(6,6)
    dx = np.dot((A-np.dot(E,P_final)),x)- np.dot(E,s_final)
    #print(ds)
    dx = dx.reshape(x_t.shape)
    return dx
solucao_X = lambda t_atual, x_t : solucao_x2(t_atual, x_t, t2, S, t_p, P, A, E)

# Resolvendo os estados x(t)
tx, states = rk4_forward(solucao_X, x0, t0, tf, h)

# Encontrando o sinal de controle ótimo u*(t)
u = np.zeros((3,len(t_p)))
for i in range(len(t_p)):
  u[:,i] = -np.dot(np.linalg.inv(R),np.dot(np.transpose(B),np.dot(Pt[i,:,:],states[i,:]))) -np.dot(np.linalg.inv(R),np.dot(np.transpose(B),S[i,:]))

# Plote os estados ao longo do tempo
plt.figure(figsize=(12, 6))
plt.subplot(2, 3, 1)
plt.plot(tx, states[:, 0], label='$\phi$')
plt.plot(t_p, r[0, :], label='$\phi$ desejado', linestyle = '-.')
plt.grid()
plt.xlabel('Tempo')
plt.ylabel('phi')
plt.legend()

plt.subplot(2, 3, 2)
plt.plot(tx, states[:, 1], label='$\theta$')
plt.plot(t_p, r[1, :], label='$\theta$ desejado', linestyle = '-.')
plt.grid()
plt.xlabel('Tempo')
plt.ylabel('theta')
plt.legend()

plt.subplot(2, 3, 3)
plt.plot(tx, states[:, 2], label='$\psi$')
plt.plot(t_p, r[2, :], label='$\psi$ desejado', linestyle = '-.')
plt.grid()
plt.xlabel('Tempo')
plt.ylabel('psi')
plt.legend()

plt.subplot(2, 3, 4)
plt.plot(tx, states[:, 3], label='$\dot \phi$')
plt.plot(t_p, r[3, :], label='$\dot \phi $ desejado', linestyle = '-.')
plt.grid()
plt.xlabel('Tempo')
plt.ylabel('dot_phi')
plt.legend()

plt.subplot(2, 3, 5)
plt.plot(tx, states[:, 4], label='$\dot \theta$')
plt.plot(t_p, r[4, :], label='$\dot \theta $ desejado', linestyle = '-.')
plt.grid()
plt.xlabel('Tempo')
plt.ylabel('dot_theta')
plt.legend()

plt.subplot(2, 3, 6)
plt.plot(tx, states[:, 5], label='$\dot \psi$')
plt.plot(t_p, r[5, :], label='$\dot \psi $ desejado', linestyle = '-.')
plt.grid()
plt.xlabel('Tempo')
plt.ylabel('dot_psi')
plt.legend()

plt.tight_layout()
plt.show()

# Plote o sinal de controle ótimo
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.plot(t_p, u[0, :], label=' torque $\phi$')
plt.grid()
plt.xlabel('Tempo')
plt.ylabel('Entrada (u2)')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(t_p, u[1, :], label=' torque $\theta$')
plt.grid()
plt.xlabel('Tempo')
plt.ylabel('Entrada (u3)')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(t_p, u[2, :], label=' torque $\psi$')
plt.grid()
plt.xlabel('Tempo')
plt.ylabel('Entrada (u4)')
plt.legend()

plt.tight_layout()
plt.show()
# Importamos librería genera números aleatorios
import random
# importing the required module
import matplotlib.pyplot as plt 
# Importar librería para la media
import numpy as np

# Función para desempatar máximos----------------------------------
def desempate(q, max):
    
    if 1 < q.count(max):
       # obtiene los indices de las posiciones repetidas y entrega un vector con ellas
       res = [idx for idx, item in enumerate(q) if item in q[:idx]]
       return random.choice(res)
    else:
        return q.index(max)
#--------------------------------------------------------------

# Parámetros del algoritmo-------------------------------------
E = 0.1 # Parámetro épsilon
k = 10  # Número de k-armed bandids
R = [0] * k # recompensa
Q = [0] * k # Vector estimador de Q
N = [0] * k # Vector memoria de ser elegido
# --------------------------------------------------------------

# Proceso creación de los K-armed bandids distributions -------
mu  = [0] * k
for i in range(k):
    mu[i]= random.uniform(-4 , 4) # Generación de las k medias    
    # print("mu =", mu)
# --------------------------------------------------------------

# Proceso de explora vs explotar -------------------------------
Runs = 4000 # Número de corridas

# Creacion de vector para graficar el retorno
y = [0]

# Corridas
for i in range(Runs):
    
    if E < random.random():
        # Opcion explotar
        max_val = max(Q)
        idx = desempate(Q, max_val)   
    else:
        # Opcion explorar
        idx = random.randint(0, k-1)

    # Proceso de actualizar las ganancias
    R[idx] = random.gauss(mu[idx], 0.05)
    N[idx] += 1
    Q[idx] += (1/N[idx])*(R[idx]-Q[idx])

    # y axis values
    # y.append(np.mean(R))
# -------------------------------------------------------------

#Reduccion de decimales para impresión de resultados ----------
mu2 = [round(elem, 2) for elem in mu]
Q2 = [round(elem, 2) for elem in Q]

# Impresión de resultados 
for i in range(k):
     print(f'mu[{i}] = {mu2[i]}, N[{i}]= {N[i]}, Q[{i}]= {Q2[i]}')

# Identificacion de la posición del óptimo
max_mu = max(mu2)
max_N = max(N)

if  mu2.index(max_mu) == N.index(max_N):
    print("Acierto !!! :) ")
    print(f'La posicion optima es {mu2.index(max_mu)}')
else:
    print("Fallo :(")
# --------------------------------------------------------------
# Generacion de graficos    
# x axis values
# x = list(range(0,Runs+1))
# plotting the points 
# plt.plot(x, y)
# function to show the plot
# plt.show()

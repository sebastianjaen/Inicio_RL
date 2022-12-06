# Importar librería para la media
from tkinter import Y
import numpy as np
# Importamos librería genera números aleatorios
import random
# Importamos la libreria de graficas
import matplotlib.pyplot as plt 

# Programación de funciones a usar
# Función para verificar si se violan los rangos de factibilidad de los parámetros a optimizar
def Cumple_Limites(p, l):
    
    suma = [0]*len(p)
    
    # Si algún parámetro se sale de los límites se devuelve verdadero
    for i in range(len(p)):
        if p[i] < l[i][0] or p[i] > l[i][1]:
            suma[i]=1
    
    if sum(suma) > 0:
        return False
    else:
        return True

# Función graph: Esta es una función que usa el modelo de dinámica de sistemas
def Graph(v):
    
    if 0 <= v < 600:
        return 2
    if 600 <= v < 800:
        return 2+((1.8-2)*(v-600)/(800-600))
    if 800 <= v < 1000:
        return 1.8+((1.6-1.8)*(v-800)/(1000-800))
    if 1000 <= v < 1200:
        return 1.6+((0.8-1.6)*(v-1000)/(1200-1000))
    else:
        return 0.8+((0.4-0.8)*(v-1200)/(1400-1200))

# Modelo de dinámica de sistemas. Funcion para evaluar los valores de parametros. 
# El método devuelve el varlor de una variable que quiero optimizar
def DSmodel(p):

    # Definción parámetros modelo Example sales model
    initial_sales_force = p[0]
    average_salary = p[1]
    widget_price= p[2]
    exit_rate = p[3]

    # Definición del nivel
    size_of_sales_force = 0
    # Definición de los flujos
    New_hires = 0
    Departures = 0
    # Definición de las variables auxiliares
    budgeted_size = 0
    sales_dep = 0
    annual_revenues = 0
    widget_sales = 0
    effectiveness_widgets = 0

    # Paso y tiempo total de simulacion
    dt = 0.1
    Total_time = int(20/dt) # 200 pasos
    
    for i in range(Total_time):
        if i == 0:
            size_of_sales_force = initial_sales_force
        else:
            size_of_sales_force += (New_hires - Departures) * dt
        
        effectiveness_widgets = Graph(size_of_sales_force)
        widget_sales = size_of_sales_force * effectiveness_widgets * 365
        annual_revenues = widget_sales * widget_price / 1000000
        sales_dep = annual_revenues * 0.5
        budgeted_size = sales_dep * 1000000 / average_salary
        New_hires = budgeted_size - size_of_sales_force
        Departures = size_of_sales_force * exit_rate
    
    return size_of_sales_force

# Inicio del algoritmo K-Armed bandid para optimizar parámetros del modelo de dinámica de sistemas

E = 0.1 # Parametro de exploracion
fac = [-0.1, -0.05, -0.01, 0, 0.01, 0.05, 0.1] # Posibles cambios que pueden tenre los parámetros 
P = [50, 25000, 100, 0.2] # Valor inicial de los parametros
Lim = [[25,75],[20000,30000],[98,102],[0.15,0.25]] # Rango de factibilidad de los parámetros
A = len(fac) # Número de acciones posibles
Q = np.zeros((A,A,A,A)) # Definición de la matriz Q del método K-Armed Bandid
Dim = Q.ndim # Dimensiones
Runs = 5000 # Corridas

R_inicial = DSmodel(P) # Solución inicial del modelo de dinámica de sistemas

# Impresión de parámetros iniciales y solución inicial
print("Parametros iniciales = ", P)
print("Recompensa inicial =", round(R_inicial,2))

# Creacion de vector para graficar el retorno
y = [0]
y[0] = R_inicial

# Proceso de explorar - explotar
for i in range(Runs):

    P_previo = list(P) # Guardo el valor de P previo
    R_previo = DSmodel(P_previo)
    
    # Se grafica las ganancias
    y.append(R_previo)

    if E <random.random():
        # Opcion explotar
        max_val = np.amax(Q) # Identifico el máximo valor
        result = np.where(Q == max_val) # Encuentro la posición del máximo valor
        I, J, K, L = [result[i][0] for i in range(Dim)] # Guardo la posición del máximo valor
        
    else:
        # Proceso de exploracion
        idxs = [0] * Dim
        I, J, K, L = [random.randint(0, A-1) for items in idxs] # Genero una posición aleatoria

    # Asigno un nuevo P de acuerdo a la posición elegida
    P[0] = P[0] * (1+fac[I])
    P[1] = P[1] * (1+fac[J])
    P[2] = P[2] * (1+fac[K])
    P[3] = P[3] * (1+fac[L])
    
    # Verifico que esta nueva solución P no viole los límites de factibilidad
    if Cumple_Limites(P,Lim) == True:
        # Si sí es una solución factible calculo su recompensa y guardo esta posición
        R_actual = DSmodel(P)
        Q[I][J][K][L] += ((R_actual-R_previo)/R_previo)
        # Verifico que la nueva solución sea mejor que la mejor solución anterior, sino, me quedo con la solución mejor.
        if R_previo >= R_actual:
            P = P_previo
    else:
        # A una solulción que no satisfaga la factibilidad le asigno la penalidad 
        Q[I][J][K][L] += -100
        P = P_previo

# Impresión de los resultados finales
 
P = [round(elem, 2) for elem in P] # Solución final encontrada redondeada.
print("Parametros finales = ", P)
print("Recompensa final =", round(DSmodel(P),2))

# Generacion de graficos    
# x axis values
x = list(range(0,Runs+1))
# Grilla y ejes
plt.grid(True)
plt.title('Optimización de modelo DS con K-Armed Bandid algoritmo')
plt.xlabel('Iteraciones del método')
plt.ylabel('Valor variable objetivo')
plt.text(1500, 1000, f'Soluciones: inicial= {round(y[0],2)}, final = {round(y[Runs-1],2)}')
plt.text(1500, 990, f'% Ganancia = {round((y[Runs-1]-y[0])/y[0], 2)}')
# plotting the points 
plt.plot(x, y)
# function to show the plot
plt.show()
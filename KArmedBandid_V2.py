# Importamos librería genera números aleatorios
import random
# importing the required module
import matplotlib.pyplot as plt 
# Importar librería para la media
import numpy as np
# Importamos libreria para las listas
from typing import List


class Environment:

    # Método inicializa la clase ambiente
    def __init__(self, k: int, runs: int):

        self.k = k  # Número de k-armed bandids
        self.Runs = runs

        # Proceso creación de los K-armed bandids distributions -------
        self.mu  = [0] * self.k
        for i in range(self.k):
            self.mu[i]= random.uniform(-4 , 4) # Generación de las k medias    

    # Método que muestra las observaciones
    def get_observation(self) -> int:
        return self.k

    # Método que anuncia cuando las corridas se terminan
    def is_done(self) -> bool:
        return self.Runs == 0 

    # Método que genera la recompensa una vez se hace la acción
    def action(self, id: int) -> float:
        
        if self.is_done(): 
            raise Exception("Game is over")
        self.Runs -= 1
        return random.gauss(self.mu[id], 2)

class Agent:

    # Método que inicializa la clase considerando el ambiente
    def __init__(self, env: Environment, E: int):
        
        self.E = E # Parámetro épsilon
        self.arms = env.get_observation() 
        self.R = 0 # Recompensa
        self.Q = [0] * self.arms # Vector estimador de Q
        self.N = [0] * self.arms # Vector memoria de ser elegido

    # Función para desempatar máximos----------------------------------
    def desempate(self, q, max)-> int:
    
        if 1 < q.count(max):
            # obtiene los indices de las posiciones repetidas y entrega un vector con ellas
            self.res = [idx for idx, item in enumerate(q) if item in q[:idx]]
            return random.choice(self.res)
        else:
            return q.index(max)

    # Método en el que se toma la decisión en cada paso
    def step(self, env: Environment):

        if self.E < random.random():
            # Opcion explotar
            self.max_val = max(self.Q)
            self.idx = self.desempate(self.Q, self.max_val) 
             
        else:
        # Opcion explorar
            self.idx = random.randint(0, self.arms-1)
            
    # Proceso de actualizar las ganancias
        self.R = env.action(self.idx)
        self.N[self.idx] += 1
        self.Q[self.idx] += (1/self.N[self.idx])*(self.R-self.Q[self.idx])

# Método main en el que se usan las clases agente y ambiente
   
if __name__ == "__main__":

    k = 10 # Brazos
    env = Environment(k, 10000) # Entran el numero de brazos y las corridas
    agent = Agent(env, 0.1) # Se entra el ambiente y el parámetro épsilon

    while not env.is_done():
        agent.step(env)

# Impresión de resultados 
for i in range(k):
    print(f'mu[{i}] = {round(env.mu[i],2)}, N[{i}]= {agent.N[i]}, Q[{i}]= {round(agent.Q[i],2)}')

# Identificacion de la posición del óptimo
max_mu = max(env.mu)

if  env.mu.index(max_mu) == agent.N.index(max(agent.N)):
    print("Acierto !!! :-) ")
    print(f'La posicion optima es {env.mu.index(max_mu)}')
else:
    print("Fallo :-(")
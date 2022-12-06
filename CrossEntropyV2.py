# Paquetes 
import gym
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

# Parámetros de la red
HIDDEN_SIZE = 128 # Neuronas en la capa oculta
BATCH_SIZE = 3 # Número de episodios por iteración
PERCENTILE = 70 # Percentil de descarte. Solo se queda con el 30% mejor


# Clase que construye la red
class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)

# Clase que guarda la recompensa y el paso de un episodio
Episode = namedtuple('Episode', field_names=['reward', 'steps'])

# Clase que guarda la observacion y la accion del episodio
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])

# Clase agente
class Agent:

    def __init__(self, hs: int, bs: int, percent : int):
        
        self.hidden_size = hs
        self.batch_size = bs
        self.percentile = percent

    # Metodo para hacer un episodio
    def iterate_batches(self, env, net):
        
        self.batch = [] # Lista que guarda instancias de Episode ( reward y steps)
        self.episode_reward = 0.0 # Se inicializa la recompensa
        self.episode_steps = [] # Lista que guarda instancias de EpisodeStep (observación y acción)
        self.obs = env.reset() # Se genera una observación nueva
        self.is_done = False
        
        #print("P inicial ", self.obs)
        
        self.sm = nn.Softmax(dim=1) # Se crea una capa con la función softmax

        # Proceso de iteración para cada batch
        for i in range(self.batch_size):

            self.obs_v = torch.FloatTensor([self.obs]) # Se convierte la observación en un tensor
            self.act_probs_v = self.sm(net(self.obs_v)) # Se pasa la observación por la red y se calculan las probabilidades de las acciones
            self.act_probs = self.act_probs_v.data.numpy()[0] # Conversión del tensor act_probs_v a un arreglo de probabilidades de acciones
            
            self.action = np.random.choice(len(self.act_probs), p=self.act_probs) # Genera la acción de manera aleatoria considerando las probabilidades
            
            #print("Corrida =", i)
            
            self.next_obs, self.reward, self.is_done = env.step(self.action, i) # Aplica la accion y calcula varias cosas
            self.episode_reward += self.reward
            
            #print(f' next obs {self.next_obs}, reward {self.reward}, is done {self.is_done}')

            self.paso = EpisodeStep(observation=self.obs, action=self.action)
            self.episode_steps.append(self.paso)
            
            if self.is_done:               
                self.e = Episode(reward=self.episode_reward, steps=self.episode_steps)
                self.batch.append(self.e)
                self.episode_reward = 0.0
                self.episode_steps = []
                self.next_obs = env.reset()

                #print("lenght batch", len(self.batch))
                #print("batch size ", self.batch_size-1)
                
                #if len(self.batch) == (self.batch_size-1):
                #    yield self.batch
                #self.batch = []
            self.obs = self.next_obs
            i+=1
        return self.batch

    #  Metodo para filtrar los batches y quedar con los mejores
    def filter_batch(self, batch):
        print(batch)
        self.rewards = list(map(lambda s: s.reward, batch))
        self.reward_bound = np.percentile(self.rewards, self.percentile)
        self.reward_mean = float(np.mean(self.rewards))

        self.train_obs = []
        self.train_act = []
        for reward, steps in batch:
            if reward < self.reward_bound:
                continue
            self.train_obs.extend(map(lambda step: step.observation, steps))
            self.train_act.extend(map(lambda step: step.action, steps))

        self.train_obs_v = torch.FloatTensor(self.train_obs)
        self.train_act_v = torch.LongTensor(self.train_act)
        return self.train_obs_v, self.train_act_v, self.reward_bound, self.reward_bound, self.reward_mean
    
    def crossentropy(self, env, net):
        
        self.objective = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(params=net.parameters(), lr=0.01)
        self.writer = SummaryWriter(comment="-SystemDynamics")

        for iter_no, batch in enumerate(self.iterate_batches(env, net)):
        
            self.obs_v, self.acts_v, self.reward_b, self.reward_m = \
                self.filter_batch(batch)
            self.optimizer.zero_grad()
            self.action_scores_v = net(self.obs_v)
            self.loss_v = self.objective(self.action_scores_v, self.acts_v)
            self.loss_v.backward()
            self.optimizer.step()
            print("%d: loss=%.3f, reward_mean=%.1f, rw_bound=%.1f" % (
                iter_no, self.loss_v.item(), self.reward_m, self.reward_b))
            self.writer.add_scalar("loss", self.loss_v.item(), iter_no)
            self.writer.add_scalar("reward_bound", self.reward_b, iter_no)
            self.writer.add_scalar("reward_mean", self.reward_m, iter_no)
            if self.reward_m > 1085:
                print("Solved!")
                break
        self.writer.close()




# Clase del entorno
class Environment:

    # Método inicializa la clase ambiente
    def __init__(self, p: float, r: float, a: int, runs: int):

        self.parameters = p
        self.ranges = r
        self.actions = a
        self.observation_space = len(p)
        self.Runs = runs
        self.reset_runs = int(runs)
        self.parameters_reset = list(p)
        self.reward = 0
        self.obs_size = len(p)

    # Método que resetea las observaciones
    def reset(self)-> float:
        return self.parameters_reset
    
    # Metodo que tiene el modelo de dinamica de sistemas
    def DSmodel(self, p)-> float:

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
        
            effectiveness_widgets = self.Graph(size_of_sales_force)
            widget_sales = size_of_sales_force * effectiveness_widgets * 365
            annual_revenues = widget_sales * widget_price / 1000000
            sales_dep = annual_revenues * 0.5
            budgeted_size = sales_dep * 1000000 / average_salary
            New_hires = budgeted_size - size_of_sales_force
            Departures = size_of_sales_force * exit_rate
    
        return size_of_sales_force
    
    # Función graph: Esta es una función que usa el modelo de dinámica de sistemas
    def Graph(self, v):
        
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
    
    # Función para verificar si se violan los rangos de factibilidad de los parámetros a optimizar
    def Cumple_Limites(self, p, l):
        
        self.suma = [0]*len(p)
        
        # Si algún parámetro se sale de los límites se devuelve verdadero
        for i in range(len(p)):
            if p[i] < l[i][0] or p[i] > l[i][1]:
                self.suma[i]=1
        
        if sum(self.suma) > 0:
            return False
        else:
            return True

    # Método que genera la recompensa una vez se hace la acción
    def step(self, id: int, ind: int) -> tuple:

        self.termino = False
        # Cambios de las acciones en los parametros
        self.changes = [1.01, 0.99, 1.01, 0.99, 1.01, 0.99, 1.01, 0.99]
       
        # Manejo del indice de las acciones
        idx2 = id
        idx = float(id)
        idx = int(idx/2)
       
        # Aplicación del cambio de los parámetros
        self.parameters[idx] = self.parameters[idx] * self.changes[idx2]
        
        # Verificación de que si sea factible la observación
        if self.Cumple_Limites(self.parameters, self.ranges):
            self.reward = self.DSmodel(self.parameters)
        else:
            self.reward = -100
        
        self.termina= self.is_done(ind) 
        self.tuplex = (self.parameters, self.reward, self.termina)
        
        return self.tuplex

    # Método que anuncia cuando las corridas se terminan
    def is_done(self, ind: int) -> bool:
        
        self.index = ind
        
        if self.index == (self.Runs-1):
            return True
        else:
            return False

if __name__ == "__main__":

    P = [50, 25000, 100, 0.2] # Valor inicial de los parametros
    Lim = [[25,75],[20000,30000],[98,102],[0.15,0.25]] # Rango de factibilidad de los parámetros
    n_actions = 8

    
    env = Environment(P, Lim, n_actions, BATCH_SIZE)
    net = Net(env.obs_size, HIDDEN_SIZE, n_actions)
    agent = Agent(HIDDEN_SIZE,BATCH_SIZE,PERCENTILE)

    #print(agent.iterate_batches(env, net))
    agent.crossentropy(env, net)
    
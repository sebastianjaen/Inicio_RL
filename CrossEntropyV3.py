from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

# Parámetros de la red
HIDDEN_SIZE = 128 # Neuronas en la capa oculta
BATCH_SIZE =  25 # Número de episodios por iteración
PERCENTILE = 70 # Percentil de descarte. Solo se queda con el 30% mejor
# Creacion de vector para graficar el retorno

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
        self.opt_observ = p
        self.max_reward = self.DSmodel(self.parameters)
        # Creacion de vector para graficar el retorno
    
    # Guarda el valor de la observacion maxima
    def guarda_max(self, obs :float, rewardT: float):

        if self.Cumple_Limites(obs, self.ranges):
            if rewardT > self.max_reward: 
                self.opt_observ = list(obs)
                self.max_reward = rewardT

    # Método que resetea las observaciones
    def reset(self)-> float:
        self.parameters_reset = [np.random.uniform(self.ranges[idx][0],self.ranges[idx][1]) for idx in range(len(self.ranges))]
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
    def step(self, id: int, obs: float) -> tuple:

        self.termino = False
        # Cambios de las acciones en los parametros
        self.changes = [1.01, 0.99, 1.01, 0.99, 1.01, 0.99, 1.01, 0.99]
       
        # Manejo del indice de las acciones
        idx2 = id
        idx = float(id)
        idx = int(idx/2)

        self.parameters = obs
        # Aplicación del cambio de los parámetros
        self.parameters[idx] = self.parameters[idx] * self.changes[idx2]
        
        # Verificación de que si sea factible la observación
        if self.Cumple_Limites(self.parameters, self.ranges):
            self.reward = self.DSmodel(self.parameters)   
        else:
            self.reward = 0
        
        self.guarda_max(self.parameters,self.reward)
        self.termina= self.is_done(self.reward)
        self.tuplex = (self.parameters, self.reward, self.termina)
        
        return self.tuplex

    # Método que anuncia cuando las corridas se terminan
    def is_done(self, reward: float) -> bool:
        
        if reward == 0:
            return True
        else:
            return False

# Metodo para hacer un episodio
def iterate_batches(env, net, batch_size):
    
    batch = [] # Lista que guarda instancias de Episode ( reward y steps)
    episode_reward = 0.0 # Se inicializa la recompensa
    episode_steps = [] # Lista que guarda instancias de EpisodeStep (observación y acción)
    obs = env.reset() # Se genera una observación nueva
    sm = nn.Softmax(dim=1) # Se crea una capa con la función softmax
    
    # Proceso de iteración para cada batch
    while True:
        
        obs_v = torch.FloatTensor([obs])
        act_probs_v = sm(net(obs_v))
        act_probs = act_probs_v.data.numpy()[0]
        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done = env.step(action, obs)
        
        #print(f' next obs {next_obs}, reward {reward}, is done {is_done}')

        episode_reward += reward
        step = EpisodeStep(observation=obs, action=action)
        episode_steps.append(step)
        if is_done:
            e = Episode(reward=episode_reward, steps=episode_steps)
            batch.append(e)
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs     
                
def filter_batch(batch, percentile):

    rewards = list(map(lambda s: s.reward, batch)) 
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []
    for reward, steps in batch:
        if reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, steps))
        train_act.extend(map(lambda step: step.action, steps))

    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean        


P = [50, 25000, 100, 0.2] # Valor inicial de los parametros
Lim = [[25,75],[20000,30000],[98,102],[0.15,0.25]] # Rango de factibilidad de los parámetros
n_actions = 8

    
env = Environment(P, Lim, n_actions, BATCH_SIZE)
net = Net(env.obs_size, HIDDEN_SIZE, n_actions)

objective = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=net.parameters(), lr=0.01)
writer = SummaryWriter(comment="-System Dynamics")


for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
    obs_v, acts_v, reward_b, reward_m = \
        filter_batch(batch, PERCENTILE)
    optimizer.zero_grad()
    action_scores_v = net(obs_v)
    loss_v = objective(action_scores_v, acts_v)
    loss_v.backward()
    optimizer.step()
    print("%d: loss=%.3f, reward_mean=%.1f, rw_bound=%.1f" % (
    iter_no, loss_v.item(), reward_m, reward_b))
    #print("%d: Max reward=%.1f" % (iter_no,env.max_reward))
    #writer.add_scalar("loss", loss_v.item(), iter_no)
    #writer.add_scalar("reward_bound", reward_b, iter_no)
    #writer.add_scalar("reward_mean", reward_m, iter_no)
    writer.add_scalar("Max_reward", env.max_reward, iter_no)
    if iter_no == 300:
        print("Termino con exito!")
        print("Parametros optimos =",env.opt_observ)
        print("Max reward =", env.max_reward )
        break    
writer.close()


import pandas as pd
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

# Parámetros de la red
HIDDEN_SIZE = 128 # Neuronas en la capa oculta
BATCH_SIZE =  6 # Número de episodios por iteración
PERCENTILE = 80 # Percentil de descarte. Solo se queda con el 30% mejor
N_ACTIONS = 12
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

# Importación de los datos de producción y demanda
Data = pd.read_excel(r'C:\Users\SEBASTIAN\OneDrive\Escritorio\HelloWorld\DemandaCEDI1.xlsx')
prod_reg = pd.DataFrame(Data, columns=['Producto_reg'])
prod_nav = pd.DataFrame(Data, columns=['Producto_nav'])
demand_reg = pd.DataFrame(Data, columns=['Demanda_reg'])
demand_nav = pd.DataFrame(Data, columns=['Demanda_nav'])

# Clase del modelo de dinámica de sistemas CEDI es el Environment
class Environment:

     # Método inicializa la clase modelo DS
    def __init__(self):
        
        # Definición de los niveles
        self.CEDI_reg = 15000
        self.CEDI_nav = 0
        self.BodExt_reg = 0
        self.BodExt_nav = 0
        # Definición de los flujos
        self.Prod_reg = 0
        self.Prod_nav = 0
        self.Desp_CEDI_reg_1opc = 0
        self.Desp_CEDI_reg_2opc = 0
        self.Desp_CEDI_nav_1opc = 0
        self.Desp_CEDI_nav_2opc = 0
        self.Desp_BodExt_reg_1opc = 0
        self.Desp_BodExt_reg_2opc = 0
        self.Desp_BodExt_nav_1opc = 0
        self.Desp_BodExt_nav_2opc = 0
        self.Transf_reg_BodExt = 0 
        self.Transf_nav_BodExt = 0 
        # Definición de las variables auxiliares
        self.Demanda_reg = 0
        self.Demanda_nav = 0
        self.Inv_proyectado_reg = 0
        self.Inv_proyectado_nav = 0
        self.Exceso_Inv_Proy = 0
        self.Exceso_Inv = 0
        self.costo_exeso = 0
        self.costo_envio = 0
        self.costo_BodExt = 0
        self.costo_venta_per = 0
        self.costo_step = 0
        self.despachado_reg = 0
        self.despachado_nav = 0
        self.paso = 0
        # Definición parámetros
        self.costo_exceso_Inv = 4
        self.costo_trans = 0.050
        self.costo_almac = 2
        self.costo_venta_perdida = 8
        self.limite_Transf = 2000
        self.dt = 1
        self.Tsim = 360
        self.captotal_inv = 30000 
        # Variables de decisión del agente
        self.Transf_reg = 0
        self.Transf_nav = 0
        self.Desp_reg_CEDI = 0
        self.Desp_reg_BodExt = 0
        self.Desp_nav_CEDI = 0
        self.Desp_nav_BodExt = 0
        # Acciones
        self.acciones = [[1, 0],[0, 1],[0, 0],[1, 0],[0,1],[1, 0],[0,1]]
        # Observaciones 
        self.obs = [] * 8 # Numero de observaciones
        self.obs_size = 8

    # Método que indica cuando parar
    def is_done(self):
        if self.paso == 360:
            return True
        else:
            return False

    # Método que inicializa la simulación
    def reset(self):

        # Inicializa el paso de la simulacion
        self.paso = 0
        # Inicialización de los niveles
        self.CEDI_reg = 15000
        self.CEDI_nav = 0
        self.BodExt_reg = 0
        self.BodExt_nav = 0
        # Inicialización de los flujos
        self.Prod_reg = float(prod_reg.loc[self.paso])
        self.Prod_nav = float(prod_nav.loc[self.paso])
        self.Desp_CEDI_reg_1opc = 0
        self.Desp_CEDI_reg_2opc = 0
        self.Desp_CEDI_nav_1opc = 0
        self.Desp_CEDI_nav_2opc = 0
        self.Desp_BodExt_reg_1opc = 0
        self.Desp_BodExt_reg_2opc = 0
        self.Desp_BodExt_nav_1opc = 0
        self.Desp_BodExt_nav_2opc = 0
        self.Transf_reg_BodExt = 0 
        self.Transf_nav_BodExt = 0 
        # Inicialización de las variables auxiliares
        self.Demanda_reg = float(demand_reg.loc[self.paso])
        self.Demanda_nav = float(demand_nav.loc[self.paso])
        self.Inv_proyectado_reg = 0
        self.Inv_proyectado_nav = 0
        self.Exceso_Inv = 0
        self.costo_exeso = 0
        self.costo_envio = 0
        self.costo_BodExt = 0
        self.costo_venta_per = 0
        self.costo_step = 0
        self.despachado_reg = 0
        self.despachado_nav = 0
        # Se genera la primera observacion
        self.obs = [self.Prod_reg, self.Prod_nav, self.Demanda_reg, self.Demanda_nav, self.CEDI_reg,self.CEDI_nav, self.BodExt_reg,self.BodExt_nav]
        return self.obs

    # Metodo para determinar los indices de las acciones
    def action_index(self, action : int):
        
        i = action
        if i < 4 :
            idx1 = 0
            if i < 2:
                idx2 = 3
            else:
                idx2 = 4
        if i > 3 and i < 8:
            idx1 = 1
            if i < 6:
                idx2 = 3
            else:
                idx2 = 4  
        if i > 7:
            idx1 = 2
            if i < 10:
                idx2 = 3
            else:
                idx2 = 4    
        if i % 2 == 0:
            idx3 = 5
        else:
            idx3 = 6
        
        indexs = [idx1, idx2, idx3]
        # print(indexs)
        
        actions = []
        for i in range(3):
            actions.append(self.acciones[int(indexs[i])])  
        
        decisiones = [item for sublist in actions for item in sublist]
        # print(decisiones)
        return decisiones

    # Método que hace la simulación del paso
    def step(self, action : int):
        
        # print(f'Inicia CEDI reg = {self.CEDI_reg}, CEDI nav = {self.CEDI_nav}')

        decision = self.action_index(action)
        # Variables de decisión del agente
        self.Transf_reg = decision[0]
        self.Transf_nav = decision[1]
        self.Desp_reg_CEDI = decision[2]
        self.Desp_reg_BodExt = decision[3]
        self.Desp_nav_CEDI = decision[4]
        self.Desp_nav_BodExt = decision[5]
        # Cargo la produccion y las demandas desde archivo de datos
        self.Prod_reg = float(prod_reg.loc[self.paso])
        self.Prod_nav = float(prod_nav.loc[self.paso])

        #print(f'Entra Prod reg = {self.Prod_reg}')
        #print(f'Entra Prod nav = {self.Prod_nav}')

        self.Demanda_reg = float(demand_reg.loc[self.paso])
        self.Demanda_nav = float(demand_nav.loc[self.paso])

        #print(f'Le piden Demanda reg = {self.Demanda_reg}')
        #print(f'Le piden Demanda nav = {self.Demanda_nav}')

        #Flujos de salida del CEDI y de BodEXterna regular
        self.Desp_CEDI_reg_1opc = min(self.CEDI_reg, self.Desp_reg_CEDI * self.Demanda_reg)    
        self.Desp_BodExt_reg_1opc = min(self.BodExt_reg, self.Desp_reg_BodExt * self.Demanda_reg)
        self.Desp_CEDI_reg_2opc = min(self.CEDI_reg, self.Demanda_reg - self.Desp_BodExt_reg_1opc) * self.Desp_reg_BodExt
        self.Desp_BodExt_reg_2opc = min(self.BodExt_reg, self.Demanda_reg - self.Desp_CEDI_reg_1opc) * self.Desp_reg_CEDI 
        #Flujos de salida del CEDI y de BodEXterna navidad
        self.Desp_CEDI_nav_1opc = min(self.CEDI_nav, self.Desp_nav_CEDI * self.Demanda_nav) 
        self.Desp_BodExt_nav_1opc = min(self.BodExt_nav, self.Desp_nav_BodExt * self.Demanda_nav)
        self.Desp_CEDI_nav_2opc = min(self.CEDI_nav, self.Demanda_nav - self.Desp_BodExt_nav_1opc) *self.Desp_nav_BodExt
        self.Desp_BodExt_nav_2opc = min(self.BodExt_nav, self.Demanda_nav - self.Desp_CEDI_nav_1opc) *self.Desp_nav_CEDI
        
        #print(f'Entrega CEDI reg = {self.Desp_CEDI_reg_1opc} y completa con BodExt reg = {self.Desp_BodExt_reg_2opc}')
        #print(f'Entrega CEDI nav = {self.Desp_CEDI_nav_1opc} y completa con BodExt nav = {self.Desp_BodExt_nav_2opc}')
        
        #print(f'Entrega BodEXt reg = {self.Desp_BodExt_reg_1opc} y completa con CEDI reg = {self.Desp_CEDI_reg_2opc}')
        #print(f'Entrega BodEXt nav = {self.Desp_BodExt_nav_1opc} y completa con CEDI nav = {self.Desp_CEDI_nav_2opc}')
        
        # Definición del inventario en exceso
        self.Inv_proyectado_reg = (self.Prod_reg- (self.Desp_CEDI_reg_1opc + self.Desp_CEDI_reg_2opc)) * self.dt + self.CEDI_reg
        self.Inv_proyectado_nav = (self.Prod_nav- (self.Desp_CEDI_nav_1opc + self.Desp_CEDI_nav_2opc)) * self.dt + self.CEDI_nav
        self.Exceso_Inv = max(0, (self.Inv_proyectado_reg + self.Inv_proyectado_nav)  -self.captotal_inv )

        #print("Exceso de inventario = ", self.Exceso_Inv)

        # Definición de los traslados
        self.Transf_reg_BodExt = min(min(self.Exceso_Inv * self.Transf_reg, self.CEDI_reg), self.limite_Transf)  
        self.Transf_nav_BodExt = min(min(self.Exceso_Inv * self.Transf_nav, self.CEDI_reg), self.limite_Transf) 

        #print(f'Transfiere reg BodExt = {self.Transf_reg_BodExt} y nav BodExt = {self.Transf_nav_BodExt}')

        # Ventas totales
        self.despachado_reg = self.Desp_CEDI_reg_1opc + self.Desp_CEDI_reg_2opc + self.Desp_BodExt_reg_1opc + self.Desp_BodExt_reg_2opc
        self.despachado_nav = self.Desp_CEDI_nav_1opc + self.Desp_CEDI_nav_2opc + self.Desp_BodExt_nav_1opc + self.Desp_BodExt_nav_2opc
        # Definiciòn de los costos
        self.costo_exeso = max(0,(self.CEDI_reg + self.CEDI_nav)-self.captotal_inv) * self.costo_exceso_Inv
        self.costo_envio = (self.Transf_reg_BodExt + self.Transf_nav_BodExt) * self.costo_trans
        self.costo_BodExt = (self.BodExt_reg + self.BodExt_nav) * self.costo_almac
        self.costo_venta_per = ((self.Demanda_reg + self.Demanda_nav) - (self.despachado_reg + self.despachado_nav)) * self.costo_venta_perdida
        self.costo_step = -1 * (self.costo_exeso + self.costo_envio + self.costo_BodExt + self.costo_venta_per)
        # Defición de los niveles
        self.CEDI_reg = (self.Prod_reg - (self.Desp_CEDI_reg_1opc + self.Desp_CEDI_reg_2opc + self.Transf_reg_BodExt)) * self.dt + self.CEDI_reg
        self.CEDI_nav = (self.Prod_nav - (self.Desp_CEDI_nav_1opc + self.Desp_CEDI_nav_2opc + self.Transf_nav_BodExt)) * self.dt + self.CEDI_nav
        self.BodExt_reg = (self.Transf_reg_BodExt - (self.Desp_BodExt_reg_1opc + self.Desp_BodExt_reg_2opc)) * self.dt + self.BodExt_reg
        self.BodExt_nav = (self.Transf_nav_BodExt - (self.Desp_BodExt_nav_1opc + self.Desp_BodExt_nav_2opc)) * self.dt + self.BodExt_nav 
        # Observaciones
        self.obs = [self.Prod_reg, self.Prod_nav, self.Demanda_reg, self.Demanda_nav, self.CEDI_reg,self.CEDI_nav, self.BodExt_reg,self.BodExt_nav]
        self.paso += 1
        #print("paso =",self.paso)
        self.Results_step = (self.obs,self.costo_step,self.is_done())
        return self.Results_step 

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
        
        if np.random.uniform() > 0.01:
            action = np.random.choice(len(act_probs), p=act_probs)
        else:
            action = np.random.choice(len(act_probs))

        next_obs, reward, is_done = env.step(action)
        
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

env = Environment()
net = Net(env.obs_size, HIDDEN_SIZE, N_ACTIONS)

objective = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=net.parameters(), lr=0.01)
writer = SummaryWriter(comment="-SD_CEDI_BodExt")


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
    writer.add_scalar("loss", loss_v.item(), iter_no)
    writer.add_scalar("reward_bound", reward_b, iter_no)
    writer.add_scalar("reward_mean", reward_m, iter_no)
    if reward_m > -8150943 or iter_no == 5000:
        print("Termino con exito!...creo.")
        # Imprimimos el batch final 
        Costototal = list(map(lambda s: s.reward, batch))
        for reward, steps in batch:
            Estrategia = list(map(lambda s: s.action, steps))
            break
        break    
writer.close()

# Recuperación del valor final de la FObjetivo y de la estrategia
# print(Costototal[0])
estrategia_agente = []         
for i in range(len(Estrategia)):
    estrategia_agente.append(env.action_index(Estrategia[i]))    

df = pd.DataFrame(estrategia_agente, columns=['Transf_reg','Transf_nav','Desp_reg_CEDI','Desp_reg_BodExt','Desp_nav_CEDI','Desp_nav_BodExt'])

df.to_excel(r'C:\Users\SEBASTIAN\OneDrive\Escritorio\HelloWorld\DemandaCEDI2.xlsx', index=False)
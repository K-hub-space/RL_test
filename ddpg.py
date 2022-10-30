import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np 
import os 
import gym
import random
from replaybuffer import ReplayBuffer

batch_size = 256

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#actor_net
class Actor(nn.Module):
    def __init__(self, state_space, action_space):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_space, 128)
        self.fc2 = nn.Linear(128, 300)
        self.fc3 = nn.Linear(300, 400)
        self.fc4 = nn.Linear(400, action_space)
        nn.init.uniform_(self.fc4.weight, -3*1e-3, 3*1e-3) 
        
        self.b1 = nn.BatchNorm1d(128)
        self.b2 = nn.BatchNorm1d(300)
        self.b3 = nn.BatchNorm1d(400)

    def forward(self, x):
        x = self.b1(F.relu(self.fc1(x)))
        x = self.b2(F.relu(self.fc2(x)))
        x = self.b3(F.relu(self.fc3(x)))
        return F.tanh(self.fc4(x))


#critic net
class Critic(nn.Module):
    def __init__(self, state_space, action_space):  
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_space, 128)
        self.fcA1 = nn.Linear(action_space, 256)
        self.fcS1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 300)
        self.fc3 = nn.Linear(300, 400)
        self.fc4 = nn.Linear(400, 1)
        
        self.b1 = nn.BatchNorm1d(128)
        self.b2 = nn.BatchNorm1d(256)

    def forward(self, state, action):
        x = self.b1(F.relu(self.fc1(state)))
        aOut = self.fcA1(F.relu(action))
        sOut = self.b2(F.relu(self.fcS1(x)))
        comb = F.relu(aOut+sOut)
        out = F.relu(self.fc2(comb))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out


#DDPG本体
class DDPG():
    def __init__(self, state_space, action_space, replay_buffer):

        #(actionにノイズをのせるためのやつ)
        self.exploreDistr = torch.distributions.normal.Normal(torch.zeros(action_space), torch.ones(action_space))
        
        #actor, criticを初期化
        self.actor = Actor(state_space, action_space).to(device)
        self.critic = Critic(state_space, action_space).to(device)

        #targetネットワーク初期化
        self.targActor = Actor(state_space, action_space).to(device)
        self.targCritic = Critic(state_space, action_space).to(device)

        self.actorOpt = torch.optim.Adam(self.actor.parameters(), lr=1e-4, weight_decay=0.005)
        self.criticOpt = torch.optim.Adam(self.critic.parameters(), lr=1e-3, weight_decay=0.005)

        self.loss = nn.MSELoss()
        self.gamma = 0.99
        self.rho = 0.001

        self.replay_buffer = replay_buffer

    #offpolicy行動
    def feed_forward_actor(self, state, exploreNoise=True, test=True):
        if test: 
            self.actor.eval()

        if state.shape == (1, 2):
               state = state[0]       
        
        state = torch.from_numpy(state[0]).float().to(device)

        feed_forward = self.actor(state.reshape(1,17)).detach()
        
        #行動にノイズ付加
        if exploreNoise:
            feed_forward += 0.1*self.exploreDistr.sample().to(device)

        return feed_forward.numpy()[0]

    #replaybufferへの保存
    def add_experience(self, action, state, new_state,reward, done):
        self.replay_buffer.add((action, state, new_state, reward, done))

    def train(self):
        self.actor.train()
        self.criticOpt.zero_grad()
        
        #replaybufferからミニバッチ取り出し
        batch_sample = self.replay_buffer.sample(batch_size)

        action, state, new_state, reward, done = batch_sample
 
        state = torch.from_numpy(state).float().to(device)
        action = torch.from_numpy(action).float().to(device)
        reward = torch.from_numpy(reward).float().to(device)
        new_state = torch.from_numpy(new_state).float().to(device)
        done = torch.from_numpy(done).float().to(device)

        #criticの損失関数用
        target = torch.unsqueeze(reward, 1)+torch.unsqueeze((1-done), 1)*self.gamma*(self.targCritic(new_state, self.targActor(new_state)))

        #critic更新
        critic_loss = self.loss(target.detach(), self.critic(state, action))
        critic_loss.backward()
        self.criticOpt.step()
        
        #actor更新
        self.actorOpt.zero_grad()
        policyLoss = -self.critic(state, self.actor(state)).mean()
        policyLoss.backward()
        self.actorOpt.step()

        #target更新
        self._update_target()

    #target更新関数
    def _update_target(self):
        for param1, param2 in zip(self.targActor.parameters(), self.actor.parameters()):
            param1.data *= (1-self.rho)
            param1.data += self.rho*param2.data

        for param1, param2 in zip(self.targCritic.parameters(), self.critic.parameters()):
            param1.data *= (1-self.rho)
            param1.data += self.rho*param2.data   

    def save(self, path='models/DDPG'):
        
        if 'models' in path and os.path.isdir('models') is False:
            os.mkdir('models')
        torch.save({'actor_weights': self.actor.state_dict(),
                    'critic_weights': self.critic.state_dict(),
                    'Coptimizer_param': self.criticOpt.state_dict(),
                    'Aoptimizer_param': self.actorOpt.state_dict()
                    }, path)
        print("Saved Model Weights!")

    def load(self, path='models/DDPG'):
        
        model_dict = torch.load(path)
        self.actor.load_state_dict(model_dict['actor_weights'])
        self.critic.load_state_dict(model_dict['critic_weights'])
        self.criticOpt.load_state_dict(model_dict['Coptimizer_param'])
        self.actorOpt.load_state_dict(model_dict['Aoptimizer_param'])
        print("Model Weights Loaded!")         

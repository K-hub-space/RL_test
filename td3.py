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


#critic net(2 critic networks)
class Critic(nn.Module):
    def __init__(self, state_space, action_space):  
        super(Critic, self).__init__()

        #first net
        self.fca1 = nn.Linear(state_space, 128)
        self.fcaA1 = nn.Linear(action_space, 256)
        self.fcaS1 = nn.Linear(128, 256)
        self.fca2 = nn.Linear(256, 300)
        self.fca3 = nn.Linear(300, 400)
        self.fca4 = nn.Linear(400, 1)
        
        self.b_a1 = nn.BatchNorm1d(128)
        self.b_a2 = nn.BatchNorm1d(256)

        #second net
        self.fcb1 = nn.Linear(state_space, 128)
        self.fcbA1 = nn.Linear(action_space, 256)
        self.fcbS1 = nn.Linear(128, 256)
        self.fcb2 = nn.Linear(256, 300)
        self.fcb3 = nn.Linear(300, 400)
        self.fcb4 = nn.Linear(400, 1)
        
        self.b_b1 = nn.BatchNorm1d(128)
        self.b_b2 = nn.BatchNorm1d(256)

    def forward(self, state, action):
        #first net
        x_a = self.b_a1(F.relu(self.fca1(state)))
        aaOut = self.fcaA1(F.relu(action))
        asOut = self.b_a2(F.relu(self.fcaS1(x_a)))
        comb_a = F.relu(aaOut+asOut)
        outa = F.relu(self.fca2(comb_a))
        outa = F.relu(self.fca3(outa))
        q1 = self.fca4(outa)

        #second net
        x_b = self.b_b1(F.relu(self.fcb1(state)))
        baOut = self.fcbA1(F.relu(action))
        bsOut = self.b_b2(F.relu(self.fcbS1(x_b)))
        comb_b = F.relu(baOut+bsOut)
        outb = F.relu(self.fcb2(comb_b))
        outb = F.relu(self.fcb3(outb))
        q2 = self.fcb4(outb)

        return q1, q2


#TD3 main
class TD3():
    def __init__(self, state_space, action_space, replay_buffer):

        #make noise
        self.exploreDistr = torch.distributions.normal.Normal(torch.zeros(action_space), torch.ones(action_space))
        
        #initialize actor, critic
        self.actor = Actor(state_space, action_space).to(device)
        self.critic = Critic(state_space, action_space).to(device)

        #initialize target
        self.targActor = Actor(state_space, action_space).to(device)
        self.targCritic = Critic(state_space, action_space).to(device)
        
        self.actorOpt = torch.optim.Adam(self.actor.parameters(), lr=1e-4, weight_decay=0.005)
        self.criticOpt = torch.optim.Adam(self.critic.parameters(), lr=1e-3, weight_decay=0.005) 

        self.loss = nn.MSELoss()
        self.gamma = 0.99
        self.tau = 0.001

        self.replay_buffer = replay_buffer

    #action
    def feed_forward_actor(self, state, exploreNoise=True, test=True):
        if test: 
            self.actor.eval()

        if state.shape == (1, 2):
               state = state[0]       
        
        state = torch.from_numpy(state[0]).float().to(device)

        feed_forward = self.actor(state.reshape(1,17)).detach()
        
        #add noise to the action
        if exploreNoise:
            feed_forward += 0.1*self.exploreDistr.sample().to(device)

        return feed_forward.numpy()[0]

    #save to the replaybuffer
    def add_experience(self, action, state, new_state,reward, done):
        self.replay_buffer.add((action, state, new_state, reward, done))

    def train(self, policy_update):
        self.actor.train()
        self.criticOpt.zero_grad()
        
        #get minibatch from replaybuffer
        batch_sample = self.replay_buffer.sample(batch_size)

        action, state, new_state, reward, done = batch_sample
 
        state = torch.from_numpy(state).float().to(device)
        action = torch.from_numpy(action).float().to(device)
        reward = torch.from_numpy(reward).float().to(device)
        new_state = torch.from_numpy(new_state).float().to(device)
        done = torch.from_numpy(done).float().to(device)

        q1, q2 = self.targCritic(new_state, self.targActor(new_state))


        next_qvalues = [min(q1, q2) for q1, q2 in zip(q1.detach().numpy().flatten(), q2.detach().numpy().flatten())]


        #修正かも 
        if next_qvalues == q1:
            #loss function for target critic
            target = torch.unsqueeze(reward, 1)+torch.unsqueeze((1-done), 1)*self.gamma*(q1)

        else:
            #loss function for target critic
            target = torch.unsqueeze(reward, 1)+torch.unsqueeze((1-done), 1)*self.gamma*(q2)
    

        #update critic(loss1, loss2)
        q1, q2 = self.critic(state, action)
        critic_loss_one = self.loss(target.detach(), q1)
        critic_loss_two = self.loss(target.detach(), q2)

        critic_loss_total = critic_loss_one + critic_loss_two

        critic_loss_total.backward()
        self.criticOpt.step()   
        
        #delayed acor update 
        if policy_update:
            self.actorOpt.zero_grad()
            policyLoss = -self.critic(state, self.actor(state))[0].mean()
            policyLoss.backward()
            self.actorOpt.step()


    #target update function 
    def _update_target(self):
        for param1, param2 in zip(self.targActor.parameters(), self.actor.parameters()):
            param1.data *= (1-self.tau)
            param1.data += self.tau*param2.data

        for param1, param2 in zip(self.targCritic.parameters(), self.critic.parameters()):
            param1.data *= (1-self.tau)
            param1.data += self.tau*param2.data   

    #save function
    def save(self, path='models/TD3'):
        
        if 'models' in path and os.path.isdir('models') is False:
            os.mkdir('models')
        torch.save({'actor_weights': self.actor.state_dict(),
                    'critic_weights': self.critic.state_dict(),
                    'Coptimizer_param': self.criticOpt.state_dict(),
                    'Aoptimizer_param': self.actorOpt.state_dict()
                    }, path)
        print("Saved Model Weights!")

    def load(self, path='models/TD3'):
        
        model_dict = torch.load(path)
        self.actor.load_state_dict(model_dict['actor_weights'])
        self.critic.load_state_dict(model_dict['critic_weights'])
        self.criticOpt.load_state_dict(model_dict['Coptimizer_param'])
        self.actorOpt.load_state_dict(model_dict['Aoptimizer_param'])
        print("Model Weights Loaded!")         
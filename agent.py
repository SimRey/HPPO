import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical, Beta, Normal
import copy
import math


class HybridActorNetwork(nn.Module):
    def __init__(self, state_dim, actions, net_width, a_lr):
        super(HybridActorNetwork, self).__init__()

        self.state_dim = state_dim
        self.a_lr = a_lr
        self.net_width = net_width

        self.actions = actions
        # Discrete actions
        self.d_actions = self.actions["discrete"].n
        # Continuous actions
        self.c_actions = self.actions["continuous"].shape[0]

        self.fc1 = nn.Linear(self.state_dim, self.net_width)
        self.fc2 = nn.Linear(self.net_width, self.net_width)
        
        # Discrete actions
        self.pi = nn.Linear(self.net_width, self.d_actions)
        
        # Continuous actions
        self.alpha = nn.Linear(self.net_width, self.c_actions)
        self.beta = nn.Linear(self.net_width, self.c_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=self.a_lr)
    

    def forward(self, state, dim=0):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))

        # Discrete
        pi = F.softmax(self.pi(x), dim=dim)
        # Continuous
        alpha = F.softplus(self.alpha(x)) + 1.0
        beta = F.softplus(self.beta(x)) + 1.0

        return pi, [alpha, beta]
    
    
    def get_dist(self,state):
        pi, ab = self.forward(state)

        # Discrete distribution
        dist_d = Categorical(pi)
        # Continuous distribution
        dist_c = Beta(*ab)
        return dist_d, dist_c
    
    
    def dist_mode(self,state):
        # Only for contiuous part
        _, ab = self.forward(state)
        alpha, beta = ab
        mode = (alpha) / (alpha + beta)
        return mode
       

class HybridCriticNetwork(nn.Module):
    def __init__(self, state_dim, net_width, c_lr):
        super(HybridCriticNetwork, self).__init__()

        self.state_dim = state_dim
        self.net_width = net_width
        self.c_lr = c_lr

        self.fc1 = nn.Linear(self.state_dim, self.net_width)
        self.fc2 = nn.Linear(self.net_width, self.net_width)
        self.v = nn.Linear(self.net_width, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=self.c_lr)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.v(x)
        return x



class PPO(object):
    def __init__(self, env_with_Dead, state_dim, actions, gamma=0.99, gae_lambda=0.95,
            net_width=200, lr=1e-4, policy_clip=0.2, n_epochs=10, batch_size=64,
            l2_reg=1e-3, entropy_coef=1e-3, adv_normalization=True, dist_type="Beta",
            entropy_coef_decay = 0.99):

        self.env_with_Dead = env_with_Dead
        self.s_dim = state_dim
        self.actions = actions
        self.acts_dims = self.actions["continuous"].shape[0]
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.net_width = net_width
        self.dist_type = dist_type
        self.lr = lr
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.optim_batch_size = batch_size

        self.l2_reg = l2_reg
        self.entropy_coef = entropy_coef
        self.adv_normalization = adv_normalization
        self.entropy_coef_decay = entropy_coef_decay

        self.actor = HybridActorNetwork(self.s_dim, self.actions, self.net_width, self.lr)
        self.critic = HybridCriticNetwork(self.s_dim, self.net_width, self.lr)
        
        # Replay buffer
        self.data = []
        


    def select_action(self, state):#only used when interact with the env
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float)
            pi, _ = self.actor(state)
            dist_d, dist_c = self.actor.get_dist(state)

            # Discrete
            action_d = dist_d.sample().item()
            probs_d = pi[action_d].item()

            # Continuous
            action_c = dist_c.sample()
            if self.dist_type == 'Beta':
                action_c = torch.clamp(action_c, 0, 1)
            probs_c = dist_c.log_prob(action_c).cpu().numpy().flatten()
            action_c.cpu().numpy().flatten()

        return action_d, probs_d, action_c, probs_c



    def evaluate(self, state):
        '''Deterministic Policy'''
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float)
            pi,_ = self.actor.forward(state)
            a_d = torch.argmax(pi).item()

            _, dist_c = self.actor.get_dist(state)
            if self.dist_type == 'Beta':
                a_c = self.actor.dist_mode(state)
            elif self.dist_type == 'GS_ms':
                a_c,_ = self.actor(state)
            elif self.dist_type == 'GS_m':
                a_c = self.actor(state)
            a_c = a_c.numpy().flatten()
        
        return a_d, a_c


    def train(self):
        s, acts_d, acts_c, r, s_prime, logprob_d, logprob_c, dones, dws = self.make_batch()
        self.entropy_coef *= self.entropy_coef_decay #exploring decay

        ''' Use TD+GAE+LongTrajectory to compute Advantage and TD target'''
        with torch.no_grad():
            vs = self.critic(s)
            vs_ = self.critic(s_prime)

            '''dw(dead and win) for TD_target and Adv'''
            deltas = r + self.gamma*vs_ * (1 - dws) - vs
            deltas = deltas.cpu().flatten().numpy()
            adv = [0]

            '''done for GAE'''
            for dlt, done in zip(deltas[::-1], dones.cpu().flatten().numpy()[::-1]):
                advantage = dlt + self.gamma * self.gae_lambda * adv[-1] * (1 - done)
                adv.append(advantage)
            adv.reverse()
            adv = copy.deepcopy(adv[:-1])
            adv = torch.tensor(adv).unsqueeze(1).float()
            td_target = adv + vs
            if self.adv_normalization:
                adv = (adv - adv.mean()) / ((adv.std() + 1e-8))  

        """PPO update"""
        #Slice long trajectopy into short trajectory and perform mini-batch PPO update
        optim_iter_num = int(math.ceil(s.shape[0] / self.optim_batch_size))

        for _ in range(self.n_epochs):
            #Shuffle the trajectory, Good for training
            perm = np.arange(s.shape[0])
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm)

            s, acts_d, acts_c, td_target, adv, logprob_d, logprob_c = \
                s[perm].clone(), acts_d[perm].clone(), acts_c[perm].clone(), td_target[perm].clone(), \
                    adv[perm].clone(), logprob_d[perm].clone(), logprob_c[perm].clone() 
            
            '''mini-batch PPO update'''
            for i in range(optim_iter_num):
                index = slice(i * self.optim_batch_size, min((i + 1) * self.optim_batch_size, s.shape[0]))


                #------------------------------------ Actor update ------------------------------------
                '''discrete update'''
                prob_d, _ = self.actor(s[index], dim=1)
                dist_d, _ = self.actor.get_dist(s[index])
                entropy_d = dist_d.entropy().sum(0, keepdim=True)
                logits_d = prob_d.gather(1, acts_d[index])
                ratio = torch.exp(torch.log(logits_d) - torch.log(logprob_d[index]))

                surr1 = -ratio * adv[index]
                surr2 = -torch.clamp(ratio, 1 - self.policy_clip, 1 + self.policy_clip) * adv[index]
                a_loss_d = torch.max(surr1, surr2) - self.entropy_coef * entropy_d
                

                # Deactivate continuouse parameters
                total_params = sum([1 for param in self.actor.parameters()])-1
                params_c = []
                for j in range(4):
                    val = total_params - j
                    params_c.append(val)
                
                for i, param in enumerate(self.actor.parameters()):
                    if i in params_c:
                        param.requires_grad = False

                self.actor.optimizer.zero_grad()
                a_loss_d.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor.optimizer.step()

                # Activate all parameters again
                for param in self.actor.parameters():
                    param.requires_grad = True
                
                
                '''continuous update'''
                _, prob_c = self.actor(s[index], dim=1)
                _, dist_c = self.actor.get_dist(s[index])
                entropy_c = dist_c.entropy().sum(0, keepdim=True)
                logits_c = dist_c.log_prob(acts_c[index])
                ratio = torch.exp(logits_c.sum(1,keepdim=True) - logprob_c[index].sum(1,keepdim=True))

                surr1 = -ratio * adv[index]
                surr2 = -torch.clamp(ratio, 1 - self.policy_clip, 1 + self.policy_clip) * adv[index]
                a_loss_c = torch.max(surr1, surr2) - self.entropy_coef * entropy_c

                # Deactivate discrete parameters
                total_params = sum([1 for param in self.actor.parameters()])-1
                params_d = []
                for j in range(6):
                    if j >= 4:
                        val = total_params - j
                        params_d.append(val)
                
                for i, param in enumerate(self.actor.parameters()):
                    if i in params_d:
                        param.requires_grad = False

                self.actor.optimizer.zero_grad()
                a_loss_c.mean().backward()
                self.actor.optimizer.step()

                # Activate all parameters again
                for param in self.actor.parameters():
                    param.requires_grad = True
                
                '''critic update'''
                c_loss = F.mse_loss(td_target[index], self.critic(s[index]))
                for name, param in self.critic.named_parameters():
                    if 'weight' in name:
                        c_loss += param.pow(2).sum() * self.l2_reg

                self.critic.optimizer.zero_grad()
                c_loss.backward()
                self.critic.optimizer.step()
        return [a_loss_d, a_loss_c], c_loss, [entropy_d, entropy_c]

        
    def make_batch(self):
        l = len(self.data)
        s_lst, acts_d_lst, acts_c_lst, r_lst, s_prime_lst, logprob_d_lst, logprob_c_lst,\
            done_lst, dw_lst = np.zeros((l,self.s_dim)),  np.zeros((l, 1)),  np.zeros((l,self.acts_dims)),\
                np.zeros((l,1)), np.zeros((l,self.s_dim)), np.zeros((l, 1)),  np.zeros((l,self.acts_dims)),\
                    np.zeros((l,1)), np.zeros((l,1))
            
        for i,transition in enumerate(self.data):
            s_lst[i], acts_d_lst[i], acts_c_lst[i], r_lst[i], s_prime_lst[i], logprob_d_lst[i], logprob_c_lst[i],\
            done_lst[i], dw_lst[i] = transition
        
        if not self.env_with_Dead:
            dw_lst *=False

        self.data = [] #Clean history trajectory

        '''list to tensor'''
        with torch.no_grad():
            s, acts_d, acts_c, r, s_prime, logprob_d, logprob_c, dones, dws = \
                torch.tensor(s_lst, dtype=torch.float), \
                torch.tensor(acts_d_lst, dtype=torch.int64), \
                torch.tensor(acts_c_lst, dtype=torch.float), \
                torch.tensor(r_lst, dtype=torch.float), \
                torch.tensor(s_prime_lst, dtype=torch.float), \
                torch.tensor(logprob_d_lst, dtype=torch.float), \
                torch.tensor(logprob_d_lst, dtype=torch.float), \
                torch.tensor(done_lst, dtype=torch.float), \
                torch.tensor(dw_lst, dtype=torch.float)

        return s, acts_d, acts_c, r, s_prime, logprob_d, logprob_c, dones, dws 

    
    def put_data(self, transition):
        self.data.append(transition)

    def save(self, episode):
        torch.save(self.critic.state_dict(), f"./model/ppo_critic{episode}.pth")
        torch.save(self.actor.state_dict(), f"./model/ppo_actor{episode}.pth")
    
    def best_save(self):
        torch.save(self.critic.state_dict(), f"./best_model/ppo_critic.pth")
        torch.save(self.actor.state_dict(), f"./best_model/ppo_actor.pth")
    
    def load(self,episode):
        self.critic.load_state_dict(torch.load(f"./model/ppo_critic{episode}.pth"))
        self.actor.load_state_dict(torch.load(f"./model/ppo_actor{episode}.pth"))
    
    def load_best(self):
        self.critic.load_state_dict(torch.load(f"./best_model/ppo_critic.pth"))
        self.actor.load_state_dict(torch.load(f"./best_model/ppo_actor.pth"))
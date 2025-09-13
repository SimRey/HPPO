import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical, Beta
import copy
import math


class HybridActorNetwork(nn.Module):
    def __init__(self, state_dim, actions, net_width, a_lr=None):
        super(HybridActorNetwork, self).__init__()

        self.state_dim = state_dim
        self.net_width = net_width
        self.actions = actions

        # Discrete actions count
        self.d_actions = int(self.actions["discrete"].n)
        # Continuous actions dims
        self.c_actions = int(self.actions["continuous"].shape[0])

        # shared body
        self.fc1 = nn.Linear(self.state_dim, self.net_width)
        self.fc2 = nn.Linear(self.net_width, self.net_width)
        self.fc3 = nn.Linear(self.net_width, self.net_width)

        # discrete head
        self.pi = nn.Linear(self.net_width, self.d_actions)

        # continuous head -> Beta params
        self.alpha = nn.Linear(self.net_width, self.c_actions)
        self.beta = nn.Linear(self.net_width, self.c_actions)

        # convenience param groups (collected by name to be robust)
        self._discrete_param_names = [n for n, _ in self.pi.named_parameters()]
        # continuous param names (alpha & beta layers)
        self._continuous_param_names = list(self.alpha._parameters.keys()) + list(self.beta._parameters.keys())

    def forward(self, state):
        """
        state: tensor of shape (B, state_dim) or (state_dim,)
        returns:
            pi: (B, d_actions) or (d_actions,)
            (alpha, beta): each (B, c_actions) or (c_actions,)
        """
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))

        # Discrete probabilities: softmax over last dim
        pi = F.softmax(self.pi(x), dim=-1)

        # Continuous Beta concentrations: keep > 1 for nicer shapes (softplus + 1)
        alpha = F.softplus(self.alpha(x)) + 1.0
        beta = F.softplus(self.beta(x)) + 1.0

        return pi, (alpha, beta)

    def get_dist(self, state):
        """
        returns Categorical and Beta distributions for given state (batched).
        """
        pi, (alpha, beta) = self.forward(state)
        dist_d = Categorical(probs=pi)
        dist_c = Beta(alpha, beta)
        return dist_d, dist_c

    def dist_mode(self, state):
        """
        Return deterministic continuous action (Beta mode when defined,
        otherwise fallback to mean). Works for batched states.
        """
        _, (alpha, beta) = self.forward(state)
        denom = alpha + beta - 2.0
        # Compute mode only when denom > 0 (i.e., alpha>1 and beta>1)
        mode = torch.where(denom > 0.0, (alpha - 1.0) / denom, alpha / (alpha + beta))
        return mode


class HybridCriticNetwork(nn.Module):
    def __init__(self, state_dim, net_width):
        super(HybridCriticNetwork, self).__init__()

        self.state_dim = state_dim
        self.net_width = net_width

        self.fc1 = nn.Linear(self.state_dim, self.net_width)
        self.fc2 = nn.Linear(self.net_width, self.net_width)
        self.v = nn.Linear(self.net_width, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        v = self.v(x)
        return v


class PPO(object):
    def __init__(self, env_with_Dead, state_dim, actions, device=None,
                 gamma=0.99, gae_lambda=0.95,
                 net_width=200, lr=1e-4, policy_clip=0.2, n_epochs=10, batch_size=64,
                 l2_reg=1e-3, entropy_coef=1e-3, adv_normalization=True,
                 entropy_coef_decay=0.99):

        self.env_with_Dead = env_with_Dead
        self.s_dim = state_dim
        self.actions = actions
        self.acts_dims = int(self.actions["continuous"].shape[0])

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.net_width = net_width
        self.lr = lr
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.optim_batch_size = batch_size

        self.l2_reg = l2_reg
        self.entropy_coef = entropy_coef
        self.adv_normalization = adv_normalization
        self.entropy_coef_decay = entropy_coef_decay

        # device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # networks
        self.actor = HybridActorNetwork(self.s_dim, self.actions, self.net_width).to(self.device)
        self.critic = HybridCriticNetwork(self.s_dim, self.net_width).to(self.device)

        # create separate optimizers for discrete actor head, continuous head, and critic
        # Collect parameters robustly using named_parameters:
        discrete_params = list(self.actor.pi.parameters())
        continuous_params = list(self.actor.alpha.parameters()) + list(self.actor.beta.parameters())
        # include shared body optionally in both optimizers? Usually shared body should be updated together.
        # We'll include the shared body in both updates (discrete and continuous) by giving both optimizers the shared params.
        shared_params = [p for n, p in self.actor.named_parameters() if not (n.startswith("pi.") or n.startswith("alpha.") or n.startswith("beta."))]
        # discrete optimizer updates discrete head + shared body
        self.optimizer_d = optim.Adam(discrete_params + shared_params, lr=self.lr)
        # continuous optimizer updates continuous head + shared body
        self.optimizer_c = optim.Adam(continuous_params + shared_params, lr=self.lr)
        # critic optimizer
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)

        # Replay buffer
        self.data = []

    def select_action(self, state):
        """
        Interact with environment: returns (a_d: int, logp_d: float scalar),
                                         (a_c: np.array shape (c_actions,), logp_c_sum: float scalar)
        """
        state_t = torch.tensor(state, dtype=torch.float, device=self.device)
        # ensure batch dim
        if state_t.ndim == 1:
            state_t = state_t.unsqueeze(0)  # (1, state_dim)

        with torch.no_grad():
            dist_d, dist_c = self.actor.get_dist(state_t)
            a_d = dist_d.sample()                 # shape (1,)
            logp_d = dist_d.log_prob(a_d)         # shape (1,)
            a_c = dist_c.sample()                 # shape (1, c_actions)
            logp_c = dist_c.log_prob(a_c)         # shape (1, c_actions)
            logp_c_sum = logp_c.sum(dim=-1, keepdim=True)  # shape (1,1)

            # clamp continuous action to [0,1] if env expects that
            a_c_clamped = torch.clamp(a_c, 0.0, 1.0)

        # convert to numpy/scalars for environment
        a_d_item = int(a_d.squeeze(0).cpu().item())
        logp_d_val = float(logp_d.squeeze(0).cpu().item())
        a_c_arr = a_c_clamped.squeeze(0).cpu().numpy().flatten()
        logp_c_val = float(logp_c_sum.squeeze(0).cpu().item())

        return a_d_item, logp_d_val, a_c_arr, logp_c_val

    def evaluate(self, state):
        """
        Deterministic evaluation: returns discrete argmax and deterministic continuous action (mode/fallback)
        Accepts 1D state or batch; returns numpy outputs.
        """
        state_t = torch.tensor(state, dtype=torch.float, device=self.device)
        if state_t.ndim == 1:
            state_t = state_t.unsqueeze(0)

        with torch.no_grad():
            pi, _ = self.actor.forward(state_t)
            a_d = torch.argmax(pi, dim=-1).cpu().numpy()  # shape (B,)
            a_c = self.actor.dist_mode(state_t).cpu().numpy()  # shape (B, c_actions)

        # if single state, return scalars/1d arrays
        if a_d.shape[0] == 1:
            return int(a_d[0]), a_c[0].flatten()
        return a_d, a_c

    def put_data(self, transition):
        """
        transition = (s, a_d, a_c, r, s_prime, logp_d, logp_c_sum, done, dw)
        where:
            - s, s_prime: numpy arrays shape (state_dim,)
            - a_d: int
            - a_c: numpy array shape (c_actions,)
            - r: float
            - logp_d: float
            - logp_c_sum: float
            - done: bool/int
            - dw: bool/int (dead or win mask for TD usage)
        """
        self.data.append(transition)

    def make_batch(self):
        l = len(self.data)
        assert l > 0, "No data to make batch"

        s_lst = np.zeros((l, self.s_dim), dtype=np.float32)
        acts_d_lst = np.zeros((l, 1), dtype=np.int64)
        acts_c_lst = np.zeros((l, self.acts_dims), dtype=np.float32)
        r_lst = np.zeros((l, 1), dtype=np.float32)
        s_prime_lst = np.zeros((l, self.s_dim), dtype=np.float32)
        logprob_d_lst = np.zeros((l, 1), dtype=np.float32)
        logprob_c_lst = np.zeros((l, 1), dtype=np.float32)  # store summed logprob as scalar
        done_lst = np.zeros((l, 1), dtype=np.float32)
        dw_lst = np.zeros((l, 1), dtype=np.float32)

        for i, transition in enumerate(self.data):
            s, a_d, a_c, r, s_prime, logp_d, logp_c_sum, done, dw = transition
            s_lst[i] = s
            acts_d_lst[i] = a_d
            acts_c_lst[i] = a_c
            r_lst[i] = r
            s_prime_lst[i] = s_prime
            logprob_d_lst[i] = logp_d
            logprob_c_lst[i] = logp_c_sum
            done_lst[i] = done
            dw_lst[i] = dw

        if not self.env_with_Dead:
            dw_lst[:, :] = 0.0

        # clear buffer
        self.data = []

        # convert to tensors on device
        s = torch.tensor(s_lst, dtype=torch.float, device=self.device)
        acts_d = torch.tensor(acts_d_lst, dtype=torch.int64, device=self.device)
        acts_c = torch.tensor(acts_c_lst, dtype=torch.float, device=self.device)
        r = torch.tensor(r_lst, dtype=torch.float, device=self.device)
        s_prime = torch.tensor(s_prime_lst, dtype=torch.float, device=self.device)
        logprob_d = torch.tensor(logprob_d_lst, dtype=torch.float, device=self.device)
        logprob_c = torch.tensor(logprob_c_lst, dtype=torch.float, device=self.device)
        dones = torch.tensor(done_lst, dtype=torch.float, device=self.device)
        dws = torch.tensor(dw_lst, dtype=torch.float, device=self.device)

        return s, acts_d, acts_c, r, s_prime, logprob_d, logprob_c, dones, dws

    def train(self):
        # Convert buffer to tensors
        s, acts_d, acts_c, r, s_prime, logprob_d, logprob_c, dones, dws = self.make_batch()
        self.entropy_coef *= self.entropy_coef_decay  # decay entropy coefficient

        # Compute values for TD target and advantages
        with torch.no_grad():
            vs = self.critic(s)
            vs_ = self.critic(s_prime)

            deltas = (r + self.gamma * vs_ * (1.0 - dws) - vs).cpu().numpy().flatten()
            adv_list = [0.0]
            dones_np = dones.cpu().numpy().flatten()
            for dlt, done in zip(deltas[::-1], dones_np[::-1]):
                advantage = dlt + self.gamma * self.gae_lambda * adv_list[-1] * (1.0 - float(done))
                adv_list.append(advantage)
            adv_list.reverse()
            adv = torch.tensor(adv_list[:-1], dtype=torch.float, device=self.device).unsqueeze(1)
            td_target = adv + vs

            if self.adv_normalization:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        N = s.shape[0]
        optim_iter_num = int(np.ceil(N / self.optim_batch_size))

        last_a_loss_d, last_a_loss_c, last_c_loss = None, None, None
        last_entropy_d, last_entropy_c = None, None

        for _ in range(self.n_epochs):
            perm = np.arange(N)
            np.random.shuffle(perm)
            perm_t = torch.LongTensor(perm).to(self.device)

            s_sh = s[perm_t]
            acts_d_sh = acts_d[perm_t]
            acts_c_sh = acts_c[perm_t]
            td_target_sh = td_target[perm_t]
            adv_sh = adv[perm_t]
            logprob_d_sh = logprob_d[perm_t]
            logprob_c_sh = logprob_c[perm_t]

            for i in range(optim_iter_num):
                start = i * self.optim_batch_size
                end = min((i + 1) * self.optim_batch_size, N)
                if start >= end:
                    continue
                idx = slice(start, end)

                # ----------------- Discrete actor -----------------
                prob_d, _ = self.actor(s_sh[idx])
                prob_d = torch.clamp(prob_d, 1e-10, 1.0)
                dist_d = torch.distributions.Categorical(probs=prob_d)
                new_logp_d = dist_d.log_prob(acts_d_sh[idx].squeeze(1)).unsqueeze(1)
                entropy_d = dist_d.entropy().mean()
                ratio_d = torch.exp(new_logp_d - logprob_d_sh[idx])
                surr1 = ratio_d * adv_sh[idx]
                surr2 = torch.clamp(ratio_d, 1.0 - self.policy_clip, 1.0 + self.policy_clip) * adv_sh[idx]
                loss_actor_d = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy_d

                self.optimizer_d.zero_grad()
                loss_actor_d.backward()
                torch.nn.utils.clip_grad_norm_(list(self.optimizer_d.param_groups[0]['params']), 0.5)
                self.optimizer_d.step()

                # ----------------- Continuous actor -----------------
                _, (alpha, beta) = self.actor(s_sh[idx])
                dist_c = torch.distributions.Beta(alpha, beta)
                acts_c_safe = torch.clamp(acts_c_sh[idx], 1e-6, 1-1e-6)
                new_logp_c = dist_c.log_prob(acts_c_safe).sum(dim=-1, keepdim=True)
                entropy_c = dist_c.entropy().mean()
                ratio_c = torch.exp(new_logp_c - logprob_c_sh[idx])
                surr1_c = ratio_c * adv_sh[idx]
                surr2_c = torch.clamp(ratio_c, 1.0 - self.policy_clip, 1.0 + self.policy_clip) * adv_sh[idx]
                loss_actor_c = -torch.min(surr1_c, surr2_c).mean() - self.entropy_coef * entropy_c

                self.optimizer_c.zero_grad()
                loss_actor_c.backward()
                torch.nn.utils.clip_grad_norm_(list(self.optimizer_c.param_groups[0]['params']), 0.5)
                self.optimizer_c.step()

                # ----------------- Critic -----------------
                v_pred = self.critic(s_sh[idx])
                loss_critic = F.mse_loss(td_target_sh[idx], v_pred)
                # L2 regularization
                for name, param in self.critic.named_parameters():
                    if "weight" in name:
                        loss_critic += self.l2_reg * param.pow(2).sum()
                self.critic_optimizer.zero_grad()
                loss_critic.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()

                # Store last losses for logging
                last_a_loss_d = loss_actor_d.item()
                last_a_loss_c = loss_actor_c.item()
                last_c_loss = loss_critic.item()
                last_entropy_d = entropy_d.item()
                last_entropy_c = entropy_c.item()

        return [last_a_loss_d, last_a_loss_c], last_c_loss, [last_entropy_d, last_entropy_c]


    # convenience save/load
    def save(self, episode):
        torch.save(self.critic.state_dict(), f"./model/ppo_critic{episode}.pth")
        torch.save(self.actor.state_dict(), f"./model/ppo_actor{episode}.pth")

    def best_save(self):
        torch.save(self.critic.state_dict(), f"./best_model/ppo_critic.pth")
        torch.save(self.actor.state_dict(), f="./best_model/ppo_actor.pth")

    def load(self, episode):
        self.critic.load_state_dict(torch.load(f"./model/ppo_critic{episode}.pth", map_location=self.device))
        self.actor.load_state_dict(torch.load(f"./model/ppo_actor{episode}.pth", map_location=self.device))

    def load_best(self):
        self.critic.load_state_dict(torch.load(f"./best_model/ppo_critic.pth", map_location=self.device))
        self.actor.load_state_dict(torch.load(f"./best_model/ppo_actor.pth", map_location=self.device))

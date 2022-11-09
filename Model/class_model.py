import torch
import torch.nn as nn
from Model.model import Qnet, Policy, soft_update, Policy, Vnet
import numpy as np
from collections import deque
from copy import deepcopy
import torch.nn.functional as F

class Buffer:
    def __init__(self,o_dim,a_dim,buffer_size = 1000000):
        self.size = buffer_size
        self.num_experience = 0
        self.o_mem = np.empty((self.size, o_dim), dtype=np.float32)
        self.a_mem = np.empty((self.size, a_dim), dtype=np.float32)
        self.no_mem = np.empty((self.size, o_dim), dtype=np.float32)
        self.r_mem = np.empty((self.size, 1), dtype=np.float32)
        self.done_mem = np.empty((self.size, 1), dtype=np.float32)
    def store_sample(self,o,a,r,no,done):
        idx = self.num_experience%self.size
        self.o_mem[idx] = o
        self.a_mem[idx] = a
        self.r_mem[idx] = r
        self.no_mem[idx] = no
        self.done_mem[idx] = done
        self.num_experience += 1
    def random_batch(self, batch_size = 256):
        N = min(self.num_experience, self.size)
        idx = np.random.choice(N,batch_size)
        o_batch = self.o_mem[idx]
        a_batch = self.a_mem[idx]
        r_batch = self.r_mem[idx]
        no_batch = self.no_mem[idx]
        done_batch = self.done_mem[idx]
        return o_batch, a_batch, r_batch, no_batch, done_batch
    def all_batch(self):
        N = min(self.num_experience,self.size)
        return self.o_mem[:N], self.a_mem[:N], self.r_mem[:N], self.no_mem[:N], self.done_mem[:N]
    def store_demo(self,demo):
        demo_len= len(demo)-1
        self.o_mem[:demo_len]  = demo[:-1]
        self.no_mem[:demo_len] = demo[1:]
        self.num_experience += demo_len



class BC_agent:
    def __init__(self,o_dim,a_dim,args):
        self.o_dim, self.a_dim = o_dim, a_dim
        self.args = args
        #SAC Hyperparameters
        self.gamma = args.gamma
        self.hidden_size = args.hidden_size
        self.batch_size = args.batch_size
        self.tau = args.tau
        self.lr = args.lr
        self.q_update_count = 0
        self.n_actions = 200
        self.n_support = 32
        self.beta =1

        self.pi = Policy(o_dim,a_dim,self.hidden_size).to(args.device_train)
        self.pi_opt = torch.optim.Adam(self.pi.parameters(), lr=self.lr)

        #Define networks
        self.q1 = Qnet(self.o_dim, self.a_dim, self.hidden_size).to(args.device_train)
        self.q2 = Qnet(self.o_dim, self.a_dim, self.hidden_size).to(args.device_train)
        self.v = Vnet(self.o_dim,  self.hidden_size).to(args.device_train)
        self.target_v = deepcopy(self.v)
        self.update_count = 0

        #Define optimizer
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=self.lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=self.lr)
        self.v_opt = torch.optim.Adam(self.v.parameters(), lr=self.lr)


    def init_pi(self,task_name,iter,epoch,path):
        path = path + '/' + task_name + '-' + str(iter) + '-' + str(epoch) + '.pt'
        self.pi.load_state_dict(torch.load(path)['policy'])
        self.target_pi = deepcopy(self.pi)
    def init_q(self,task_name,iter,epoch,path):
        path = path + '/' + task_name + '-' + str(iter) + '-' + str(epoch) + '.pt'
        self.q1.load_state_dict(torch.load(path)['q1'])
        self.q2.load_state_dict(torch.load(path)['q2'])
        self.v.load_state_dict(torch.load(path)['v'])

    def save_checkpoint(self,task_name,iter,epoch,path):
        path = path +'/'+task_name+'-'+str(iter)+'-'+ str(epoch) + '.pt'
        print('Saving model to {}'.format(path))
        torch.save({'policy': self.pi.state_dict(),
                    'q1'    : self.q1.state_dict(),
                    'q2'    : self.q2.state_dict(),
                    'v'     : self.v.state_dict(),
                    }, path)

    def select_action(self, o, eval=False):
        action  = self.pi(torch.FloatTensor(o).to(self.args.device_train))
        return action.cpu().detach().numpy()[0]

    def train_bc(self, batch):
        self.pi.train()
        # if self.buffer.num_experience >= self.training_start:
        # state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.buffer.random_batch(self.args.SAC_batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch

        state_batch = torch.FloatTensor(state_batch).to(self.args.device_train)
        action_batch = torch.FloatTensor(action_batch).to(self.args.device_train)

        self.pi_opt.zero_grad()
        pred_action = self.pi(state_batch)
        action_loss = F.mse_loss(pred_action,action_batch)
        action_loss.backward()
        self.pi_opt.step()

    def train_Q(self, batch,cql=False):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, Return_batch = batch

        state_batch = torch.FloatTensor(state_batch).to(self.args.device_train)
        action_batch = torch.FloatTensor(action_batch).to(self.args.device_train)
        reward_batch = torch.FloatTensor(reward_batch).to(self.args.device_train).unsqueeze(1)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.args.device_train)
        done_batch = torch.FloatTensor(done_batch).to(self.args.device_train).unsqueeze(1)
        Return_batch = torch.FloatTensor(Return_batch).to(self.args.device_train).unsqueeze(1)

        self.v_train(state_batch, action_batch, reward_batch, next_state_batch, done_batch,Return_batch)
        with torch.no_grad():
            soft_update(self.target_v, self.v, self.tau)


    def test_q(self,batch):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, Return_batch = batch
        state_batch = torch.FloatTensor(state_batch).to(self.args.device_train)
        with torch.no_grad():
            v_value = self.v(state_batch)
        return v_value

    def train_QBC(self,batch,cql=False):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, Return_batch = batch

        state_batch = torch.FloatTensor(state_batch).to(self.args.device_train)
        action_batch = torch.FloatTensor(action_batch).to(self.args.device_train)
        Return_batch = torch.FloatTensor(Return_batch).to(self.args.device_train).unsqueeze(1)

        self.pi.train()
        with torch.no_grad():
            v_value = self.v(state_batch)
            idx = torch.where(Return_batch - v_value * 1.1 > 0.0, 1.0, 0.0)
            weight = (torch.exp((Return_batch - v_value) / 0.05) - torch.ones_like(Return_batch)).clamp(0.0, 100.0)
            weight = weight * idx

        self.pi_opt.zero_grad()
        pred_action = self.pi(state_batch)
        action_loss = torch.mean(((pred_action - action_batch)**2)*weight.reshape(-1, 1))
        action_loss.backward()
        self.pi_opt.step()

    def v_train(self,state_batch,action_batch,reward_batch,next_state_batch,done_batch,Return_batch):
        self.v.zero_grad()
        v_loss = F.mse_loss(input=self.v(state_batch), target=Return_batch)
        v_loss.backward()
        self.v_opt.step()


def cal_critic_loss(value, target_value, use_quantile_critic, sum_over_quantiles=True):
    """
    The quantile-regression loss, as described in the QR-DQN.
    Partially taken from https://github.com/bayesgroup/tqc_pytorch.
    :param current_quantiles: current estimate of quantiles,
        must be (batch_size, n_quantiles)
    :param target_quantiles: target of quantiles,
        must be either (batch_size, n_target_quantiles) or (batch_size, 1, n_target_quantiles)
    :param cum_prob: cumulative probabilities to calculate quantiles (also called midpoints in QR-DQN paper),
        must be either (batch_size, n_quantiles) or (batch_size, 1, n_quantiles).
    :param sum_over_quantiles: if summing over the quantile dimension or not
    :return: the loss
    """

    if use_quantile_critic:  # use_quantile_critic == True 이면, 분포 강화학습의 loss 계산할 것임.
        n_quantiles = value.shape[-1]
        current_quantiles = value
        target_quantiles = target_value

        cum_prob = (torch.arange(n_quantiles, device=current_quantiles.device, dtype=torch.float) + 0.5) / n_quantiles
        # For QR-DQN, current_quantiles have a shape (batch_size, n_quantiles), and make cum_prob
        # broadcastable to (batch_size, n_quantiles, n_target_quantiles)
        cum_prob = cum_prob.view(1, -1, 1)

        # QR-DQN
        # target_quantiles: (batch_size, n_target_quantiles) -> (batch_size, 1, n_target_quantiles)
        # current_quantiles: (batch_size, n_quantiles) -> (batch_size, n_quantiles, 1)
        # pairwise_delta: (batch_size, n_target_quantiles, n_quantiles)

        pairwise_delta = target_quantiles.unsqueeze(-2) - current_quantiles.unsqueeze(-1)
        abs_pairwise_delta = torch.abs(pairwise_delta)
        huber_loss = torch.where(abs_pairwise_delta > 1, abs_pairwise_delta - 0.5, pairwise_delta ** 2 * 0.5)
        loss = torch.abs(cum_prob - (pairwise_delta.detach() < 0).float()) * huber_loss

        if sum_over_quantiles:
            critic_loss = loss.sum(dim=-2).mean()
        else:
            critic_loss = loss.mean()

    # 일반 강화학습의 loss (그냥 L2 loss 사용)
    else:
        critic_loss = ((value - target_value)**2).mean()

    return critic_loss
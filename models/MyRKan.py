import torch
import torch.nn as nn
import numpy as np
from .KANLayer import *
from .Symbolic_KANLayer import *
from .LBFGS import *
import os
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from torch.autograd import Variable
import copy
from .RKAN  import RKAN
from .HidenKAN import HidenKan
from .KAN import KAN

from sklearn.cluster import KMeans



adjoint = False
if adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint
class MyRKan(nn.Module):
    def __init__(self,widthRKAN=None,widthsolute=None,resultKan=None):
        super(MyRKan, self).__init__()

        self.num = 1
        self.RKan = RKAN(widthRKAN)

        self.ResultKan = KAN(resultKan)
        self.LKan = HidenKan(widthsolute)



        self.attention = nn.Parameter(torch.randn(12,self.num))

        self.kmeans = KMeans(n_clusters=self.num, random_state=0)
        # self.judge = nn.ModuleList(
        #     nn.ReLU()
        # )


    def forward(self,x):
        # data_flag = x[-1, :, -1]
        # unique_flags = torch.unique(data_flag)

        # x = x[:,:,:-1]
        history_input = x[:-1,:,:]
        now_feature = x[-1,:,:].unsqueeze(0)
        b, t, n = x.size(0), x.size(1), x.size(2)
        batch = x.shape[1]
        time = x.shape[0]
        history_hidden_state = torch.stack(self.RKan(x)) #  self.RKan(history_input)


        # calculate the influence
        time_gap_all = now_feature[:,:,2] - x[:, :, 2]

        # Singel
        hident_batch = 0
        for t in range(time_gap_all.size(0)):
            time_batch = torch.stack([torch.linspace(0, gap.item(), 5) for gap in time_gap_all[t]])
            # history_hidden_state = self.KAN1(x[t,:,:])
            # hident_batch += self.integrate(self.LKan, history_hidden_state, time_batch)[-1]
            hident_batch += self.integrate(self.LKan, history_hidden_state[t], time_batch)[-1]

        ans = self.ResultKan(hident_batch).squeeze()
        return hident_batch;

    def result(self,x):
        return self.ResultKan(x)

    def integrate(self,func,history_hidden_state, t):
        time_grid = t

        solution = torch.empty(t.size(1), *history_hidden_state.shape, dtype=history_hidden_state.dtype, device='cpu')
        solution[0] = history_hidden_state

        j = 1
        y0 = history_hidden_state
        for i in range(time_grid.size(1)-1):
            t0 = time_grid[:,i]
            t1 = time_grid[:,i+1]
        # for t0, t1 in zip(time_grid[:,:-1], time_grid[:,1:]):
            dt = t1 - t0
            f0 = func(t0,y0)
            dy = f0*f0*dt.reshape(-1,1);
            y1 = y0 + dy

            while j < t.size(1) and sum(t1 >= time_grid[:,j])>=t1.size(0) :
                solution[j] = self._linear_interp(t0, t1, y0, y1, time_grid[:,j])
                j += 1
            y0 = y1

        return solution

    def _linear_interp(self, t0, t1, y0, y1, t):
        if torch.sum(t > t0) < t.size(0):
            return y0
        if torch.sum(t == t1) < t.size(0):
            return y1
        slope = (t - t0) / (t1 - t0+0.0001)
        return y0 + slope.unsqueeze(-1) * (y1 - y0)


    def get_index(self,inputs,target):
        x  = inputs.transpose(0, 1)
        # history_hidden_state = self.RKan(x)
        # clusters = kmeans.fit_predict(history_hidden_state)
        # cat_x = torch.einsum("ij,jk->ik", torch.cat(history_hidden_state, dim=-1), self.attention)
        # att = nn.functional.softmax(cat_x, dim=-1)
        # indices = torch.max(att, dim=-1).indices

        indices = self.kmeans.predict(inputs.reshape(-1,inputs.shape[1]*inputs.shape[2]).detach().numpy())
        grouped_inputs = []
        grouped_targets = []
        for i in range(self.num):
            mask = indices == i
            grouped_inputs.append(inputs[mask])
            grouped_targets.append(target[mask])
        return grouped_inputs,grouped_targets;

    def fit_myKMean(self,data):
        data = data.reshape(-1, data.shape[1] * data.shape[2])
        self.kmeans.fit(data)




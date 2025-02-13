import copy
import json
import os
import pickle
import sys
import time
import models
from typing import Dict, Optional, List
import torch.nn.functional as F
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from models import glo
import random
from .data import get_dataloaders, PredictorDataset
from .evaluate import evaluate
from collections import defaultdict
def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True,
                                   only_inputs=True,allow_unused=True )[0]
    else:
        return gradients(gradients(u, x), x, order=order - 1)
def create_pad_mask(t, pad):
    mask = (t == pad).unsqueeze(-2)
    return mask
class OursTrainer(object):
    def __init__(self,
                 model: nn.Module,
                 loss: nn.Module,
                 scaler,
                 device: torch.device,
                 optimizer,
                 reg_weight_decay: float,
                 reg_norm: int,
                 max_grad_norm: Optional[float] or Optional[str]
    ):
        self.model = model.to(device)
        self.loss = loss.to(device)
        self.optimizer = optimizer
        self.scaler = scaler
        self.clip = max_grad_norm
        self.device = device
        self.norm = reg_norm
        self.weight_decay = reg_weight_decay


        self.step_size = 0.01
        self.normalize_loss = False
        self.is_robust = False
        self.mix = False
        self.btl = False
        self.gamma = 0.1
        self.n_groups=3
        self.adj = torch.zeros(self.n_groups).float()
        self.adv_probs = (torch.ones(self.n_groups) / self.n_groups).to(self.device)
        self.exp_avg_loss = torch.zeros(self.n_groups).to(self.device)
        self.exp_avg_initialized = torch.zeros(self.n_groups).byte().to(self.device)
        self.reset_stats()

        self.adv_u = (torch.ones(self.n_groups) / self.n_groups).to(self.device)
        self.adv_v = (torch.ones(self.n_groups) / self.n_groups).to(self.device)
        self.step_size1 = 0.1
        self.step_size2 = 0.01
        self.lamb = 1
        # self.MyRKan = glo.get_value('MyRKan')

    def reset_stats(self):
        self.processed_data_counts = torch.zeros(self.n_groups).to(self.device)
        self.update_data_counts = torch.zeros(self.n_groups).to(self.device)
        self.update_batch_counts = torch.zeros(self.n_groups).to(self.device)
        self.avg_group_loss = torch.zeros(self.n_groups).to(self.device)
        self.avg_group_acc = torch.zeros(self.n_groups).to(self.device)
        self.avg_per_sample_loss = 0.
        self.avg_actual_loss = 0.
        self.avg_acc = 0.
        self.batch_count = 0.

    def update_exp_avg_loss(self, group_loss, group_count):
        prev_weights = (1 - self.gamma*(group_count>0).float()).to(self.device) * (self.exp_avg_initialized>0).float().to(self.device)
        curr_weights = 1 - prev_weights
        self.exp_avg_loss = self.exp_avg_loss * prev_weights + group_loss*curr_weights
        self.exp_avg_initialized = (self.exp_avg_initialized>0) + (group_count>0)


    def cosine_similarity_loss(self, Z , Z_pos):
        loss = 1 - F.cosine_similarity(Z, Z_pos).mean()
        return loss

    def triplet_loss(self,anchor, positive, negative=None ,margin=1.0):
        """
        anchor: Tensor of shape (feature_dim,)
        positive: Tensor of shape (feature_dim,)
        negatives: Tensor of shape (num_negatives, feature_dim)
        temperature: Scaling factor for similarity
        """
        # 计算 Anchor 和 Positive 之间的欧氏距离
        pos_distance = F.pairwise_distance(anchor, positive, p=2)  # 欧氏距离
        # 计算 Anchor 和 Negative 之间的欧氏距离
        # neg_distance = F.pairwise_distance(anchor, negative, p=2)  # 欧氏距离

        # 计算 Triplet Loss
        # loss = F.relu(pos_distance - neg_distance + margin)  # 只保留大于 0 的部分
        loss = F.relu( pos_distance )
        return loss.mean()  # 返回平均损失

    def compute_robust_loss(self, group_loss, group_count):
        adjusted_loss = group_loss
        if torch.all(self.adj>0):
            adjusted_loss += self.adj/torch.sqrt(self.group_counts)
        if self.normalize_loss:
            adjusted_loss = adjusted_loss/(adjusted_loss.sum())
        # GDRO version 2017 NIPS
        self.adv_probs = self.adv_probs * torch.exp(self.step_size*adjusted_loss.data)
        self.adv_probs = self.adv_probs/(self.adv_probs.sum())


        # KL version 2024 NIPS
        self.adv_u = (1-self.step_size1)+self.step_size1 * adjusted_loss.data
        self.adv_v = (1-self.step_size2) * self.adv_v + self.step_size2 * torch.exp(self.adv_u/self.lamb)
        self.adv_probs = torch.exp(self.adv_u) / self.adv_v
        self.adv_probs = self.adv_probs / (self.adv_probs.sum())


        robust_loss = group_loss @ self.adv_probs
        return robust_loss, self.adv_probs



    def train(self, inputs, targets,y_Assistance=None):
        self.optimizer.zero_grad()
        targets = targets.to(self.device)
        inputs = inputs.to(self.device).requires_grad_()

        ## 修改处
        new_target = list()
        predicts = list()
        loss = 0;
        data_flag = inputs[:, -1, -1]
        # if use moe
        inputs = inputs[:,:,:-1]
        if glo.get_value('model_name') == 'MyRKan':

            positive_samples = []
            # y_Assistance = (y_Assistance > 0.7).float()


            output = self._run(inputs[:,:,2:])
            output = self.model.result(output)


            actual_loss = self.get_loss(output, targets, data_flag)

            Lkan_reg = self.reg(self.model.LKan)
            Resultkan_reg = self.reg(self.model.ResultKan)
            Rkan_reg = self.reg(self.model.RKan)
            reg = (Rkan_reg + Lkan_reg + Resultkan_reg)*0.05

            (actual_loss).backward()
            # (actual_loss+reg).backward()


        elif glo.get_value('model_name') == 'KAN':
            #kan
            output = self._run(inputs[:,:,2:])

            actual_loss = self.get_loss(output, targets, data_flag)
            # loss += self.loss(output.to(self.device).squeeze(), targets.to(self.device).squeeze())
            KAN_Reg = self.reg(self.model) * 0
            (actual_loss+KAN_Reg).backward()
            # actual_loss.backward()

        elif glo.get_value('model_name') == 'pre':

            Z = self.MyRKan(inputs[:,:,2:].transpose(0, 1))

            self.optimizer.zero_grad()
            output = self._run(Z)
            loss = self.loss(output.to(self.device).squeeze(), targets.to(self.device).squeeze())

            (loss).backward()

        else:
            output = self._run(inputs)

            actual_loss = self.get_loss(output,targets,data_flag)

            params = list(self.model.parameters())
            # l2_norm = torch.sqrt(sum(torch.sum(param ** 2) for param in params if param.requires_grad)) * 0
            # print("模型的二范数:", l2_norm.item())
            reg = get_regularization(self.model, weight_decay=self.weight_decay, p=2)*0 # 加入正则项
            (actual_loss+reg ).backward()
        # if self.clip is not None:

        #     nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        return output

    def compute_group_avg(self, losses, group_idx):
        # compute observed counts and mean loss for each group
        group_map = (group_idx == (torch.arange(3) / 3).unsqueeze(1).to(self.device)).float()
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count==0).float() # avoid nans
        group_loss = (group_map @ losses.view(-1))/group_denom
        return group_loss, group_count


    def get_loss(self,output,targets,data_flag):

        # class_counts = torch.bincount(targets.squeeze().long(), minlength=2)
        #
        # class_weights = torch.tensor([2,1]).to(self.device)  # 归一化到 [0, 1]

        # class_weights = class_counts.sum() / (class_counts.float() + 1e-6)
        # class_weights = class_weights / class_weights.sum()  # 归一化到 [0, 1]

        # class_weights = 1.0 / (torch.log(class_counts.float() + 1e-6) + 1e-6)
        # class_weights = class_weights / class_weights.sum()  # 归一化到 [0, 1]


        # sample_weights = class_weights[targets.to(self.device).squeeze().long()]
        loss = self.loss(output.to(self.device).squeeze(), targets.to(self.device).squeeze())
        # loss = self.loss(output.to(self.device).squeeze(), targets.to(self.device).squeeze())
        # loss = loss * sample_weights

        group_loss, group_count = self.compute_group_avg(loss, data_flag)
        group_acc, group_count = self.compute_group_avg(
            (torch.argmax(output.unsqueeze(-1), 1).squeeze() == targets.squeeze()).float(), data_flag)

        # update historical losses
        self.update_exp_avg_loss(group_loss, group_count)
        if self.is_robust and not self.mix:
            actual_loss, weights = self.compute_robust_loss(group_loss, group_count)
        elif self.is_robust and self.mix:
            actual_loss, weights = self.compute_robust_loss(group_loss, group_count)
            actual_loss = group_loss.mean() + 0.5*actual_loss
        else:
            actual_loss = group_loss.mean()
            weights = None
        # if self.is_robust and not self.btl:
        #     actual_loss, weights = self.compute_robust_loss(group_loss, group_count)
        # elif self.is_robust and self.btl:
        #     actual_loss, weights = self.compute_robust_loss_btl(group_loss, group_count)
        # else:
        #     actual_loss = group_loss.mean()
        #     weights = None

        self.update_stats(actual_loss, group_loss, group_acc, group_count, weights)
        return actual_loss

    def update_stats(self, actual_loss, group_loss, group_acc, group_count, weights=None):
        # avg group loss
        denom = self.processed_data_counts + group_count
        denom += (denom==0).float()
        prev_weight = self.processed_data_counts/denom
        curr_weight = group_count/denom
        self.avg_group_loss = prev_weight*self.avg_group_loss + curr_weight*group_loss

        # avg group acc
        self.avg_group_acc = prev_weight*self.avg_group_acc + curr_weight*group_acc

        # batch-wise average actual loss
        denom = self.batch_count + 1
        self.avg_actual_loss = (self.batch_count/denom)*self.avg_actual_loss + (1/denom)*actual_loss

        # counts
        self.processed_data_counts += group_count
        if self.is_robust:
            self.update_data_counts += group_count*((weights>0).float())
            self.update_batch_counts += ((group_count*weights)>0).float()
        else:
            self.update_data_counts += group_count
            self.update_batch_counts += (group_count>0).float()
        self.batch_count+=1

        # avg per-sample quantities
        group_frac = self.processed_data_counts/(self.processed_data_counts.sum())
        self.avg_per_sample_loss = group_frac @ self.avg_group_loss
        self.avg_acc = group_frac @ self.avg_group_acc

    def reg(self,models):

        def nonlinear(x, th=1e-16, factor=1):
            return (x < th) * x * factor + (x > th) * (x + (factor - 1) * th)

        reg_ = 0.
        for i in range(len(models.acts_scale)):
            vec = models.acts_scale[i].reshape(-1, )

            p = vec / torch.sum(vec)
            l1 = torch.sum(nonlinear(vec))
            entropy = - torch.sum(p * torch.log2(p + 1e-4))
            reg_ += 1.0 * l1 #+ 1.0 * entropy  # both l1 and entropy

        # regularize coefficient to encourage spline to be zero
        # for i in range(len(self.act_fun)):
        #     coeff_l1 = torch.sum(torch.mean(torch.abs(self.act_fun[i].coef), dim=1))
        #     coeff_diff_l1 = torch.sum(torch.mean(torch.abs(torch.diff(self.act_fun[i].coef)), dim=1))
        #     reg_ += lamb_coef * coeff_l1 + lamb_coefdiff * coeff_diff_l1

        return reg_


    def predict(self, inputs,targets):
        targets = targets.to(self.device)
        inputs = inputs.to(self.device)
        if glo.get_value('model_name') == 'MyRKan' :
            self.optimizer.zero_grad()
            predicts = self._run(inputs[:,:,2:-1])
            # predicts = self._run(inputs[:, :, 2:])
            predicts = self.model.result(predicts)
            return predicts
        elif  glo.get_value('model_name') == 'KAN':
            self.optimizer.zero_grad()
            predicts = self._run(inputs[:, :, 2:-1])
            return predicts
        else:
            if glo.get_value('model_name') == 'pre':
                MyRKan = glo.get_value('MyRKan')
                Z = MyRKan(inputs[:, :, 2:-1].transpose(0, 1))

            else :
                Z = inputs[:,:,:-1]
            # inputs = inputs.to(self.device).requires_grad_()
            return self._run(Z)

    def _run(self, inputs):
        #inputs[..., 0] = self.scaler.transform(inputs[..., 0], 0.0)
        if glo.get_value('model_name') == 'MyRKan':
            inputs = inputs.transpose(0, 1).to(self.device)
            outputs = self.model(inputs)
            inputs = inputs.transpose(0, 1)
        elif glo.get_value('model_name')=='KAN':
            inputs = inputs.transpose(0, 1).to(self.device)
            outputs =self.model(inputs[-1])
            inputs = inputs.transpose(0, 1)
        elif glo.get_value('model_name')=='DKT':
            inputs = inputs.transpose(0, 1).to(self.device)
            outputs = self.model(inputs[:-1, :, :], inputs[-1])
            inputs = inputs.transpose(0, 1)
        elif glo.get_value('model_name')=='FIFAKT':
            inputs = inputs.transpose(0, 1).to(self.device)
            outputs = self.model(inputs[:-1, :, :], inputs[-1])
            inputs = inputs.transpose(0, 1)
        elif glo.get_value('model_name')=='SimpleKT':
            outputs = self.model(inputs[:, :, 0], inputs[:, :, 0], inputs[:, :, -1])  # SimpleKT
        elif glo.get_value('model_name')=='MIKT':
            outputs = self.model(inputs[:, :-1, 0], inputs[:, :-1, -1], inputs[:, 1:, 0], inputs[:, 1:, -1])  # MIKT
        elif glo.get_value('model_name')=='QIKTNet':
            outputs = self.model(inputs[:, :, 0], inputs[:, :, 0], inputs[:, :, -1], inputs)  # QIKT
        elif glo.get_value('model_name') == 'pre':
            # inputs = inputs.transpose(0, 1).to(self.device)
            outputs = self.model(inputs)
        elif glo.get_value('model_name') == 'KANwithMlp':
            inputs = inputs.transpose(0, 1).to(self.device)
            outputs = self.model(inputs[-1,:,2:])
            inputs = inputs.transpose(0, 1)
        elif glo.get_value('model_name') == 'LogisticRegression':
            outputs = self.model(inputs[:,:-1,:],inputs[:,-1,:])
            # inputs = inputs.transpose(0, 1)
        # outputs = self.model(inputs[:,:-1,0],inputs[:,:-1,-1],inputs[:,1:,0],inputs[:,1:,-1]) # MIKT
        # outputs = self.model(inputs[:, :, 4], inputs[:, :, 4], inputs[:, :, 1])
        # outputs = self.model(inputs[:, :, 0], inputs[:, :, 0], inputs[:, :, -1], inputs) #QIKT
        # outputs = self.model(inputs[:, :, 0], inputs[:, :, 0], inputs[:, :, -1]) #SimpleKT
        # outputs = self.model(inputs[:-1,:,:],inputs[-1]); #DKT-FKAN
        # outputs = self.model(inputs,i)
        # outputs = self.model(inputs[-1])
        # inputs = inputs.transpose(0, 1)
        return outputs  #self.scaler.inverse_transform(outputs, 0.0)

    def load_state_dict(self, model_state_dict, optimizer_state_dict):
        self.model.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(optimizer_state_dict)
        self.model = self.model.to(self.device)
        set_device_recursive(self.optimizer.state, self.device)
        '''
        print('Resume:load_state_dict__________________________')
        #print model's state_dict
        print('Model.state_dict:')
        for param_tensor in self.model.state_dict():
            #打印 key value字典
            print(param_tensor,'\t',self.model.state_dict()[param_tensor].size())
 
        #print optimizer's state_dict
        print('Optimizer,s state_dict:')
        for var_name in self.optimizer.state_dict():
            print(var_name,'\t',self.optimizer.state_dict()[var_name])
        '''

    def allplot(self,inputs):
        # models.prune();
        # self.featureTKanKernel = KAN([3,1])
        # self.LFeatureKanKernel = KAN([1,3])
        # self.model.RKan.plot(beta=100)
        # self.model.LKan.plot(beta=100)
        # self.model.ResultKan.plot(beta=100)
        models = "kans"
        if glo.get_value('model_name') == "MyRKan":
            inputs= inputs[:,:,2:-1]
            state = self._run(inputs)
            self.model.result(state)
            # 提纯后
            # self.model.RKan = self.model.RKan.prune(threshold=0.1)
            # self.model.LKan = self.model.LKan.prune(threshold=0.1)
            # self.model.ResultKan = self.model.ResultKan.prune(threshold=0.1)

            # inputs = inputs.transpose(0, 1).to(self.device)
            # outputs = self.model(inputs[:,:-1,4],inputs[:,:-1,1],inputs[:,1:,4],inputs[:,1:,1])
            # outputs = self.model(inputs[:, :, 4], inputs[:, :, 4], inputs[:, :, 1])

            #修改处
            # input_group, targets_group = self.model.get_index(inputs, torch.zeros_like(inputs))
            # Lkan_reg = 0;
            # for i in range(len(input_group)):
            #     if (len(input_group[i]) == 0): continue;
            #     predicts = self._run(input_group[i], i).reshape(-1, 1)



            # self.model.RKan.plot(beta=100)
            # self.model.LKan.plot(beta=100)
            # self.model.ResultKan.plot(beta=100)
            lib = ['x', 'x^2', 'exp', 'log', 'sqrt']

            # state = self._run(inputs)
            # self.model.result(state)
            # self._run(input_group[0], 0).reshape(-1, 1)
            # self._run(input_group[1], 1).reshape(-1, 1)
            # self._run(input_group[2], 2).reshape(-1, 1)
            self.model.RKan.auto_symbolic(lib=lib)
            self.model.LKan.auto_symbolic(lib=lib)
            self.model.ResultKan.auto_symbolic(lib=lib)
            print("RKAN", self.model.RKan.symbolic_formula()[0])
            print("RKAN1", self.model.RKan.symbolic_formula()[0][1])
            print("RKAN2", self.model.RKan.symbolic_formula()[0][2])
            print("LKan", self.model.LKan.symbolic_formula()[0])
            print("LKan1_2", self.model.LKan.symbolic_formula()[0][1])
            print("LKan1_3", self.model.LKan.symbolic_formula()[0][2])
            print("ResultKan", self.model.ResultKan.symbolic_formula()[0])

            # # self._run(input_group[1], 1).reshape(-1, 1)
            # self.model.LKan[1].auto_symbolic(lib=lib)
            # self.model.ResultKan[1].auto_symbolic(lib=lib)
            # print("LKan1_1", self.model.LKan[1].symbolic_formula()[0][0])
            # print("LKan1_2", self.model.LKan[1].symbolic_formula()[0][1])
            # print("LKan1_3", self.model.LKan[1].symbolic_formula()[0][2])
            # print("ResultKan1", self.model.ResultKan[1].symbolic_formula()[0][0])
            #
            #
            #
            #
            #
            # # self._run(input_group[2], 2).reshape(-1, 1)
            # self.model.LKan[2].auto_symbolic(lib=lib)
            # self.model.ResultKan[2].auto_symbolic(lib=lib)
            #
            # print("LKan2_1", self.model.LKan[2].symbolic_formula()[0][0])
            # print("LKan2_2", self.model.LKan[2].symbolic_formula()[0][1])
            # print("LKan2_3", self.model.LKan[2].symbolic_formula()[0][2])
            # print("ResultKan2", self.model.ResultKan[2].symbolic_formula()[0][0])

        elif glo.get_value('model_name') == "KAN":
            # self.model.plot()
            inputs = inputs.transpose(0, 1).to(self.device)
            inputs = inputs[-1]
            self.model(inputs)
            # self.model = self.model.prune()
            # self.model(inputs)
            lib = ['x', 'x^2', 'exp', 'log', 'sqrt']
            self.model.auto_symbolic(lib=lib)
            print("RKAN1_1", self.model.symbolic_formula()[0][0])




def train_model(
        datasets: Dict[str, PredictorDataset],
        batch_size: int,
        folder: str,
        trainer: OursTrainer,
        scheduler,
        epochs: int,
        early_stop_steps: int ,
        min: int,
        max: int
):#Optional[int]
    datasets['train'].inputs = (datasets['train'].inputs - np.array(min))/(np.array(max)-np.array(min)+1)

    datasets['val'].inputs = (datasets['val'].inputs - np.array(min)) / (np.array(max) - np.array(min) + 1)

    datasets['test'].inputs = (datasets['test'].inputs - np.array(min)) / (np.array(max) - np.array(min) + 1)

    # filelog_name = args + ".txt"



    data_loaders = get_dataloaders(datasets, batch_size)
    # data_loaders_test = get_dataloaders(datasets['test'], batch_size)

    train_x_input = torch.tensor(datasets['test'].inputs[:30000,:,:],dtype=torch.float);
    save_path = os.path.join(folder, 'best_model.pkl')
    begin_epoch = 0

    if os.path.exists(save_path):
        save_dict = torch.load(save_path)

        trainer.load_state_dict(save_dict['model_state_dict'], save_dict['optimizer_state_dict'])

        best_val_loss = save_dict['best_val_loss']
        begin_epoch = save_dict['epoch'] + 1
    else:
        save_dict = dict()
        # best_val_loss = float('inf')
        begin_epoch = 0
        best_val_loss = 0

    phases = ['train', 'val', 'test']
    # phases = ['train']


    writer = SummaryWriter(folder)

    since = time.perf_counter()
    print(trainer.model)
    print(f'Trainable parameters: {get_number_of_parameters(trainer.model)}.')
    torch.backends.cudnn.enabled = False



    dataName = {'0':'En2De','2':'En2Es','5':'Duolingo','8':'MeiMemo'}
    # dataName = {'0': 'En2De', '-10': 'En2Es', '-20': 'Duolingo', '-30': 'MeiMemo'}
    try:
        for epoch in range(begin_epoch, begin_epoch + epochs):
            running_metrics = dict()
            text_x_input =  0
            for phase in phases:
                steps, predicts, targets,predict_uion = 0, list(), list(), list()
                testdict = {}
                for x, y in tqdm(data_loaders[phase], f'{phase.capitalize():5} {epoch}'):

                    if phase == 'train':
                        # y_Assistance = Assistance_model(x.transpose(0, 1)[:-1, :, :-1].to(0),
                        #                                 x.transpose(0, 1)[-1, :, :-1].to(0))
                        # output = trainer.train(x, y,y_Assistance)
                        output = trainer.train(x, y)


                        # output = torch.zeros_like(y)
                        # if glo.get_value('model_name')=='pre':
                        #     output = trainer.train(x, y)
                        # else:
                        #     y_Assistance = Assistance_model(x.transpose(0, 1)[:-1, :, :-1].to(0),
                        #                                     x.transpose(0, 1)[-1, :, :-1].to(0))
                        #     output = trainer.train(x,y,y_Assistance)

                    if phase == 'val' :
                        with torch.no_grad():
                            # y_ = trainer.predict(x)



                            output = trainer.predict(x, y)
                            # output = torch.zeros_like(y)
                    if phase == 'test':
                        with torch.no_grad():
                            # y_ = trainer.predict(x)
                            data_flag = x[:,-1,-1]


                            output = trainer.predict(x, y)
                            # output = torch.zeros_like(y)

                            # unique_flags = torch.unique(data_flag)
                            # for flag in unique_flags:
                            #     # 获取当前 flag 的掩码
                            #     mask = (data_flag == flag)
                            #
                            #     # 按掩码筛选预测输出和真实值
                            #     filtered_output = output[mask]  # 对应的预测输出
                            #     filtered_y = y[mask]  # 对应的真实值
                            #
                            #     if str(dataName[f'{round(flag.item()*10)}']) not in testdict:
                            #         testdict[str(dataName[f'{round(flag.item()*10)}'])] = {"outputs": [], "y": []}
                            #
                            #         # 追加到字典中
                            #     testdict[str(dataName[f'{round(flag.item()*10)}'])]["outputs"].append(filtered_output)
                            #     testdict[str(dataName[f'{round(flag.item()*10)}'])]["y"].append(filtered_y)

                    if glo.get_value('model_name') != 'MyRKan'and phase == 'train':
                        targets.append(y.numpy().copy())
                        predicts.append(output.detach().cpu().numpy())
                    else:
                        targets.append(y.numpy().copy())
                        predicts.append(output.detach().cpu().numpy())

                        # For the issue that the CPU memory increases while training. DO NOT know why, but it works.
                        torch.cuda.empty_cache()
                # elif phase == 'test':
                #     testdict={}
                #     for datasetName in ['Anki','Duolingo','MeiMemo']:
                #         targets1=[]
                #         predicts1=[]
                #         for x,y in tqdm(data_loaders_test[datasetName], f'{phase.capitalize():5} {epoch}'):
                #             with torch.no_grad():
                #                 # y_ = trainer.predict(x)
                #                 output = trainer.predict(x,y)
                #             targets1.append(y.numpy().copy())
                #             predicts1.append(output.detach().cpu().numpy())


                        # evaluate
                        # testdict[f'{datasetName}'] = evaluate(np.concatenate(predicts1), np.concatenate(targets1))
                    # for key, value in testdict.items():
                    #     print(f'{phase} : {key} = {value}')
                    # else:
                    #     with torch.no_grad():
                    #         # y_ = trainer.predict(x)
                    #         output = trainer.predict(x,y)


                # if glo.get_value('model_name')=='DKT' and phase == 'train':
                #     torch.save(copy.deepcopy(trainer.model.state_dict()), 'Assisant_model/DRO_DKT.pth')
                #     print(f"save epoch{epoch}:  Assisant_model/DRO_DKT.pth")




                if phase == 'test':
                    running_metrics[phase] = evaluate(np.concatenate(predicts), np.concatenate(targets))
                    with open("ROKAN.txt", 'a') as log_file:
                        sys.stdout = log_file  # 重定向标准输出到文件（追加模式）
                        # 所有的 print 输出都会被追加到文件
                        print(f"epoch {epoch}")
                        for key, value in running_metrics.items():
                            print(f'{phase} : {key} = {value}')

                        print()
                        sys.stdout = sys.__stdout__

                    # print('worse gourp test')
                    # dic= {}
                    # for key,value in testdict.items():
                    #     dic[f'{key}'] = evaluate(np.array(torch.cat(testdict[key]["outputs"]).cpu()),np.array(torch.cat(testdict[key]["y"]).cpu().squeeze()))
                    # for key, value in dic.items():
                    #     print(f'{key} : {value}')
                else:
                    running_metrics[phase] = evaluate(np.concatenate(predicts), np.concatenate(targets))
                if phase == 'val':
                    if running_metrics['val']['AUC'] > best_val_loss:
                        best_val_loss = running_metrics['val']['AUC']
                        os.makedirs(os.path.split(save_path)[0], exist_ok=True)
                        save_dict.update(
                            model_state_dict=copy.deepcopy(trainer.model.state_dict()),
                            epoch=epoch,
                            best_val_loss=best_val_loss,
                            optimizer_state_dict=copy.deepcopy(trainer.optimizer.state_dict()))
                        # save_model('Assisant_model/', **save_dict)
                        # torch.save(copy.deepcopy(trainer.model.state_dict()), 'Assisant_model/model.pth')
                        # print(f'A better model at epoch {epoch} recorded.')
                    elif early_stop_steps is not None and epoch - save_dict['epoch'] > early_stop_steps:
                        trainer.model.load_state_dict(save_dict['model_state_dict'])
                        trainer.allplot(train_x_input)
                        flag = 0;
                        break;
        # if(flag != 0):

        ## if you want to plot the ROKAN use the allplot
        # trainer.allplot(train_x_input)


            # scheduler.step()
            # loss_dict = {f'model {phase} loss: ': running_metrics[phase] for phase in phases}
            # txt_file = open("log.txt", "w")
            # txt_file.write(f"{epoch}\n")
            # for i in loss_dict.items():
            #     txt_file.write(f"{i}\n")
            #     print(i)
            # loss_dict = {f'{phase} loss': running_metrics[phase].pop('loss') for phase in phases}


            # if scheduler is not None:
            #     if isinstance(scheduler, ReduceLROnPlateau):
            #         scheduler.step(loss_dict['train'])
            #     else:
            #         scheduler.step()

            # writer.add_scalars('Loss', loss_dict, global_step=epoch)
            # for metric in running_metrics['train'].keys():
            #     for phase in phases:
            #         for key, val in running_metrics[phase].items():
            #             writer.add_scalars(f'{metric}/{key}', {f'{phase}': val}, global_step=epoch)
        # if (flag == 1):
        #     trainer.model.load_state_dict(save_dict['model_state_dict'])

        # for epoch in range(1, 5):
        #     running_metrics = dict()
        #     text_x_input = 0
        #     for phase in phases:
        #         steps, predicts, targets, predict_uion = 0, list(), list(), list()
        #         for x, y in tqdm(data_loaders[phase], f'{phase.capitalize():5} {epoch}'):
        #
        #             targets.append(y.numpy().copy())
        #             if phase == 'train':
        #                 y_ = trainer.train(x, y)
        #             else:
        #                 with torch.no_grad():
        #                     y_ = trainer.predict(x)
        #             predicts.append(y_.detach().cpu().numpy())
        #
        #             # For the issue that the CPU memory increases while training. DO NOT know why, but it works.
        #             torch.cuda.empty_cache()
        #         running_metrics[phase] = evaluate(np.concatenate(predicts), np.concatenate(targets))
        #
        #         if phase == 'val':
        #             if running_metrics['val']['AUC'] > best_val_loss:
        #                 best_val_loss = running_metrics['val']['AUC']
        #                 os.makedirs(os.path.split(save_path)[0], exist_ok=True)
        #                 save_dict.update(
        #                     model_state_dict=copy.deepcopy(trainer.model.state_dict()),
        #                     epoch=epoch,
        #                     best_val_loss=best_val_loss,
        #                     optimizer_state_dict=copy.deepcopy(trainer.optimizer.state_dict()))
        #                 # save_model(save_path, **save_dict)
        #                 torch.save(copy.deepcopy(trainer.model.state_dict()), save_path)
        #                 print(f'A better model at epoch {epoch} recorded.')
        #             # elif early_stop_steps is not None and epoch - save_dict['epoch'] > 7:
        #             #     trainer.model.load_state_dict(save_dict['model_state_dict'])
        #             #     # trainer.allplot(train_x_input)
        #             #     raise Exception
        #
        #     loss_dict = {f'model {phase} loss: ': running_metrics[phase] for phase in phases}
        #     txt_file = open("log.txt", "w")
        #     txt_file.write(f"{epoch}\n")
        #     for i in loss_dict.items():
        #         txt_file.write(f"{i}\n")
        #         print(i)
        #     loss_dict = {f'{phase} loss': running_metrics[phase].pop('loss') for phase in phases}
        #
        #     if scheduler is not None:
        #         if isinstance(scheduler, ReduceLROnPlateau):
        #             scheduler.step(loss_dict['train'])
        #         else:
        #             scheduler.step()
        #
        #     # writer.add_scalars('Loss', loss_dict, global_step=epoch)
        #     # for metric in running_metrics['train'].keys():
        #     #     for phase in phases:
        #     #         for key, val in running_metrics[phase].items():
        #     #             writer.add_scalars(f'{metric}/{key}', {f'{phase}': val}, global_step=epoch)
        #     # loss_dict = {f'model {phase} loss: ': running_metrics[phase] for phase in phases}
        #     # txt_file = open("log.txt", "w")
        #     # txt_file.write(f"{epoch}\n")
        #     # for i in loss_dict.items():
        #     #     txt_file.write(f"{i}\n")
        #     #     print(i)
        #     # loss_dict = {f'{phase} loss': running_metrics[phase].pop('loss') for phase in phases}
        #     #
        #     # if scheduler is not None:
        #     #     if isinstance(scheduler, ReduceLROnPlateau):
        #     #         scheduler.step(loss_dict['train'])
        #     #     else:
        #     #         scheduler.step()
        #     #
        #     # writer.add_scalars('Loss', loss_dict, global_step=epoch)
        #     # for metric in running_metrics['train'].keys():
        #     #     for phase in phases:
        #     #         for key, val in running_metrics[phase].items():
        #     #             writer.add_scalars(f'{metric}/{key}', {f'{phase}': val}, global_step=epoch)

    except (ValueError, KeyboardInterrupt) as e:
        print(e)
        # trainer.allplot(train_x_input)
        # # lib
        # trainer.model.RKan.featureTKanKernel.auto_symbolic(lib=lib)
        # trainer.model.RKan.LFeatureKanKernel.auto_symbolic(lib=lib)
        # trainer.model.LKan.auto_symbolic(lib=lib)
        # trainer.model.ResultKan.auto_symbolic(lib=lib)


    time_elapsed = time.perf_counter() - since
    print(f"cost {time_elapsed} seconds")
    print(f'The best adaptor and model of epoch {save_dict["epoch"]} successfully saved at `{save_path}`')


def test_model(
        datasets: Dict[str, PredictorDataset],
        batch_size: int,
        trainer: OursTrainer,
        folder: str):
    dataloaders = get_dataloaders(datasets, batch_size)

    saved_path = os.path.join(folder, 'best_model.pkl')

    saved_dict = torch.load(saved_path)
    trainer.model.load_state_dict(saved_dict['model_state_dict'])

    predictions, running_targets = list(), list()
    with torch.no_grad():
        for inputs, targets in tqdm(dataloaders['test'], 'Test model'):
            running_targets.append(targets.numpy().copy())
            predicts = trainer.predict(inputs)
            predictions.append(predicts.cpu().numpy())

    # 性能
    predictions, running_targets = np.concatenate(predictions), np.concatenate(running_targets)
    scores = evaluate(predictions, running_targets)
    scores.pop('loss')
    print('test results:')
    print(json.dumps(scores, cls=JsonEncoder, indent=4))
    with open(os.path.join(folder, 'test-scores.json'), 'w+') as f:
        json.dump(scores, f, cls=JsonEncoder, indent=4)

    np.savez(os.path.join(folder, 'test-results.npz'), predictions=predictions, targets=running_targets)


def save_model(path: str, **save_dict):
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    torch.save(save_dict, path)


class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JsonEncoder, self).default(obj)


def get_number_of_parameters(model: nn.Module):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def get_scheduler(name, optimizer, **kwargs):
    return getattr(optim.lr_scheduler, name)(optimizer, **kwargs)


def get_optimizer(name: str, parameters, **kwargs):
    return getattr(optim, name)(parameters, **kwargs)


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def set_device_recursive(var, device):
    for key in var:
        if isinstance(var[key], dict):
            var[key] = set_device_recursive(var[key], device)
        else:
            try:
                var[key] = var[key].to(device)
            except AttributeError:
                pass
    return var


def set_requires_grad(models: List[nn.Module], required: bool):
    for model in models:
        for param in model.parameters():
            param.requires_grad_(required)


def get_regularization(model: nn.Module, weight_decay: float, p: float = 2.0):
    weight_list = list(filter(lambda item: 'weight' in item[0], model.named_parameters()))
    return weight_decay * sum(torch.norm(w, p=p) for name, w in weight_list)

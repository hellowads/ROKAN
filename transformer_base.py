import argparse
import json
import os
import shutil
import torch
import yaml
import utils
import models
import random
import numpy as np
from utils import get_datasets,get_datasetssingle
import torch.nn as nn
from models import glo

def train(_config,args,resume: bool = False, test: bool = False):
    print(json.dumps(_config, indent=4))
    device = torch.device(_config['device'])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device.index)
    dataset = _config['data']['dataset']
    optimizer_name = _config['optimizer']['name']
    scheduler_name = _config['scheduler']['name']
    loss = torch.nn.MSELoss(reduction='none')

    model = models.MyRKan(**_config['model']['MyRKan'])


    # device = torch.device('cuda:0')
    # model.to(device)
    optimizer = utils.get_optimizer(optimizer_name, model.parameters(), **_config['optimizer'][optimizer_name])
    scheduler = None
    if scheduler_name is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=1)
    save_folder = os.path.join('saves', dataset,"MyRKan")
    if not resume and not test:
        shutil.rmtree(save_folder, ignore_errors=True)
        os.makedirs(save_folder)
    # with open(os.path.join(save_folder, 'config.yaml'), 'w+') as _f:
    #     yaml.safe_dump(_config, _f)



    # dataset1 = 'multi_data/Anki'
    # dataset2 = 'multi_data/Duolingo'
    # dataset3 = 'multi_data/MeiMemo'
    if args.sigle :
        dataset = 'data/multi_data/'+args.dataset
        datasets = get_datasetssingle(dataset)
    else:
        dataset = args.dataset
        datasets = get_datasets(dataset)
    # datasets = get_datasets2(dataset)
    # if momo mask this code

    min, max = datasets['train'].__minmax__()

    # if model_name == 'KAN' or model_name == 'MyRKan' :
    #     min = min[2:]
    #     max = max[2:]
    #     datasets['train'].inputs = datasets['train'].inputs[: ,: , 2:]
    #     datasets['val'].inputs = datasets['val'].inputs[:, :, 2:]
    #     datasets['test']['Anki'].inputs = datasets['test']['Anki'].inputs[:, :, 2:]
    #     datasets['test']['Duolingo'].inputs = datasets['test']['Duolingo'].inputs[:, :, 2:]
    #     datasets['test']['MeiMemo'].inputs = datasets['test']['MeiMemo'].inputs[:, :, 2:]

    scaler = 0
    trainer = utils.OursTrainer(model, loss, scaler, device, optimizer, **_config['trainer'])
    #
    glo._init()
    glo.set_value('model_name','MyRKan')

    utils.train_model(
        datasets=datasets,
        batch_size=_config['data']['batch-size'],
        folder=save_folder,
        trainer=trainer,
        scheduler=scheduler,
        epochs=_config['epochs'],
        early_stop_steps=_config['early_stop_steps'],
        min=min,max=max,
    )
#---------------------------------------------------#
#   设置种子
#---------------------------------------------------#
def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False





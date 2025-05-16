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
    # loss = torch.nn.BCELoss(reduction='none')
    loss = torch.nn.MSELoss(reduction='none')
    # loss = torch.nn.BCEWithLogitsLoss(reduction='none')

    model_name = 'MyRKan'
    glo._init()

    model = models.MyRKan(**_config['model']['MyRKan'])
    glo.set_value('model_name', model_name)
    optimizer = utils.get_optimizer(optimizer_name, model.parameters(), **_config['optimizer'][optimizer_name])
    scheduler = None
    if scheduler_name is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=1)
    save_folder = os.path.join('saves', dataset,model_name)
    if not resume and not test:
        shutil.rmtree(save_folder, ignore_errors=True)
        os.makedirs(save_folder)
    with open(os.path.join(save_folder, 'config.yaml'), 'w+') as _f:
        yaml.safe_dump(_config, _f)

    if args.dataset == 'anki':
        args.dataset = '/frsr'
    # sigle = True
    if args.sigle :
        dataset = 'data/multi_data' + args.dataset

        datasets = get_datasetssingle(dataset)
    else:
        dataset = args.dataset
        datasets = get_datasets(dataset)
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


    utils.train_model(
        datasets=datasets,
        batch_size=_config['data']['batch-size'],
        folder=save_folder,
        trainer=trainer,
        scheduler=scheduler,
        epochs=_config['epochs'],
        early_stop_steps=_config['early_stop_steps'],
        min=min,max=max,
        Assisant = None,
        args = model_name
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## 2
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--config', type=str, default="config",
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='if to resume a trained model?')
    parser.add_argument('--test', action='store_true', default=False,
                        help='if in the test mode?')
    parser.add_argument('--name', type=str, default="stage",
                        help='The name of the folder where the model is stored.')






    args = parser.parse_args()
    seed_everything(args.seed)
    with open(os.path.join('config', f'{args.config}.yaml')) as f:
        config = yaml.safe_load(f)
        config['name'] = args.name
    if args.resume:
        print(f'Resume to {config["name"]}.')
        train(config, resume=True)
    elif args.test:
        print(f'Test {config["name"]}.')
        train(config, test=True)
    else:
        train(config,args)



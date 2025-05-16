import argparse
import os
import torch
import yaml
import random
import numpy as np

import transformer_base as runfile

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## 2
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--config', type=str, default="config",
                        help='Configuration filename for restoring the model.')
    # parser.add_argument('--sigle', type=bool, default=False,
    #                     help='if you use sigle dataset set True, else DG set False')
    ## dataset = 'En2Fr'  'En2Es'  'En2It'   'MeiMemo'
    parser.add_argument('--dataset', type=str,required=True,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--sigle', type=bool,default=False,
                        help='Configuration filename for restoring the model.')
    args = parser.parse_args()
    runfile.seed_everything(args.seed)
    with open(os.path.join('config', f'{args.config}.yaml')) as f:
        config = yaml.safe_load(f)
    runfile.train(config,args, test=True)



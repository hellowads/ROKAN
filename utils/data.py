import os
from typing import Dict

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

class PredictorDataset(Dataset):
    #
    def __init__(self,En2De=None,En2Es=None, Duolingo=None,MeiMemo=None):
        if En2De != None:
            datasEn2De = np.load(En2De)
            ones = torch.ones(datasEn2De['x'].shape[0], datasEn2De['x'].shape[1], 1)
            inputs_datasEn2De = torch.cat((torch.tensor(datasEn2De['x']), ones), dim=2)
            targets_datasEn2De = datasEn2De['y']

        if En2Es != None:
            datasEn2Es = np.load(En2Es)
            twos = torch.ones(datasEn2Es['x'].shape[0], datasEn2Es['x'].shape[1], 1) * 2
            inputs_datasEn2Es = torch.cat((torch.tensor(datasEn2Es['x']), twos), dim=2)
            targets_datasEn2Es = datasEn2Es['y']

        if Duolingo != None:
            datasDuolingo = np.load(Duolingo)
            threes = torch.ones(datasDuolingo['x'].shape[0], datasDuolingo['x'].shape[1], 1) * 3
            inputs_Duolingo = torch.cat((torch.tensor(datasDuolingo['x']), threes), dim=2)
            targets_Duolingo = datasDuolingo['y']
        if MeiMemo != None:
            datasMeiMemo = np.load(MeiMemo)
            four = torch.ones(datasMeiMemo['x'].shape[0], datasMeiMemo['x'].shape[1], 1) * 4
            inputs_MeiMemo = torch.cat((torch.tensor(datasMeiMemo['x']), four), dim=2)
            targets_MeiMemo = datasMeiMemo['y']

        # flag to identify datasets

        # twos = torch.ones(datasEn2Es['x'].shape[0], datasEn2Es['x'].shape[1], 1) * 2
        # threes = torch.ones(datasDuolingo['x'].shape[0], datasDuolingo['x'].shape[1], 1)*3
        # four = torch.ones(datasMeiMemo['x'].shape[0], datasMeiMemo['x'].shape[1], 1)*4
        #
        #
        #
        # inputs_datasEn2Es = torch.cat((torch.tensor(datasEn2Es['x']), twos), dim=2)
        # targets_datasEn2Es = datasEn2Es['y']
        #
        # inputs_Duolingo = torch.cat((torch.tensor(datasDuolingo['x']),threes ), dim=2)
        # targets_Duolingo = datasDuolingo['y']
        #
        # inputs_MeiMemo=  torch.cat((torch.tensor(datasMeiMemo['x']),four ), dim=2)
        # targets_MeiMemo = datasMeiMemo['y']

        # self.inputs  =  inputs_MeiMemo
        # self.targets =  targets_MeiMemo

        # combine data
        if En2De == None:
            combined_x = inputs_MeiMemo
            combined_y = targets_MeiMemo
        else:
            combined_x = np.concatenate((inputs_datasEn2De, inputs_datasEn2Es, inputs_Duolingo), axis=0)
            combined_y = np.concatenate((targets_datasEn2De, targets_datasEn2Es, targets_Duolingo ),  axis=0)

        # combined_x = np.array(inputs_MeiMemo)
        # combined_y = np.array(targets_MeiMemo)
        ## shuffle
        indices = np.arange(combined_x.shape[0])
        np.random.shuffle(indices)

        self.inputs  =  combined_x[indices]
        self.targets =  combined_y[indices]

        # min = np.min(self.inputs.reshape(-1, 11), axis=0)
        # max = np.max(self.inputs.reshape(-1, 11), axis=0)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int):
        x, y = self.inputs[idx], self.targets[idx]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __minmax__(self):
        min = np.min(self.inputs.reshape(-1, 9), axis=0)
        max = np.max(self.inputs.reshape(-1, 9), axis=0)
        return torch.tensor(min, dtype=torch.float32), torch.tensor(max, dtype=torch.float32)


class ValAndTestDataset(Dataset):
    def __init__(self, DataSets: str):
        datas = np.load(DataSets)

        self.inputs  = datas['x']
        self.targets = datas['y']

        # min = np.min(self.inputs.reshape(-1, 11), axis=0)
        # max = np.max(self.inputs.reshape(-1, 11), axis=0)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int):
        x, y = self.inputs[idx], self.targets[idx]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __minmax__(self):
        min = np.min(self.inputs.reshape(-1, 9), axis=0)
        max = np.max(self.inputs.reshape(-1, 9), axis=0)
        return torch.tensor(min, dtype=torch.float32), torch.tensor(max, dtype=torch.float32)

def reValAndTest(Anki: str, Duolingo: str,MeiMemo: str):
    return {
        'Anki':ValAndTestDataset(Anki),
        'Duolingo':ValAndTestDataset(Duolingo),
        'MeiMemo':ValAndTestDataset(MeiMemo)
    }

def get_datasets(dataset: str):
    # dataset = 'data/multi_data/En2Es'
    # dataset = 'data/multi_data/En2It'
    # dataset = 'data/multi_data/MeiMemo'
    # dataset = 'data/multi_data/frsr'
    En2Es = 'data/multi_data/En2Es'
    En2Fr = 'data/multi_data/En2Fr'
    Anki = 'data/multi_data/frsr'
    MaiMemo = 'data/multi_data/MeiMemo'

    # ## momo
    if dataset == 'MeiMemo':
        train_datasets_conbime = PredictorDataset(En2Es+f'/train.npz',En2Fr+f'/train.npz',Anki+f'/train.npz')
        min = np.min(train_datasets_conbime.inputs.reshape(-1, 9), axis=0)
        max = np.max(train_datasets_conbime.inputs.reshape(-1, 9), axis=0)
        #
        # train_datasets_conbime = PredictorDataset(MeiMemo=dataset3 + f'/train.npz')
        #
        val_datasets_conbime =  PredictorDataset(En2Es+f'/train.npz',En2Fr+f'/train.npz',Anki+f'/train.npz')
        # val_datasets_conbime = PredictorDataset(MeiMemo=dataset3 + f'/val.npz')

        test_datasets_conbime = PredictorDataset(MeiMemo= MaiMemo + f'/test.npz')
    elif dataset == 'En2Es' :
        ## En2Fr
        train_datasets_conbime = PredictorDataset(En2Fr + f'/train.npz', Anki + f'/train.npz',
                                                  MaiMemo + f'/train.npz')
        min = np.min(train_datasets_conbime.inputs.reshape(-1, 9), axis=0)
        max = np.max(train_datasets_conbime.inputs.reshape(-1, 9), axis=0)
        #
        # train_datasets_conbime = PredictorDataset(MeiMemo=dataset3 + f'/train.npz')
        #
        val_datasets_conbime = PredictorDataset(En2Fr + f'/train.npz', Anki + f'/train.npz',
                                                  MaiMemo + f'/train.npz')      # val_datasets_conbime = PredictorDataset(MeiMemo=dataset3 + f'/val.npz')

        test_datasets_conbime = PredictorDataset(MeiMemo=En2Es + f'/test.npz')
    elif dataset == 'frsr':
        ## En2Es
        train_datasets_conbime = PredictorDataset(MaiMemo + f'/train.npz', En2Es + f'/train.npz',
                                                  En2Fr + f'/train.npz')
        min = np.min(train_datasets_conbime.inputs.reshape(-1, 9), axis=0)
        max = np.max(train_datasets_conbime.inputs.reshape(-1, 9), axis=0)
        #
        # train_datasets_conbime = PredictorDataset(MeiMemo=dataset3 + f'/train.npz')
        #
        val_datasets_conbime = PredictorDataset(MaiMemo + f'/train.npz', En2Es + f'/train.npz',
                                                  En2Fr + f'/train.npz')
        # val_datasets_conbime = PredictorDataset(MeiMemo=dataset3 + f'/val.npz')

        test_datasets_conbime = PredictorDataset(MeiMemo=Anki + f'/test.npz')
    elif dataset == 'En2Fr':
        ## En2It
        train_datasets_conbime = PredictorDataset(MaiMemo + f'/train.npz', En2Es + f'/train.npz',
                                                  Anki + f'/train.npz')
        min = np.min(train_datasets_conbime.inputs.reshape(-1, 9), axis=0)
        max = np.max(train_datasets_conbime.inputs.reshape(-1, 9), axis=0)
        #
        # train_datasets_conbime = PredictorDataset(MeiMemo=dataset3 + f'/train.npz')
        #
        val_datasets_conbime = PredictorDataset(MaiMemo + f'/train.npz', En2Es + f'/train.npz',
                                                  Anki + f'/train.npz')

        # val_datasets_conbime = PredictorDataset(MeiMemo=dataset3 + f'/val.npz')

        test_datasets_conbime = PredictorDataset(MeiMemo=En2Fr + f'/test.npz')

    return {
        'train': train_datasets_conbime,
        'val': val_datasets_conbime,
        'test': test_datasets_conbime
    }
def get_datasetssingle(dataset: str):

    train_datasets_conbime = PredictorDatasetSigle(dataset+f'/train.npz')
    #
    # train_datasets_conbime = PredictorDataset(MeiMemo=dataset3 + f'/train.npz')
    #
    val_datasets_conbime =  PredictorDatasetSigle(dataset+f'/val.npz')
    # val_datasets_conbime = PredictorDataset(MeiMemo=dataset3 + f'/val.npz')

    test_datasets_conbime = PredictorDatasetSigle(dataset + f'/test.npz')
    # train_datasets_conbime.inputs = (train_datasets_conbime.inputs - np.array(min)) / (np.array(max) - np.array(min) + 1)
    # train_datasets_conbime.inputs = (train_datasets_conbime.inputs - np.array(min)) / (
    #             np.array(max) - np.array(min) + 1)

    return {
        'train': train_datasets_conbime,
        'val': val_datasets_conbime,
        'test': test_datasets_conbime
    }
# def get_datasets2(dataset: str):
#     # dataset0 = 'data/multi_data/En2De'
#     # dataset1 = 'data/multi_data/En2Es'
#     # dataset2 = 'data/multi_data/Duolingo'
#     dataset3 = 'data/multi_data/MeiMemo'
#
#     train_datasets_conbime = PredictorDataset(dataset3+f'/train.npz')
#
#     val_datasets_conbime =  PredictorDataset(dataset3+f'/val.npz')
#
#     test_datasets_conbime = PredictorDataset(dataset3 + f'/test.npz')
#     return {
#         'train': train_datasets_conbime,
#         'val': val_datasets_conbime,
#         'test': test_datasets_conbime
#     }


def get_dataloaders(datasets: Dict[str, Dataset],
                    batch_size: int,
                    num_workers: int = 0) -> Dict[str, DataLoader]:
    return {key: DataLoader(dataset=ds,
                            batch_size=batch_size,
                            shuffle=(key == 'train'),
                            num_workers=num_workers) for key, ds in datasets.items()}



class PredictorDatasetSigle(Dataset):
    #
    def __init__(self,dataset):

        datasMeiMemo = np.load(dataset)
        four = torch.ones(datasMeiMemo['x'].shape[0], datasMeiMemo['x'].shape[1], 1) * 4
        inputs = torch.cat((torch.tensor(datasMeiMemo['x']), four), dim=2)
        targets = datasMeiMemo['y']

        # flag to identify datasets

        # twos = torch.ones(datasEn2Es['x'].shape[0], datasEn2Es['x'].shape[1], 1) * 2
        # threes = torch.ones(datasDuolingo['x'].shape[0], datasDuolingo['x'].shape[1], 1)*3
        # four = torch.ones(datasMeiMemo['x'].shape[0], datasMeiMemo['x'].shape[1], 1)*4
        #
        #
        #
        # inputs_datasEn2Es = torch.cat((torch.tensor(datasEn2Es['x']), twos), dim=2)
        # targets_datasEn2Es = datasEn2Es['y']
        #
        # inputs_Duolingo = torch.cat((torch.tensor(datasDuolingo['x']),threes ), dim=2)
        # targets_Duolingo = datasDuolingo['y']
        #
        # inputs_MeiMemo=  torch.cat((torch.tensor(datasMeiMemo['x']),four ), dim=2)
        # targets_MeiMemo = datasMeiMemo['y']

        # self.inputs  =  inputs_MeiMemo
        # self.targets =  targets_MeiMemo

        # combine data
        # if En2De == None:
        #     combined_x = inputs_MeiMemo
        #     combined_y = targets_MeiMemo
        # else:
        #     combined_x = np.concatenate((inputs_datasEn2De, inputs_datasEn2Es, inputs_Duolingo), axis=0)
        #     combined_y = np.concatenate((targets_datasEn2De, targets_datasEn2Es, targets_Duolingo ),  axis=0)

        # combined_x = np.array(inputs_MeiMemo)
        # combined_y = np.array(targets_MeiMemo)
        ## shuffle
        # indices = np.arange(combined_x.shape[0])
        # np.random.shuffle(indices)

        self.inputs  =  np.array(inputs)
        self.targets =  np.array(targets)

        # min = np.min(self.inputs.reshape(-1, 11), axis=0)
        # max = np.max(self.inputs.reshape(-1, 11), axis=0)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int):
        x, y = self.inputs[idx], self.targets[idx]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __minmax__(self):
        min = np.min(self.inputs.reshape(-1, 9), axis=0)
        max = np.max(self.inputs.reshape(-1, 9), axis=0)
        return torch.tensor(min, dtype=torch.float32), torch.tensor(max, dtype=torch.float32)
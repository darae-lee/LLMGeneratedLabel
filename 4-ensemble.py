import warnings
warnings.filterwarnings(action='ignore')

import os
import gc
import ast
import argparse
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import math
from sklearn.metrics import r2_score
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.autograd import Variable
from corr_model.dataloader import *
from corr_model.model import *
from corr_model.model_nomax import *
from corr_model.model_relative2 import *

import statistics
import numpy as np


def ensemble(path):
    path_list = glob.glob(f'{path}/*')
    file_list = []

    for file_name in path_list:
        file_list.append(pd.read_csv(file_name))
        print(file_list)

    ensemble_csv = pd.DataFrame(columns=["adm2","adm3","sigmaf",'pop'],dtype=object)

    for index in range(0, len(file_list[0])):
        mean = 0
        adm2 = file_list[0].iloc[index]['adm2']
        adm3 = file_list[0].iloc[index]['adm3']
        pop = file_list[0].iloc[index]['pop']

        for file in file_list:
            mean += file.iloc[index]['sigmaf']
        mean /= len(file_list)

        newdata = pd.DataFrame({'adm2':[adm2], 'adm3':[adm3],'sigmaf':[mean],'pop':[pop]})
        ensemble_csv = ensemble_csv._append(newdata, ignore_index=True)
    
    print("Pearson corr: {}".format(round(ensemble_csv[['sigmaf','pop']].corr(method='pearson')['sigmaf']['pop'],4)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    args = parser.parse_args()

    ensemble(args.path)
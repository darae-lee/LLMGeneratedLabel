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


def ensemble(is_grid, label_types, contexts):
    
    file_list = []
    n = len(label_type)
    if is_grid:
        for i in range(n):
            label_type = label_types[i]
            context = contexts[i]
            file_list.append(pd.read_csv(f'./finalcsv_grid/context{context}_type{label_type}.csv'))
    else:
        for i in range(n):
            label_type = label_types[i]
            context = contexts[i]
            file_list.append(pd.read_csv(f'./finalcsv/context{context}_type{label_type}.csv'))

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
    
    print(f"Context: {context}, Types: {label_types}")
    print("Pearson corr: {}".format(round(finalcsv1[['sigmaf','pop']].corr(method='pearson')['sigmaf']['pop'],4)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate labels from images.")
    parser.add_argument("--grid", type=bool, help="Is this grid or not? true or false")
    parser.add_argument("--type", type=int, nargs="+", help="Specify the list of type (among 0, 1, 2, 3, and 4).")
    parser.add_argument("--context", type=int, nargs="+", help="Specify the list of context (among 0, 1, 2, or 3).")
    args = parser.parse_args()

    label_types = args.type if args.type else []

    contexts = args.context if args.context else []

    ensemble(args.grid, label_types, contexts)
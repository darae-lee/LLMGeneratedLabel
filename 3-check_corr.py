import warnings
warnings.filterwarnings(action='ignore')

import os
import gc
import ast
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
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
import re

def evaluation (singlemodel,adm3_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = models.resnet18(pretrained = True)
    feature_size = net.fc.in_features
    net.fc = nn.Sequential()
    model = BinMultitask_nomax(net, feature_size, 10, 200, ordinal=False)

    MODELNAME = singlemodel
    match = re.match(r"./save_model/checkpoint_context(\d+)_type(\d+)\.ckpt", MODELNAME)
    context = int(match.group(1))
    label_type = int(match.group(2))

    parallelnum = 0
    try:
        model.load_state_dict(torch.load(MODELNAME)['state_dict'], strict=True)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, device_ids = [0,1,2,3])
    except:
        parallelnum = 1

    if parallelnum == 1:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, device_ids = [0,1,2,3])
        model.load_state_dict(torch.load(MODELNAME)['state_dict'], strict=True)

    model.cuda()
    print("Model loaded!")
    
    model.eval()

    totalcsv = pd.DataFrame(columns=["adm2","adm3","sigmaf",'imgnum','avgf'],dtype=object)
    for batch in enumerate(adm3_loader):
        mcity = batch[1]['city'][0]
        adm3imgnum = len(batch[1]['images'][0])
 
        if batch[1]['images'][0].dim() == 4:
            mysum = round(torch.sum(model(batch[1]['images'][0].cuda())[1]).item(),2)
        else:
            adm3imgnum = 1
            mysum = round(torch.sum(model(batch[1]['images'][0].unsqueeze(0).cuda())[1]).item(),2)
        newdata = pd.DataFrame({'adm2':[mcity], 'adm3':[batch[1]['adm3'][0]],'sigmaf':[mysum],'imgnum':[adm3imgnum],'avgf':[mysum/adm3imgnum]})
        totalcsv = totalcsv._append(newdata, ignore_index=True)

    citygus = pd.read_csv('./Corr_checking_metadata/RealAdm3s_in_Adm2.csv',index_col=0)
    
    adm3popcsv = pd.read_csv('./Corr_checking_metadata/AfterKSC_Adm3PopCsv.csv',index_col=0)
    popplusf = pd.merge(totalcsv,adm3popcsv,on=['adm2','adm3'])
    popplusf = popplusf[['adm2','adm3','sigmaf','pop']]
    adm2sumcsv = popplusf.groupby(['adm2'],as_index=False).sum()
    adm2sumcsv = adm2sumcsv[['adm2','sigmaf','pop']]
    adm2sumcsv.columns = ['adm2','bunmo','bunja']
    finalcsv = pd.merge(popplusf, adm2sumcsv, on=['adm2'])
    finalcsv['predicted'] = finalcsv['sigmaf'] * finalcsv['bunja'] / finalcsv['bunmo']
    finalcsv=finalcsv.dropna()
    print("[LLM]Pearson corr: {}".format(round(finalcsv[['sigmaf','pop']].corr(method='pearson')['sigmaf']['pop'],4)))
    finalcsv = finalcsv[['adm2','adm3','sigmaf','pop','predicted']]
    finalcsv.to_csv(f'./finalcsv/context{context}_type{label_type}.csv', index=False)
    
    del adm3_loader
    gc.collect()
    torch.cuda.empty_cache()
    
    model.eval()
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    allcsv = pd.read_csv('./Corr_checking_metadata/AfterKSC_(RowByRow)RealAdm3s_in_Adm2.csv',index_col=0)
    allcsv.fillna(value='None',inplace=True)

    totalgridcsv = pd.DataFrame(columns=["adm1","adm2","adm3","gridID",'gridscore'],dtype=object)
    for row in allcsv.itertuples():
        myprovince = row.adm1
        mycity = row.adm2
        myadm3 = row.adm3

        if myprovince == 'None':
            image_root_path = "{}/{}/{}".format('./FinalAlbum_광역시Adm2_pruned', mycity, myadm3)
        else:
            image_root_path = "{}/{}/{}".format('./FinalAlbum_광역시Adm2_pruned', myprovince, myadm3)
        for gridimage in os.listdir(image_root_path):
            gridID = gridimage.split('.')[0]
            tempimg = transform(io.imread("{}/{}".format(image_root_path, gridimage))).unsqueeze(0).cuda()
            gridscore = round(model(tempimg)[1].item(),4)
            newdata = {'adm1':myprovince, 'adm2':mycity, 'adm3':myadm3, 'gridID':gridID, 'gridscore':gridscore}
            totalgridcsv = totalgridcsv._append(newdata, ignore_index=True)
    totalgridcsv['pop'] = 0.0
    for row in totalgridcsv.itertuples():
        myprovince = row.adm1
        mycity = row.adm2
        myadm3 = row.adm3
        mygridID = row.gridID
        if row.Index != 0:
            prevpath = pop_root_path
        if myprovince == 'None':
            pop_root_path = "{}/{}.csv".format('./FinalCsv_광역시Adm2', mycity)
        else:
            pop_root_path = "{}/{}.csv".format('./FinalCsv_광역시Adm2', myprovince)
        if row.Index == 0:
            popcsv = pd.read_csv(pop_root_path,index_col=0)
            prevpath = pop_root_path
        if prevpath != pop_root_path:
            popcsv = pd.read_csv(pop_root_path,index_col=0)
        if len(popcsv[popcsv['areaid']==mygridID]) == 1:
            totalgridcsv.iloc[row.Index,5] = popcsv[popcsv['areaid']==mygridID]['pop'].item()
        else:
            totalgridcsv.iloc[row.Index,5] = popcsv[popcsv['areaid']==mygridID].iloc[0]['pop']
    adm2sumcsv = totalgridcsv.groupby(['adm1','adm2'],as_index=False).sum()
    adm2sumcsv = adm2sumcsv[['adm1','adm2','gridscore','pop']]
    adm2sumcsv.columns = ['adm1','adm2','bunmo','bunja']
    finalcsv = pd.merge(totalgridcsv, adm2sumcsv, on=['adm1','adm2'])
    finalcsv['predicted'] = finalcsv['gridscore'] * finalcsv['bunja'] / finalcsv['bunmo']
    finalcsv=finalcsv.dropna()
    print("[LLM]Pearson corr: {}".format(round(finalcsv[['gridscore','pop']].corr(method='pearson')['gridscore']['pop'],4)))
    finalcsv = finalcsv[['adm2','adm3','gridscore','pop','predicted']]
    finalcsv.to_csv(f'./finalcsv_grid/context{context}_type{label_type}.csv', index=False)
    

if __name__ == '__main__':
    modellist = glob.glob('./save_model/*')
    adm3_data = Adm3Dataset(metadata = './Corr_checking_metadata/AfterKSC_(RowByRow)RealAdm3s_in_Adm2.csv',
                        root_csv = './FinalCsv_광역시Adm2',
                        root_album = './FinalAlbum_광역시Adm2_pruned',
                        transform=transforms.Compose([ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
    adm3_loader = torch.utils.data.DataLoader(adm3_data, batch_size=1, shuffle=False, num_workers=1)
    for onemodel in modellist:
        print(onemodel)
        evaluation(onemodel,adm3_loader)
        
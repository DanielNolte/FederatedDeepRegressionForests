# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 10:56:48 2021

@author: Daniel
"""
#import base calss

import utils
opt = utils.parse_arg()

from torch.utils.data import Dataset
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch

#%%
class CCLEFedImages:
    def __init__(self,opt):
        """
        Load already mapped image data for REFINED-CNN and REFINED-DRF
        """
        pathInit = '/CCLE'+opt.mapping+'MappingInitImgSamples.pickle'
        pathRest = '/CCLE'+opt.mapping+'MappingRestImgSamples.pickle'
        pathDrugNames = '/CCLEDrugNamesInit&Rest.pickle'
        path = opt.data_path
        with open(path+pathInit, 'rb') as handle:
            Xinit = pickle.load(handle)
            Yinit = pickle.load(handle)
        with open(path+pathRest, 'rb') as handle:
            Xrest = pickle.load(handle)
            Yrest = pickle.load(handle)
        with open(path+pathDrugNames, 'rb') as handle:
            self.DrugsInit = pickle.load(handle)
            self.DrugsRest = pickle.load(handle)
        
        print("Data Loaded")
        self.X = Xrest
        self.Y = Yrest 
        self.Xinit = np.array(pd.DataFrame(Xinit))
        self.Yinit = np.array(pd.DataFrame(Yinit))

        
    #return_split_data: Function called by runSim to obtain federated
    #data based on numClients,fracTestData,and localBatchSizePerRound
    def split_data(self,numClients,fracTestData,seed,iid):
        """
        Splits the loaded data based on input args
        Args:
            numClients (int): Number of clients in federation
            fracTestData (float): Fraction of test data
            seed (int): Random seed for data split
            iid (int): IID parameter for determining the degree of 
                        heterogenity among clients 
        Returns:
            XYtrain (List of Datasets): Dataset for each client
            XYtest (Dataset): Test dataset
            XYinit (Dataset): Initial dataset
            centralizedData (Dataset): Centralized dataset of all train and 
                                        initial data
            XYval (Dataset): Validation dataset
        """
        np.random.seed(seed=seed)
        XYtest = {}
        XYinit = {}
        XYtrain = {}
        centralizedData = {}
        # Set data
        drugCellX = pd.DataFrame(self.X)
        drugCellX['DrugName'] = self.DrugsRest
        drugCellX['Y'] = self.Y
        drugCellY = self.Y
        
        # Get test and init size
        numSamples =len(self.Y)+len(self.Yinit)
        testSamples = round(numSamples*fracTestData)


        print('Number of Samples:' +str(numSamples))
        #Split data
        X_train, X_test, y_train, y_test = train_test_split(drugCellX, drugCellY, test_size=testSamples,random_state=seed)
        X_Val, X_test, y_Val, y_test = train_test_split(X_test, y_test, test_size=.5,random_state=seed)

    
        uniqueDrugs = X_train['DrugName'].unique()
        clientDrugs = [[] for i in range(numClients)]
        remainingDrugs = uniqueDrugs
        for i in range(numClients):
            drugRemainForClient = remainingDrugs[~np.isin(remainingDrugs,clientDrugs[i])]
            drug = np.random.choice(drugRemainForClient)
            clientDrugs[i].append(drug)
            remainingDrugs = np.delete(remainingDrugs,np.where(remainingDrugs == drug))
            drugRemainForClient = remainingDrugs[~np.isin(remainingDrugs,clientDrugs[i])]
            drug = np.random.choice(drugRemainForClient)
            clientDrugs[i].append(drug)
            remainingDrugs = np.delete(remainingDrugs,np.where(remainingDrugs == drug))

        remainingDrugs = uniqueDrugs
        for i in range(numClients):
            for drugsPerClient in range(int(iid-2)):
                drugRemainForClient = remainingDrugs[~np.isin(remainingDrugs,clientDrugs[i])]
                drug = np.random.choice(drugRemainForClient)
                clientDrugs[i].append(drug)
        
            
        clientDrugs = pd.DataFrame(clientDrugs)
        drugGroups = X_train.groupby('DrugName')
        splitDrugs = {}
        for name,group in drugGroups:
            n = clientDrugs.where(clientDrugs==name).count().sum()
            splitDrugs[name] = np.array_split(group, n)

        #Client split the non init training data
        splitData = [[] for i in range(numClients)]     
        for i in range(numClients):
            for drug in clientDrugs.iloc[i]:
                allSamplesOfDrug = splitDrugs[drug]
                splitData[i].append(allSamplesOfDrug[0])
                splitDrugs[drug] = allSamplesOfDrug[1:]

        for i in range(len(splitData)):
            splitData[i] = pd.concat(splitData[i]).drop('DrugName',axis=1)
            splitData[i] = {'X':splitData[i][[0,1]].to_numpy(),'Y':splitData[i].Y.to_numpy()}
            splitData[i] = FedCCLEDataSet(splitData[i])   
        print("Data splitting is done!")
        XYtrain = splitData
        X_train = X_train.drop(['DrugName','Y'],axis=1).to_numpy()
        X_Val = X_Val.drop(['DrugName','Y'],axis=1).to_numpy()
        X_test = X_test.drop(['DrugName','Y'],axis=1).to_numpy()
        centralizedData = {'X':np.concatenate((X_train,self.Xinit), axis=0),'Y':np.concatenate((y_train, self.Yinit), axis=0)}
        XYval = {'X':X_Val,'Y':y_Val}
        XYtest = {'X':X_test,'Y':y_test}
        XYinit = {'X':self.Xinit,'Y':self.Yinit}
        centralizedData = FedCCLEDataSet(centralizedData)
        XYval = FedCCLEDataSet(XYval)
        XYtest = FedCCLEDataSet(XYtest)
        XYinit = FedCCLEDataSet(XYinit)
        del self.Xinit,self.Yinit,X_test,y_test,splitData,self.Y,self.X,drugCellX,drugCellY,X_train,y_train
        return XYtrain,XYtest,XYinit,centralizedData,XYval
#%%

class FedCCLEDataSet(Dataset):
    def __init__(self, data):
        """
        Args:
            data (Dict of numpy arrays): Dict of 'X' and 'Y' containing numpy arrays
        """
        self.X = data['X']
        self.Y = data['Y']

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):        
        x = [torch.from_numpy(self.X[idx][0]).float(),torch.from_numpy(self.X[idx][1]).float()]
        y = torch.from_numpy(np.asarray(self.Y[idx])).float()
        sample = {'data': x, 'target': y}

        return sample
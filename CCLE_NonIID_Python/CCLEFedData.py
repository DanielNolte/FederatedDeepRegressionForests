# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 10:56:48 2021

@author: Daniel
"""
#import base calss
import utils
opt = utils.parse_arg()

from torch.utils.data import Dataset
import pickle as pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch

#%%
class CCLEFedData:
    def __init__(self):
        """
        Load data for ANN
        """
        path = opt.data_path
        pathDrugNames = '/CCLEDrugNamesInit&Rest.pickle'
        with open(path+'/initCCLE2.pickle', 'rb') as handle:
            Xinit = pickle.load(handle)
            Yinit = pickle.load(handle)
        with open(path+'/restCCLE2.pickle', 'rb') as handle:
            Xrest = pickle.load(handle)
            Yrest = pickle.load(handle)
        with open(path+pathDrugNames, 'rb') as handle:
            self.DrugsInit = pickle.load(handle)
            self.DrugsRest = pickle.load(handle)
        print("Data Loaded")
        self.Xx = Xrest
        self.Yy = Yrest 
        self.Xinit = Xinit
        self.Yinit = Yinit


        
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
        assert iid >=2 & iid <=24, 'iid parameter must be within range 2-24'
        XYtest = {}
        XYinit = {}
        XYtrain = {}
        centralizedData = {}
        # Set data
        drugCellX = self.Xx
        drugCellX = pd.DataFrame(self.Xx)
        drugCellX['DrugName'] = self.DrugsRest
        drugCellX['Y'] = self.Yy
        drugCellY = self.Yy
        
        # Get test and init size
        numSamples =len(self.Yy)+len(self.Yinit)
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
        splitData = [[] for i in range(numClients)]     
        for i in range(numClients):
            for drug in clientDrugs.iloc[i]:
                allSamplesOfDrug = splitDrugs[drug]
                splitData[i].append(allSamplesOfDrug[0])
                splitDrugs[drug] = allSamplesOfDrug[1:]
        for i in range(len(splitData)):
            splitData[i] = pd.concat(splitData[i]).drop('DrugName',axis=1)
            splitData[i] = {'X':splitData[i].drop('Y',axis=1).to_numpy(),'Y':splitData[i].Y.to_numpy()}
            splitData[i] = FedCCLEDataSet(splitData[i])

        print("Data splitting is done!")
        XYtrain = splitData
        X_train = X_train.drop(['DrugName','Y'],axis=1)
        X_Val = X_Val.drop(['DrugName','Y'],axis=1)
        X_test = X_test.drop(['DrugName','Y'],axis=1)
        centralizedData = {'X':np.concatenate((X_train.to_numpy(), self.Xinit.to_numpy()), axis=0),'Y':np.concatenate((y_train.to_numpy(), self.Yinit.to_numpy()), axis=0)}
        XYval = {'X':X_Val.to_numpy(),'Y':y_Val.to_numpy()}
        XYtest = {'X':X_test.to_numpy(),'Y':y_test.to_numpy()}
       
        XYinit = {'X':self.Xinit.to_numpy(),'Y':self.Yinit.to_numpy()}
        centralizedData = FedCCLEDataSet(centralizedData)
        XYval = FedCCLEDataSet(XYval)
        XYtest = FedCCLEDataSet(XYtest)
        XYinit = FedCCLEDataSet(XYinit)
        del X_test,y_test,splitData,drugCellX,drugCellY
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
        x = torch.tensor(self.X[idx]).float()
        y = torch.tensor(self.Y[idx]).float()
        sample = {'data': x, 'target': y}
        return sample
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 14:29:47 2020

@author: Daniel
"""
import utils # other utilities
from Client import Client
import torch
#import agg rule
# from  AggRules import fed_avg
#import Federated Model
from FedDRF import FedDRF
from resnet import CNNCCLE, ANNCCLE
import time
import pandas as pd
import numpy as np

class Server:
    def __init__(self,numClients,fracOfClientsPerRound,fracTestData,opt):
        """
        Depending on the options, the server initialized the datasets, trains
        or evaluates centralized and initial models, and creates the simulated
        federation based on input options
        
        Args:
            numClients (int): number of clients in federation
            fracOfClientsPerRound (float): Fraction of clients to include in each training round
            fracTestData (float): Fraction of test data
            opt: Input options
        """
        self.opt = opt
        self.clientValLossHist = []
        self.clientTestLossHist = []
        self.clientTrainLossHist = []
        self.serLossHist = []
        # Set GPU if selected
        if self.opt.cuda:
            torch.cuda.set_device(self.opt.gpuid)

        #set fracOfClientsPerRound
        self.fracOfClientsPerRound = fracOfClientsPerRound
        
        #Use user defined datasetClass for obtaining the train/test datasets
        if self.opt.model_type == 'ANN':
            from CCLEFedData import CCLEFedData
            fedDatset = CCLEFedData()
        else:
            from CCLEFedImages import CCLEFedImages
            fedDatset = CCLEFedImages(opt)
        self.train,self.test,init,self.central,self.val= fedDatset.split_data(numClients,
                                fracTestData,self.opt.randomSeed,self.opt.iid)
        del fedDatset
        
        #If evalCentralized = True, then train and test the centralized model
        #for baseline performance
        if self.opt.train_central:
            print('Training Central Model')
            if self.opt.model_type == 'ANN':
                self.centralizedModel = ANNCCLE(self.opt)
            elif self.opt.model_type == 'FedDRF':
                self.centralizedModel = FedDRF(self.opt)
            elif self.opt.model_type == 'CNN':
                self.centralizedModel = CNNCCLE(self.opt)
            
            t = time.time()
            self.centralizedModel,lossHistory = self.centralizedModel.train1(self.central,self.val,False)
            elapsed = time.time() - t
            utils.save_model(self.centralizedModel, self.opt,'Central')
            utils.save_losses_time(lossHistory,elapsed,self.opt,'Central')
            
        if self.opt.eval_central:
            save_dir = utils.get_save_dir(self.opt,'Central')
            if self.opt.cuda:
                self.centralizedModel = torch.load(save_dir,map_location=torch.device('cuda'))
            else:
                self.centralizedModel = torch.load(save_dir,map_location=torch.device('cpu')) 
                self.centralizedModel.opt.cuda = 0
            
            print('Central Model Eval')  
            CentTrain = self.centralizedModel.evaluate_model(self.central)
            CentVal= self.centralizedModel.evaluate_model(self.val)
            CentTest = self.centralizedModel.evaluate_model(self.test)
            CentInitResults = pd.DataFrame(index=['RMSE','NRMSE','MedAE','NMAE','PCC','R2'])
            CentInitResults['CentTrain'+str(self.opt.randomSeed)] = CentTrain
            CentInitResults['CentVal'+str(self.opt.randomSeed)] = CentVal
            CentInitResults['CentTest'+str(self.opt.randomSeed)] = CentTest
            CentInitResults.drop(['RMSE','MedAE','NMAE','R2'])
            CentInitResults=CentInitResults.T
            print(CentInitResults)
                
                
        if self.opt.train:
            print('Training Initial Model')
            if self.opt.model_type == 'ANN':
                self.model = ANNCCLE(self.opt)
            elif self.opt.model_type == 'FedDRF':
                self.model = FedDRF(self.opt)
            elif self.opt.model_type == 'CNN':
                self.model = CNNCCLE(self.opt)
                
            t = time.time()
            _,lossHistory = self.model.train1(init,self.val,False)
            elapsed = time.time() - t
            utils.save_losses_time(lossHistory,elapsed,self.opt,'Init')
            utils.save_model(self.model, self.opt,'Init')
      
        if self.opt.eval:
            save_dir = utils.get_save_dir(self.opt,'Init')
            if self.opt.cuda:
                self.model = torch.load(save_dir,map_location=torch.device('cuda'))
                self.model.opt.cuda = 1
            else:
                self.model = torch.load(save_dir,map_location=torch.device('cpu'))
                self.model.opt.cuda = 0
            print('Initial Model Eval')  
            InitTrain = self.model.evaluate_model(init)
            InitVal = self.model.evaluate_model(self.val)
            InitTest = self.model.evaluate_model(self.test)
            CentInitResults = pd.DataFrame(index=['RMSE','NRMSE','MedAE','NMAE','PCC','R2'])
            CentInitResults['InitTrain'+str(self.opt.randomSeed)] = InitTrain
            CentInitResults['InitVal'+str(self.opt.randomSeed)] = InitVal
            CentInitResults['InitTest'+str(self.opt.randomSeed)] = InitTest
            CentInitResults.drop(['RMSE','MedAE','NMAE','R2'])
            CentInitResults=CentInitResults.T
            print(CentInitResults)
                

            

        self.model.opt = self.opt
        torch.cuda.empty_cache()
        # use Client to make n clients with partitioned data and selected model
        if (self.opt.model_type == 'ANN') & (self.opt.train):
            self.clients = [Client(self.train[client],self.val,ANNCCLE,self.opt) 
                            for client in range(numClients)]
        elif (self.opt.model_type == 'FedDRF') & (self.opt.train):
            self.clients = [Client(self.train[client],self.val,FedDRF,self.opt) 
                            for client in range(numClients)]
        elif (self.opt.model_type == 'CNN' ) & (self.opt.train): 
            self.clients = [Client(self.train[client],self.val,CNNCCLE,self.opt) 
                            for client in range(numClients)]


        
        
    def server_agg(self,clientUpdates,clientAccs,clientWeights):
        """
        Calls aggregation rule
        Args:
            clientUpdates (list): The client updates in the form of state dicts
            clientAccs(list): The client accuracys on the validation set
            clientWeights(list): The client weights/ number of samples
    
        Returns: the aggregated state dict and the aggreagted NRMSE
        """

        ##Use agg rule, clientUpdates, and clientWeights for
        ##aggregated update
        update,NRMSE = self.fed_avg(clientUpdates,clientAccs,clientWeights)
        return update,NRMSE  
    
    def fed_avg(self,clientUpdates,clientAccs,clientWeights):
        """
        Federated Averaging aggregation
        Args:
            clientUpdates(list): The client updates in the form of state dicts
            clientAccs(list): The client accuracys on the validation set
            clientWeights(list): The client weights/ number of samples
    
        Returns: 
            the aggregated state dict and the aggreagted NRMSE
        """

        #Calc total weight
        totalWeight = np.sum(clientWeights)
        aggregatedUpdate = {}
        #For each client, for each layer in the model, take the weighted average
        for ii, (name, value) in enumerate(clientUpdates[0].items()):
                    aggregatedUpdate[name] = value*(clientWeights[0]/totalWeight)
        for i in range(len(clientWeights)-1):
                i+=1
                for ii, (name, value) in enumerate(clientUpdates[i].items()):
                    aggregatedUpdate[name] += value*(clientWeights[i]/totalWeight)
        #Calculate weighted losses
        weightedAccs = np.array(clientAccs)*(clientWeights/totalWeight)
        NRMSE = np.sum(weightedAccs)
        del weightedAccs, totalWeight

        #Return data weighted average of updates and losses
        return aggregatedUpdate,NRMSE    
    def fed_eval(self,model):
        """
        Evaluation of a model on train, validation and test data
        
        Args:
            model:Model to evaluate
        Returns: 
            the train, validation,and test RMSE,NRMSE,MedAE,NMAE,PCC,R2
            as a DataFrame
        """
        Train = model.evaluate_model(self.central)
        Val = model.evaluate_model(self.val)
        Test = model.evaluate_model(self.test)
        
        Results = pd.DataFrame(index=['RMSE','NRMSE','MedAE','NMAE','PCC','R2'])
        Results['Train'+str(self.opt.randomSeed)] = Train
        Results['Val'+str(self.opt.randomSeed)] = Val
        Results['Test'+str(self.opt.randomSeed)] = Test
        Results.drop(['RMSE','MedAE','NMAE','R2'])
        Results=Results.T
        print(Results)
        return Results
        
        
    def run_one_round(self):
        """
        Runs one federated training round            
    
        Returns: the model update, aggregated NRMSE, NRMSE of the model, 
        the full model, and the loss history for saving
        """
        #Get current server weights
        serverUpdate = self.model.get_model_weights()
        #Randomly sample clients for round
        # clientsForRound = sample(self.clients,
        #                 round(len(self.clients)*self.fracOfClientsPerRound))
        clientsForRound = self.clients
        #Call selected clients for update using serverUpdate and zip results
        clientUpdates,clientValResults,clientWeights = zip(
            *[client.client_update(serverUpdate) 
              for client in clientsForRound])
        self.clientValLossHist.append(clientValResults)

        update,aggNRMSE = self.server_agg(clientUpdates,pd.DataFrame(clientValResults)[:][1],clientWeights)
        del clientUpdates,clientValResults,clientWeights,serverUpdate
        self.model = self.model.update_model(update)

        #If training_progress, run server eval
        if self.opt.training_progress:
                RMSE, NRMSE,MAE,NMAE,PCC,R2 = self.model.evaluate_model(self.val)
                self.serLossHist.append([RMSE, NRMSE,MAE,NMAE,PCC,R2])
                print('Aggregated Global Model Val NRMSE:'+str(NRMSE))
        return update,aggNRMSE,NRMSE,self.model,self.clientValLossHist,self.serLossHist
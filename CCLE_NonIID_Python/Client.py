# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 14:30:37 2020

@author: Daniel
"""
import torch
class Client:
    #Initialize client with given parameters
    def __init__(self,data,valData,model,opt):
        """
        Args:
            data (DataSet): clients split data
            valData (DataSet): Validation Data
            model (class): Model to use for training
            opt: Input options
        """
        self.clientIteration = 0
        self.data = data
        self.model = model(opt)
        self.opt = opt
        #Epochs controlled by Training rounds
        self.model.opt.epochs=1
        self.val = valData
        
            
    def reduce_lr(self):
        """
        Once called by server, the client reduced the learning rate
        """
        for g in self.model.optim.param_groups:
            g['lr'] =  g['lr']*0.1
            
    #Given the serverModelWeights, Initialize model with weights, and 
    #train on the batched data
    def client_update(self,serverModelParams):
        """
        Client runs one round of training on local client data
        Args:
            serverModelParams: Aggregated state_dict of model parameters
        Returns:
            model state_dict, 
            validation results, and number of samples
        """
        clientWeight = 0
        #update client model with provided server weights
        self.model.update_model(serverModelParams)
        #Fit model further on client data
        if self.opt.cuda:
            self.model.cuda()
        self.model,_ = self.model.train1(self.data,self.val,clientTrain=True)
        #Set client update to be the model weights after training
        clientUpdate = self.model.get_model_weights()

        #evalute on the current data
        valResults = self.model.evaluate_model(self.val)
        #update client weight with number of samples from current batch
        clientWeight = len( self.data)
        #increase clientIteration by 1
        self.clientIteration+=1
        if self.opt.cuda:
            self.model.cpu()
            torch.cuda.empty_cache()
        #Return client update,evalResults and the clientWeight
        return clientUpdate,valResults,clientWeight
    
    
    
        
    
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 15:31:12 2020

@author: Daniel
"""
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging       
from torch.utils.data import  DataLoader
from sklearn.metrics import median_absolute_error
from utils import NRMSE
import resnet
from torch.nn.parameter import Parameter
from collections import OrderedDict
import sys
# smallest positive float number
FLT_MIN = float(np.finfo(np.float32).eps)
FLT_MAX = float(np.finfo(np.float32).max)




class FeatureLayer(nn.Sequential):
    def __init__(self,opt):
        """
        Args:
            opt (int): input options
        """
        super(FeatureLayer, self).__init__()
        np.random.seed(opt.randomSeed)
        torch.manual_seed(opt.randomSeed)
        self.num_output = opt.num_output
        # a model using a resnet-like backbone is used for feature extraction 
        model = resnet.DRF_CNNCCLE(opt)
        self.add_module('feat_extract', model)
        
    def get_out_feature_size(self):
        return self.num_output

class Tree(nn.Module):
    def __init__(self, depth, feature_length, vector_length, is_cuda = True):
        """
        Args:
            depth (int): depth of the neural decision tree.
            feature_length (int): number of neurons in the last feature layer
            vector_length (int): length of the mean vector stored at each tree 
                                leaf node (1 for regression)
            use_cuda (boolean): whether to use GPU
        """
        super(Tree, self).__init__()
        self.depth = depth
        self.n_leaf = 2 ** depth
        self.feature_length = feature_length
        self.vector_length = vector_length
        self.is_cuda = is_cuda
        onehot = np.eye(feature_length)
        # randomly use some neurons in the feature layer to compute decision function
        using_idx = np.random.choice(feature_length, self.n_leaf, replace=False)
        self.feature_mask = onehot[using_idx].T
        self.feature_mask = Parameter(torch.from_numpy(self.feature_mask).type(torch.FloatTensor),requires_grad=False)
        # a leaf node contains a mean vector and a covariance matrix
        self.mean = np.ones((self.n_leaf, self.vector_length))
        self.mu_cache = []
        # use sigmoid function as the decision function
        self.decision = nn.Sequential(OrderedDict([
                        ('sigmoid', nn.Sigmoid()),
                        ]))
        # used for leaf node update
        self.covmat = np.array([np.eye(self.vector_length) for i in range(self.n_leaf)])
        # also stores the inverse of the covariant matrix for efficiency
        self.covmat_inv = np.array([np.eye(self.vector_length) for i in range(self.n_leaf)])
        # also stores the determinant of the covariant matrix for efficiency
        self.factor = np.ones((self.n_leaf))       
        if not is_cuda:
            self.mean = Parameter(torch.from_numpy(self.mean).type(torch.FloatTensor), requires_grad=False)
            self.covmat = Parameter(torch.from_numpy(self.covmat).type(torch.FloatTensor), requires_grad=False)
            self.covmat_inv = Parameter(torch.from_numpy(self.covmat_inv).type(torch.FloatTensor), requires_grad=False)
            self.factor = Parameter(torch.from_numpy(self.factor).type(torch.FloatTensor), requires_grad=False)  
        else:
            self.mean = Parameter(torch.from_numpy(self.mean).type(torch.FloatTensor).cuda(), requires_grad=False)
            self.covmat = Parameter(torch.from_numpy(self.covmat).type(torch.FloatTensor).cuda(), requires_grad=False)
            self.covmat_inv = Parameter(torch.from_numpy(self.covmat_inv).type(torch.FloatTensor).cuda(), requires_grad=False)
            self.factor = Parameter(torch.from_numpy(self.factor).type(torch.FloatTensor).cuda(), requires_grad=False)            


    def forward(self, x):
        """
        Args:
            param x (Tensor): input feature batch of size [batch_size, n_features]
        Return:
            (Tensor): routing probability of size [batch_size, n_leaf]
        """ 
        if x.is_cuda and not self.feature_mask.is_cuda:
            self.feature_mask = self.feature_mask.cuda()
        feats = torch.mm(x, self.feature_mask) 
        decision = self.decision(feats) 
        decision = torch.unsqueeze(decision,dim=2) 
        decision_comp = 1-decision
        decision = torch.cat((decision,decision_comp),dim=2) 
        batch_size = x.size()[0]
        
        mu = x.data.new(batch_size,1,1).fill_(1.)
        begin_idx = 1
        end_idx = 2
        for n_layer in range(0, self.depth):
            # mu stores the probability that a sample is routed to certain node
            # repeat it to be multiplied for left and right routing
            mu = mu.repeat(1, 1, 2)
            # the routing probability at n_layer
            _decision = decision[:, begin_idx:end_idx, :] # -> [batch_size,2**n_layer,2]
            mu = mu*_decision # -> [batch_size,2**n_layer,2]
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (n_layer+1)
            # merge left and right nodes to the same layer
            mu = mu.view(batch_size, -1, 1)
        mu = mu.view(batch_size, -1) 
        del x
        return mu

    def pred(self, x):
        p = torch.mm(self(x), self.mean)
        return p
    
    def update_label_distribution(self,target_batch):
        """
        fix the feature extractor of RNDF and update leaf node mean vectors and covariance matrices 
        based on a multivariate gaussian distribution 
        Args:
            param target_batch (Tensor): a batch of regression targets of size [batch_size, vector_length]
        """
        if self.is_cuda:
            self.cpu()
            target_batch = torch.cat(target_batch, dim = 0).cpu()
            mu = torch.cat(self.mu_cache, dim = 0).cpu()
        else:
            target_batch = torch.cat(target_batch, dim = 0)
            mu = torch.cat(self.mu_cache, dim = 0)
        batch_size = len(mu)
        # no need for gradient computation
        with torch.no_grad():
            leaf_prob_density = mu.data.new(batch_size, self.n_leaf)
            for leaf_idx in range(self.n_leaf):
            # vectorized code is used for efficiency
                temp = target_batch - self.mean[leaf_idx, :]
                leaf_prob_density[:, leaf_idx] = (self.factor[leaf_idx]*torch.exp(-0.5*(torch.mm(temp, self.covmat_inv[leaf_idx, :,:])*temp).sum(dim = 1))).clamp(FLT_MIN, FLT_MAX) # Tensor [batch_size, 1]
            nominator = (mu * leaf_prob_density).clamp(FLT_MIN, FLT_MAX) # [batch_size, n_leaf]
            denomenator = (nominator.sum(dim = 1).unsqueeze(1)).clamp(FLT_MIN, FLT_MAX) # add dimension for broadcasting
            zeta = nominator/denomenator # [batch_size, n_leaf]
            # new_mean if a weighted sum of all training samples
            new_mean = (torch.mm(target_batch.transpose(0, 1), zeta)/(zeta.sum(dim = 0).unsqueeze(0))).transpose(0, 1) # [n_leaf, vector_length]
            # allocate for new parameters
            new_covmat = new_mean.data.new(self.n_leaf, self.vector_length, self.vector_length)
            new_covmat_inv = new_mean.data.new(self.n_leaf, self.vector_length, self.vector_length)
            new_factor = new_mean.data.new(self.n_leaf)
            for leaf_idx in range(self.n_leaf):
                # new covariance matrix is a weighted sum of all covmats of each training sample
                weights = zeta[:, leaf_idx].unsqueeze(0)
                temp = target_batch - new_mean[leaf_idx, :]
                new_covmat[leaf_idx, :,:] = torch.mm(weights*(temp.transpose(0, 1)), temp)/(weights.sum())
                # update cache (factor and inverse) for future use
                new_covmat_inv[leaf_idx, :,:] = new_covmat[leaf_idx, :,:].inverse()
                new_factor[leaf_idx] = 1.0/max((torch.sqrt(new_covmat[leaf_idx, :,:].det())), FLT_MIN)
        # update parameters
        self.mean = Parameter(new_mean, requires_grad = False)
        self.covmat = Parameter(new_covmat, requires_grad = False) 
        self.covmat_inv = Parameter(new_covmat_inv, requires_grad = False)
        self.factor = Parameter(new_factor, requires_grad = False) 
        if self.is_cuda:
            self.cuda()
        return
    
class Forest(nn.Module):
    # a neural decision forest is an ensemble of neural decision trees
    def __init__(self, n_tree, tree_depth, feature_length, vector_length, use_cuda = True):
        """
        Args:
            n_tree (int): Number of trees in the forest
            tree_depth (int): Depth of the trees
            feature_length (int): number of neurons in the last feature layer
            vector_length (int): length of the mean vector stored at each tree 
                                leaf node (1 for regression)
            use_cuda (boolean): whether to use GPU
        """
        super(Forest, self).__init__()
        self.trees = nn.ModuleList()
        self.n_tree  = n_tree
        self.tree_depth = tree_depth
        self.feature_length = feature_length
        self.vector_length = vector_length
        for _ in range(n_tree):
            tree = Tree(tree_depth, feature_length, vector_length, use_cuda)
            self.trees.append(tree)

    def forward(self, x, save_flag = False):
        predictions = []
        cache = []
        for tree in self.trees:
            if save_flag:
                # record some intermediate results
                mu, cache_tree = tree(x, save_flag = True)
                p = torch.mm(mu, tree.mean)
                cache.append(cache_tree)
            else:    
                p = tree.pred(x)
            predictions.append(p.unsqueeze(2))
        del x
        prediction = torch.cat(predictions,dim=2)
        prediction = torch.sum(prediction, dim=2)/self.n_tree
        if save_flag:
            return prediction, cache
        else:
            return prediction

class NeuralDecisionForest(nn.Module):
    def __init__(self, feature_layer, forest):
        """
        Args:
            feature_layer (FeatureLayer): Defined FeatureLayer to be used 
            forest (Forest):  Defined Forest to be used 
        """
        super(NeuralDecisionForest, self).__init__()
        self.feature_layer = feature_layer
        self.forest = forest
        
    def forward(self, x, debug = False, save_flag = False):
        feats, reg_loss = self.feature_layer(x)
        del x
        if save_flag:
            # return some intermediate results
            pred, cache = self.forest(feats, save_flag = True)
            del feats
            return pred, reg_loss, cache
        else:
            pred = self.forest(feats)
            del feats
            return pred, reg_loss        

class FedDRF(nn.Module):
    def __init__(self,opt):
        """
        Args:
            opt: Initial input options
        """
        super(FedDRF, self).__init__()
        # RNDF consists of two parts:
        #1. a feature extraction model using residual learning
        #2. a neural decision forst
        self.opt = opt
        self.feat_layer = FeatureLayer(opt)
        self.forest = Forest(opt.n_tree, opt.tree_depth, opt.num_output,1, use_cuda=opt.cuda)
        self.model = NeuralDecisionForest(self.feat_layer, self.forest)   
        if self.opt.cuda:
            self.cuda()
        # else:
        #     raise NotImplementedError
        self.optim, self.sche = self.prepare_optim(self.model, opt)
        self.numBadEps = 0
        self.bestRMSE = 1000
        
    def get_model_weights(self):
        param = self.model.state_dict()
        return param
    def update_model(self,update):
        self.model.load_state_dict(update)
        del update
        return self
        
    def prepare_optim(self,model, opt):
        params = [ p for p in model.parameters() if p.requires_grad]
        
        optimizer = torch.optim.Adam(params, lr = opt.lr,amsgrad =False)
        
        # scheduler for automatically decreasing learning rate
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                            mode='min',
                                                            factor=0.1,
                                                            patience=opt.LRsched-1,
                                                            verbose=True,
                                                            min_lr=0.000001)
        return optimizer, scheduler   
    def evaluate_model(self,test):
        """
        Args:
            test (Dataset): Dataset to evaluate the model on 
        Returns:
            RMSE,NRMSE1,CNN_MedAE,NMAE,PCC,CNN_R2 (Float): Metrics of model on dataset
        """
        self.model.eval()
        with torch.no_grad():
            if self.opt.cuda:
                self.model.cuda()
            params = {'batch_size': self.opt.eval_batch_size,
              'shuffle': True,
              'num_workers': self.opt.num_threads}
            validation_generator = DataLoader(test, pin_memory=True,**params)
            MAE = 0
            eval_loss = 0
            PCC = 0
            count = 0
            Ybar = 0
            for batch_idx, batch in enumerate(validation_generator):
                Y = batch['target']
                if self.opt.cuda:
                    Y = Y.cuda()
                Ybar += torch.sum(Y)
                count +=len(Y)
            Ybar /= count
            
            count = 0
            Nom = 0
            Denom = 0
            Nom1 = 0
            Denom1 = 0
            Ypred=[]
            Target =[]
            for batch_idx, batch in enumerate(validation_generator):
                X = batch['data']
                Y = batch['target']
                
                if self.opt.cuda:
                    X = [X[:][0].cuda(),X[:][1].cuda()]
                    Y = Y.cuda()
                Y = Y.view(len(Y), -1)
                # Ybar = torch.mean(Y)
                prediction, reg_loss = self.model(X)  
                Yprediction = prediction.view(len(prediction), -1)
                MAE += torch.sum(torch.abs((Yprediction - Y)))#.sum(dim = 1).sum(dim = 0)
                Nom += torch.sum((Yprediction - Y)**2);    Denom += torch.sum((Ybar - Y)**2)
                eval_loss += F.mse_loss(Yprediction, Y, reduction='sum').data.item()
                Nom1 += torch.sum(torch.abs((Yprediction - Y)));    Denom1 += torch.sum(torch.abs((Ybar - Y)))
                Ypred.extend(Yprediction.cpu().squeeze().tolist())
                Target.extend(Y.cpu().squeeze().tolist())
                count += len(Y)
            del X,Y,validation_generator
            PCC = np.corrcoef(np.asarray(Ypred),np.asarray(Target),rowvar=False)[0][1]
            RMSE =  np.sqrt(eval_loss/count)
            MAE = MAE.data.item()/count
            NRMSE1 = torch.sqrt(Nom/Denom).data.item()
            NMAE = Nom1.data.item()/Denom1.data.item()
            YTrain = np.asarray(Target)
            predTrain = np.asarray(Ypred)
            CNN_NRMSE, CNN_R2 = NRMSE(YTrain, predTrain)
            CNN_MedAE = median_absolute_error(YTrain, predTrain)
            torch.cuda.empty_cache()
            gc.collect()
        return RMSE,NRMSE1,CNN_MedAE,NMAE,PCC,CNN_R2
    def prepare_batches(self,model,train):
    # prepare some feature batches for leaf node distribution update
        with torch.no_grad():
            target_batches = []
            params = {'batch_size': self.opt.leaf_batch_size,
                      'shuffle': True,
                      'num_workers': self.opt.num_threads}
            training_generator = DataLoader(train,pin_memory=True, **params)
            for batch_idx, sample in enumerate(training_generator):
                data = sample['data']                
                target = sample['target']
                target = target.view(len(target), -1)
                if self.opt.cuda:
                    data = [data[:][0].cuda(),data[:][1].cuda()]
                    target = target.cuda()      
                # else:
                #     data = [torch.tensor(list(list(zip(*data))[0])),torch.tensor(list(list(zip(*data))[1]))]                      
                # Get feats
                feats, _ = model.feature_layer(data)
                # release data Tensor to save memory
                del data
                for tree in model.forest.trees:
                    mu = tree(feats)
                    # add the minimal value to prevent some numerical issue
                    mu += FLT_MIN # [batch_size, n_leaf]
                    # store the routing probability for each tree
                    tree.mu_cache.append(mu)
                # release memory
                del feats
                # the update rule will use both the routing probability and the 
                # target values
                target_batches.append(target)   
            del target
        return target_batches

     
    def train1(self,train,val,clientTrain=False):
        """
        Args:
            train (Dataset): training dataset
            val (Dataset): Validation dataet
            clientTrain (Boolean): If client training is being performed
        """
        params = {'batch_size': self.opt.batch_size,
          'shuffle': True,
          'num_workers': self.opt.num_threads}
        losses = []
        for epoch in range(1, self.opt.epochs + 1):
            # At each epoch, train the neural decision forest and update
            # the leaf node distribution separately 
            
            # Train neural decision forest
            # set the model in the training mode
            self.model.train()
            training_generator = DataLoader(train,pin_memory=True, **params)
            print("Epoch %d : Update Neural Weights"%(epoch))
            sys.stdout.flush()
            for idx,sample in enumerate(training_generator):
                
                data = sample['data']
                target = sample['target']
                target = target.view(len(target), -1)
                if self.opt.cuda:
                    data = [data[:][0].cuda(),data[:][1].cuda()]
                    target = target.cuda()       
                if len(target)>1:
                    # erase all computed gradient        
                    self.optim.zero_grad()
                 
                    # forward pass to get prediction
                    prediction, reg_loss = self.model(data)
                    
                    loss = F.mse_loss(prediction, target) + reg_loss
                    
                    # compute gradient in the computational graph
                    loss.backward()
        
                    # update parameters in the model 
                    self.optim.step()

                del data, target,sample
                torch.cuda.empty_cache()
                # Update the leaf node estimation    
                if idx+1 == len(training_generator):
                    logging.info("Epoch %d : Update leaf node prediction"%(epoch))
                    print("Epoch %d : Update leaf node prediction"%(epoch))
                    
                    self.model.eval()
                    target_batches = self.prepare_batches(self.model, train)
                    self.model.train()
                    print('Batches Prepared')
                    for i in range(self.opt.label_iter_time):
                        # prepare features from the last feature layer
                        # some cache is also stored in the forest for leaf node
                        for tree in self.model.forest.trees:
                            tree.update_label_distribution(target_batches)
                    # release cache
                    del target_batches
                    for tree in self.model.forest.trees:   
                        del tree.mu_cache
                        tree.mu_cache = []
                    torch.cuda.empty_cache()      

                if (idx+1 == len(training_generator)) and not clientTrain:
                    self.model.eval()
                    
                    RMSE,NRMSE,CNN_MedAE,NMAE,PCC,CNN_R2 = self.evaluate_model(val)
                    losses.append([RMSE,NRMSE,CNN_MedAE,NMAE,PCC,CNN_R2])
                    if self.opt.cuda:
                        self.model = self.model.cuda()
                    print('Val NRMSE:'+str(NRMSE))
                    sys.stdout.flush()
                    # update learning rate
                    if clientTrain == False:
                        self.sche.step(RMSE)
                    
                    # Eary Stopping
                    if (self.bestRMSE-self.opt.EStol)<=RMSE:
                        self.numBadEps +=1
                    else:
                        self.numBadEps = 0
                        self.bestRMSE = RMSE 
                        
                    if self.numBadEps >= self.opt.ESpat:
                        print('Stopping Early')
                        if ~clientTrain:
                            return self,losses
                    del RMSE,NRMSE,CNN_MedAE,NMAE,PCC,CNN_R2
                
        del training_generator,val,train           
        torch.cuda.empty_cache()
        gc.collect()
        logging.info('Training finished.')
        return self,losses
        

        
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import median_absolute_error
from utils import NRMSE
import sys
    
class ANNCCLE(nn.Module):
    def __init__(self,opt):
        """
        Args:
            train (Dataset): training dataset
            val (Dataset): Validation dataet
            clientTrain (Boolean): If client training is being performed
        """
        super(ANNCCLE, self).__init__()
        self.opt = opt
        self.bestRMSE = 10
        self.dense1 = nn.Linear(2173, 1500)
        self.dense2 = nn.Linear(1500, 1000)
        self.dense3 = nn.Linear(1000, 600)
        self.dense4 = nn.Linear(600, 300)
        self.dense5 = nn.Linear(300, 100)
        self.dense6 = nn.Linear(100, 50)
        self.drop1 = nn.Dropout(p=0.5)
        self.dense7 = nn.Linear(50, 1)
        self.act = nn.ReLU()
        if self.opt.cuda:
            self.cuda()

        params = [ p for p in self.parameters() if p.requires_grad]    
        self.optim  = torch.optim.Adam(params, lr = self.opt.lr,amsgrad =False)
        self.sche = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim ,
                                                            mode='min',
                                                            factor=0.1,
                                                            patience=opt.LRsched,
                                                            verbose=True,
                                                            min_lr=0.000001)        
    def forward(self,x):
        x = self.act(self.dense1(x))
        x = self.act(self.dense2(x))
        x = self.act(self.dense3(x))
        x = self.act(self.dense4(x))
        x = self.act(self.dense5(x))
        x = self.drop1(self.act(self.dense6(x)))
        x = self.dense7(x)
        reg_loss = 0
        return x,reg_loss  
    def get_model_weights(self):
        param = self.state_dict()
        return param
    def update_model(self,update):
        self.load_state_dict(update)
        del update
        return self
    def train1(self,train1,val,clientTrain=False):
        """
        Args:
            train (Dataset): training dataset
            val (Dataset): Validation dataet
            clientTrain (Boolean): If client training is being performed
        """     
        params = {'batch_size': self.opt.batch_size,
          'shuffle': True,
          'num_workers': self.opt.num_threads}
        training_generator = DataLoader(train1,pin_memory=True, **params)
        losses = []
        for epoch in range(1, self.opt.epochs + 1):     
            # Train neural decision forest
            # set the model in the training mode
            self.train()
            print("Epoch %d : Update Neural Weights"%(epoch))
            for idx,sample in enumerate(training_generator):
                data = sample['data']
                target = sample['target']
                target = target.view(len(target), -1)
                if self.opt.cuda:
                    data = data.cuda()
                    target = target.cuda()       
                if len(target)>1:
                    # erase all computed gradient        
                    self.optim.zero_grad()
                 
                    # forward pass to get prediction
                    prediction, reg_loss = self(data)
                    loss = F.mse_loss(prediction, target)
    
                    # compute gradient in the computational graph
                    loss.backward()
        
                    # update parameters in the model 
                    self.optim.step()
                # torch.cuda.empty_cache()
                del data, target
                                # if self.opt.eval and idx % self.opt.eval_every == 0:
                if (idx+1 == len(training_generator)) and not clientTrain:
                #     # evaluate model
                    self.eval()
                    
                    RMSE, NRMSE,MAE,NMAE,PCC,R2 = self.evaluate_model(val)
                    losses.append([RMSE, NRMSE,MAE,NMAE,PCC,R2])
                    if self.opt.cuda:
                        self = self.cuda()
                    print('Val NRMSE:'+str(NRMSE))
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
                    del RMSE, NRMSE,MAE,NMAE,PCC,R2  
                sys.stdout.flush()
        torch.cuda.empty_cache()

        return self,losses
    def evaluate_model(self,test):
        """
        Args:
            test (Dataset): Dataset to evaluate the model on 
        Returns:
            RMSE,NRMSE1,CNN_MedAE,NMAE,PCC,CNN_R2 (Float): Metrics of model on dataset
        """
        self.eval()
        with torch.no_grad():
            if self.opt.cuda:
                self.cuda()
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
                    X = X.cuda()
                    Y = Y.cuda()
                Y = Y.view(len(Y), -1)
                prediction, reg_loss = self(X)  
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
        return RMSE,NRMSE1,CNN_MedAE,NMAE,PCC,CNN_R2
    
class DRF_CNNCCLE(nn.Module):
    def __init__(self,opt):
        """
        Args:
            train (Dataset): training dataset
            val (Dataset): Validation dataet
            clientTrain (Boolean): If client training is being performed
        """
        super(DRF_CNNCCLE, self).__init__()
        nb_filters = 8
        nb_conv = 5
        self.cellConv1 = nn.Conv2d(1,nb_filters,nb_conv,stride=1)
        self.cellbn1 = nn.BatchNorm2d(nb_filters)
        self.cellConv2 = nn.Conv2d(nb_filters,nb_filters*2,nb_conv,stride=1)
        self.cellbn2 = nn.BatchNorm2d(nb_filters*2)
        self.cellConv3 = nn.Conv2d(nb_filters*2,nb_filters*3,nb_conv,stride=1)
        self.cellbn3 = nn.BatchNorm2d(nb_filters*3)
        
        self.drugConv1 = nn.Conv2d(1,nb_filters,nb_conv,stride=1)
        self.drugbn1 = nn.BatchNorm2d(nb_filters)
        self.drugConv2 = nn.Conv2d(nb_filters,nb_filters*2,nb_conv,stride=1)
        self.drugbn2 = nn.BatchNorm2d(nb_filters*2)
        self.drugConv3 = nn.Conv2d(nb_filters*2,nb_filters*3,nb_conv,stride=1)
        self.drugbn3 = nn.BatchNorm2d(nb_filters*3)
        
        self.xConv1 = nn.Conv1d(1,nb_filters*2,(nb_conv),stride=(3))
        self.xbn1 = nn.BatchNorm1d(nb_filters*2)
        self.xConv2 = nn.Conv1d(nb_filters*2,nb_filters*2,(nb_conv),stride=(3))
        self.xbn2 = nn.BatchNorm1d(nb_filters*2)
        self.xConv3 = nn.Conv1d(nb_filters*2,nb_filters*2,(nb_conv),stride=(3))
        self.xbn3 = nn.BatchNorm1d(nb_filters*2)
        self.drop1 = nn.Dropout(p=0.3)
        
        self.dense1 = nn.Linear(13136, 4000)
        self.xbn4 = nn.BatchNorm1d(4000)
        self.drop2 = nn.Dropout(p=0.3)
        self.dense2 = nn.Linear(4000, 1000)
        self.xbn5 = nn.BatchNorm1d(1000)
        self.drop3 = nn.Dropout(p=0.3)
        self.dense4 = nn.Linear(1000, opt.num_output)
        self.act = nn.ReLU(inplace=True)

        
    def forward(self,x):
        cell = x[0].unsqueeze(1)
        drug = x[1].unsqueeze(1)
        cell = self.act(self.cellbn1(self.cellConv1(cell)))
        cell = self.act(self.cellbn2(self.cellConv2(cell)))
        cell = self.act(self.cellbn3(self.cellConv3(cell)))
        cell = cell.view(-1, 11616)
        
        drug = self.act(self.drugbn1(self.drugConv1(drug)))
        drug = self.act(self.drugbn2(self.drugConv2(drug)))
        drug = self.act(self.drugbn3(self.drugConv3(drug)))
        drug = drug.view(-1, 10584)
        
        x = torch.cat((drug,cell),dim=1)
        x = x.unsqueeze(1)
        x = self.act(self.xbn1(self.xConv1(x)))
        x = self.act(self.xbn2(self.xConv2(x)))
        x = self.act(self.xbn3(self.xConv3(x)))
        x = self.drop1(x)
        x = x.view(-1, 13136)
        x = self.drop2(self.act(self.xbn4(self.dense1(x))))
        x = self.drop3(self.act(self.xbn5(self.dense2(x))))
        x = self.dense4(x)
        
        reg_loss = 0
        return x,reg_loss
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        print(num_features)
        return num_features   

from torch.utils.data import  DataLoader  
import numpy as np  
class CNNCCLE(nn.Module):
    def __init__(self,opt):
        """
        Args:
            train (Dataset): training dataset
            val (Dataset): Validation dataet
            clientTrain (Boolean): If client training is being performed
        """
        self.opt = opt
        super(CNNCCLE, self).__init__()
        nb_filters = 8
        nb_conv = 5
        self.cellConv1 = nn.Conv2d(1,nb_filters,nb_conv,stride=1)
        self.cellbn1 = nn.BatchNorm2d(nb_filters,momentum=0.01,eps=0.001)
        self.cellConv2 = nn.Conv2d(nb_filters,nb_filters*2,nb_conv,stride=1)
        self.cellbn2 = nn.BatchNorm2d(nb_filters*2,momentum=0.01,eps=0.001)
        self.cellConv3 = nn.Conv2d(nb_filters*2,nb_filters*3,nb_conv,stride=1)
        self.cellbn3 = nn.BatchNorm2d(nb_filters*3,momentum=0.01,eps=0.001)
        
        self.drugConv1 = nn.Conv2d(1,nb_filters,nb_conv,stride=1)
        self.drugbn1 = nn.BatchNorm2d(nb_filters,momentum=0.01,eps=0.001)
        self.drugConv2 = nn.Conv2d(nb_filters,nb_filters*2,nb_conv,stride=1)
        self.drugbn2 = nn.BatchNorm2d(nb_filters*2,momentum=0.01,eps=0.001)
        self.drugConv3 = nn.Conv2d(nb_filters*2,nb_filters*3,nb_conv,stride=1)
        self.drugbn3 = nn.BatchNorm2d(nb_filters*3,momentum=0.01,eps=0.001)
        
        self.xConv1 = nn.Conv1d(1,nb_filters*2,(nb_conv),stride=(3))
        self.xbn1 = nn.BatchNorm1d(nb_filters*2,momentum=0.01,eps=0.001)
        self.xConv2 = nn.Conv1d(nb_filters*2,nb_filters*2,(nb_conv),stride=(3))
        self.xbn2 = nn.BatchNorm1d(nb_filters*2,momentum=0.01,eps=0.001)
        self.xConv3 = nn.Conv1d(nb_filters*2,nb_filters*2,(nb_conv),stride=(3))
        self.xbn3 = nn.BatchNorm1d(nb_filters*2,momentum=0.01,eps=0.001)
        self.drop1 = nn.Dropout(p=0.3)
        
        self.dense1 = nn.Linear(13136, 128)
        self.xbn4 = nn.BatchNorm1d(128,momentum=0.01,eps=0.001)
        self.drop2 = nn.Dropout(p=0.3)
        self.dense2 = nn.Linear(128, 1)
        if self.opt.cuda:
            self.cuda()
        self.bestRMSE = 10
        params = [ p for p in self.parameters() if p.requires_grad]    
        self.optim  = torch.optim.Adam(params, lr = self.opt.lr,amsgrad =False)
        self.sche = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim ,
                                                            mode='min',
                                                            factor=0.1,
                                                            patience=5,
                                                            verbose=True,
                                                            min_lr=0.000001)        
    def get_model_weights(self):
        param = self.state_dict()
        return param
    def update_model(self,update):
        self.load_state_dict(update)
        del update
        return self    
    def forward(self,x):
        cell = x[0].unsqueeze(1)
        drug = x[1].unsqueeze(1)
        
        drug = torch.relu(self.drugbn1(self.drugConv1(drug)))
        drug = torch.relu(self.drugbn2(self.drugConv2(drug)))
        drug = torch.relu(self.drugbn3(self.drugConv3(drug)))
        drug = drug.view(-1, 10584)
        
        cell = torch.relu(self.cellbn1(self.cellConv1(cell)))
        cell = torch.relu(self.cellbn2(self.cellConv2(cell)))
        cell = torch.relu(self.cellbn3(self.cellConv3(cell)))
        cell = cell.view(-1, 11616)
        
        
        x = torch.cat((drug,cell),dim=1)
        x = x.unsqueeze(1)
        x = torch.relu(self.xbn1(self.xConv1(x)))
        x = torch.relu(self.xbn2(self.xConv2(x)))
        x = torch.relu(self.xbn3(self.xConv3(x)))
        x = self.drop1(x)
        x = x.view(-1, 13136)
        x = self.drop2(torch.relu(self.xbn4(self.dense1(x))))
        x = self.dense2(x)
        
        reg_loss = 0
        return x,reg_loss
    def train1(self,train1,val,clientTrain=False):
        """
        Args:
            train (Dataset): training dataset
            val (Dataset): Validation dataet
            clientTrain (Boolean): If client training is being performed
        """     
        params = {'batch_size': self.opt.batch_size,
          'shuffle': True,
          'num_workers': self.opt.num_threads}
        training_generator = DataLoader(train1,pin_memory=True, **params)
        losses = []
        for epoch in range(1, self.opt.epochs + 1):
            # X,Y = shuffle(X,Y,random_state= self.opt.seed)
            # At each epoch, train the neural decision forest and update
            # the leaf node distribution separately 
            
            # Train neural decision forest
            # set the model in the training mode
            self.train()
            print("Epoch %d : Update Neural Weights"%(epoch))
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
                    prediction, reg_loss = self(data)
                    loss = F.mse_loss(prediction, target)
    
                    # compute gradient in the computational graph
                    loss.backward()
        
                    # update parameters in the model 
                    self.optim.step()

                del data, target

                if (idx+1 == len(training_generator)) and not clientTrain:
                #     # evaluate model
                    self.eval()
                    RMSE, NRMSE,MAE,NMAE,PCC,R2 = self.evaluate_model(val)
                    losses.append([RMSE, NRMSE,MAE,NMAE,PCC,R2])
                    if self.opt.cuda:
                        self = self.cuda()
                    print('Val NRMSE:'+str(NRMSE))
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
                    del RMSE, NRMSE,MAE,NMAE,PCC,R2     
                sys.stdout.flush()
        torch.cuda.empty_cache()

        return self,losses
    def evaluate_model(self,test):
        """
        Args:
            test (Dataset): Dataset to evaluate the model on 
        Returns:
            RMSE,NRMSE1,CNN_MedAE,NMAE,PCC,CNN_R2 (Float): Metrics of model on dataset
        """
        self.eval()
        with torch.no_grad():
            if self.opt.cuda:
                self.cuda()
            else:
                self.cpu()
                # self.model.feature_layer.cuda()
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
                prediction, reg_loss = self(X)  
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
        return RMSE,NRMSE1,CNN_MedAE,NMAE,PCC,CNN_R2
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        print(num_features)
        return num_features  

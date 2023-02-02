# -*- coding: utf-8 -*-
"""
Code for running federated simulation
The code parses the input arguments and creates a server based on them. Then 
the code either runs federated training or evaluation based on input parameters
"""

# from Server import Server
from Server import Server
import utils
import torch
import time

if __name__ == '__main__':
    #Parameters  
    n = 12
    fracOfClientsPerRound = 1
    fracTestData = 0.2
    
    opt = utils.parse_arg()
    fed_epochs = opt.epochs
    opt.save_dir = opt.save_dir+'seed'+str(opt.randomSeed)+'iid'+str(opt.iid)
    print('Number of clients: '+str(n))
    print('IID Parameter: '+str(opt.iid))
    #Init server with defined parameters
    ser = Server(n,fracOfClientsPerRound,fracTestData,opt)
    #Early Stopping Params
    bestRMSE = 10
    numBadEps = 0
    LRdelay = 0
    if opt.train:
        #Run the FL Process
        print('Training Federated Model')
        t = time.time()
        for rnd in range(fed_epochs):
            print()
            print('Training Rounnd: ',rnd+1)
            LRdelay += 1
            #for each round, call run_one_round and save the outputs
            round_output = ser.run_one_round()
            #Check for early stopping and learning rate reduction
            if (bestRMSE-opt.EStol)<=round_output[1]:
                numBadEps +=1
            else:
                numBadEps = 0
                bestRMSE = round_output[1] 
                
            if (numBadEps >= opt.LRsched) & (LRdelay >= opt.LRsched):
                print('###Reducing LR####')
                LRdelay = 0
                for c in ser.clients:
                    c.reduce_lr()
                print(c.model.optim.param_groups[0]['lr'])
            if numBadEps >= opt.ESpat:
                print('###Stopping Early###')
                break
        elapsed = time.time() - t
        #Save Losses and model
        utils.save_losses_time(round_output[4],elapsed,opt,'FedValClients'+str(n)+'IID'+str(opt.iid))
        utils.save_losses_time(round_output[5],elapsed,opt,'FedValGlobalModel'+str(n)+'IID'+str(opt.iid))
        utils.save_model(round_output[3], opt,'Fed'+str(n)+'IID'+str(opt.iid))
        print('Finished Training')
    elif opt.eval:
        #Load Federated model based on input arguments
        save_dir = utils.get_save_dir(opt,'Fed'+str(n)+'IID'+str(opt.iid))
        print(opt.cuda)  
        if opt.cuda:
            fedModel = torch.load(save_dir,map_location=torch.device('cuda'))
            fedModel.opt = opt
            ser.opt = opt
        else:
            fedModel = torch.load(save_dir,map_location=torch.device('cpu'))   
        print('Federated Eval')     
        #Test loaded Federated model on current seeds datasets
        FedTestResults = ser.fed_eval(fedModel)
    del ser





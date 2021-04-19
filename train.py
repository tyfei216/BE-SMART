import torch 
import log 
import dataset 
import functions 
import models
import argparse
import os
import pickle
import time
import torch.nn as nn
import numpy as np
import bayesianNetwork
import configparser 
device = None

def writeConfig(args, bestEpoch, path):
    config = configparser.ConfigParser() 

    splitname = args.split 
    if splitname == None:
        splitname = args.splitSave
        
    config["meta"] = {
        "dataset":args.ds, 
        "predictBase":args.baseIndex, 
        "editBase":args.editbase,
        "bN":os.path.join(args.checkpoints, "bayesianNetwork.pkl"),
        "window":args.evalpositions, 
        "split":splitname
    }
    config["train"] = {
        "epoches":args.epoch, 
        "best":bestEpoch
    }
    with open(path, "wb") as f:
        config.write(f)  

def Args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-gpu", action="store_true", help="whether to use gpu to train")
    parser.add_argument("-models", action="extend", nargs="+", type=int, default=None, help="input dimensions of every base models in the full model")
    parser.add_argument("-log", default=".\\log\\", type=str, help="the path to the logfile folder")
    parser.add_argument("-checkpoints", type=str, default="./checkpoints/", help="the path to the folder for saving models")
    parser.add_argument("-ds", required=True, type=str, help="path to the dataset")
    parser.add_argument("-editBase", default=3, type=int, help="which base to edit")
    parser.add_argument("-split", type=str, default=None, help="path to the split file of the dataset")
    parser.add_argument("-splitSave", type=str, default=None, help="name of the splitfile name to save. Used only when split file is not given")
    parser.add_argument("-savefreq", default=-1, type=int, help="saving the model every ? epoches")
    parser.add_argument("-epoch", default=100, type=int, help="number of epoches to train")
    parser.add_argument("-result", default=".\\results\\", type=str, help="path to save the results")
    parser.add_argument("-evalpositions", action="extend", nargs="+", type=int, default=None, help="the position for pearson calculation, intergers[0-20]")
    parser.add_argument("-lr", type=float, default=0.00002, help="learning rate")
    parser.add_argument("-weight_decay", type=float, default=0.0005, help="weight decay of parameters")
    parser.add_argument("-dropout", type=float, default=0.3, help="dropout rates")
    parser.add_argument("-baseIndex", type=int, default=2, help="which base to predict (T:1, G:2)")
    args = parser.parse_args()
    if not os.path.exists(args.log):
        os.makedirs(args.log)
    logname = log.init_logs(args.log) 
    if args.evalpositions == None:
        args.evalpositions = [3,4,5,6,7]
    if args.models == None:
        args.models = [3,3,4,4,5,5]#[3,3,3,4,4,4,5,5,5] 

    if args.gpu:
        if not torch.cuda.is_available():
            log.error("CUDA is not available! Try without the -gpu option to run on cpu")
            log.error("Please install the correct version of pytorch to enable gpu")
            exit()
    global device
    if args.gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if not os.path.exists(args.ds):
        log.error("dataset doesn't exist")
        exit()
        
    dsname = os.path.basename(args.ds)
    dirName = args.checkpoints + logname+dsname[:dsname.find(".")]+"\\"

    if args.split == None:
        if not os.path.exists("./split"):
            log.info("creating split file directory")
            os.mkdir("./split")
        if args.splitSave == None:
            args.splitSave = os.path.join("./split",logname+"_"+dsname[:dsname.find(".")]+".pkl")

    if not os.path.exists(dirName):
        os.makedirs(dirName)
    args.checkpoints = dirName
    log.info("models will be saved under "+dirName)

    dirName = args.result + logname + dsname[:dsname.find(".")]+"\\"
    # if not os.path.exists(dirName):
    #     os.makedirs(dirName)
    args.result = dirName
    log.info("test results will be saved under "+dirName)

    log.info("model uses "+str(args.evalpositions)+" for evaluation")
    return args

def GetDataset(args, path, path2=None):
    
    log.info("reading dataset " + path)
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    if path2 != None:
        log.info("using split file " + path2)
        with open(path2, "rb") as f:
            indices = pickle.load(f)
    else:
        indices = None
     
    indel = np.array(data['indel'], dtype=np.float)/np.array(data['cnts'], dtype=np.float)
    ds = dataset.BaseEditingDataset(data['mapping'], data['seq'], indel, data['allp'], editBase = 3, rawSequence=False)

    dsTrain, dsValid, dsTest, indices = dataset.SplitDataset(ds, split=indices, savepath=args.splitSave)
    log.info("finish dataset construction")

    log.info("initializing BayesianNetwork")
    bN = bayesianNetwork.BayesianNetwork(path, [13,14,15,16,17,18,19,20], indices=indices[:7*len(indices)//10])
    log.info("finish BayesianNetwork initialization")

    return dsTrain, dsValid, dsTest, bN

def main():
    args = Args()
    dsTrain, dsValid, dsTest, bN = GetDataset(args, args.ds, path2=args.split)

    log.info("saving bayesianNetwork")
    with open(os.path.join(args.checkpoints, "bayesianNetwork.pkl"), "wb") as f:
        pickle.dump(bN, f)

    log.info("build model")
    model = models.FullModel(lengthlist=args.models, dropout=args.dropout)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    bestval = -2.0
    bestepoch = -1

    cri = nn.MSELoss()
    checkpoint_path = os.path.join(args.checkpoints, '{epoch}-{net}.pth')
    log.info("start training\n-------------------")
    
    for i in range(args.epoch):
        totalLoss = functions.trainonce(model, dsTrain, optim, cri, device, args.baseIndex)
        log.info("epoch " + str(i)+": Total Loss: "+str(totalLoss))

        # pre, tru = functions.test(model, dsTrain, args.baseIndex)
        # res1, res2 = functions.eval(pre, tru, args.evalpositions)
        #indelpearson = functions.CalculatePearson(indelpre, indeltruth)
        log.info("results on training set")
        pre, tru = functions.test(model, dsTrain, args.baseIndex)
        res1, res2 = functions.eval(pre, tru, args.evalpositions)
        #indelpearson = functions.CalculatePearson(indelpre, indeltruth)
        log.info("training results: pearson "+str(res1)+" RMSE "+str(res2))
        #log.info("training indel results "+str(indelpearson))

        log.info("results on testing set")
        pre, tru = functions.test(model, dsTest, args.baseIndex)
        res1, res2 = functions.eval(pre, tru, args.evalpositions)
        log.info("testing results: pearson "+str(res1)+" RMSE "+str(res2))

        log.info("testing on validation set")
        pre, tru = functions.test(model, dsValid, args.baseIndex)
        res1, res2 = functions.eval(pre, tru, args.evalpositions)
        #indelpearson = functions.CalculatePearson(indelpre, indeltruth)



        log.info("validation results: pearson "+str(res1)+" RMSE "+str(res2))
        #log.info("validation indel results "+str(indelpearson))
        if res1 > bestval:
            bestval = res1 
            bestepoch = i
            log.info("Best model, saving....")
            torch.save(model, checkpoint_path.format(epoch=i, net="best"))

        if args.savefreq > 0:
            if (i+1) % args.savefreq == 0:
                log.info("regularly saving model...")
                torch.save(model, checkpoint_path.format(epoch=i, net="regular"))
        log.info("finish epoch "+str(i)+"\n-------------------")

    log.info("finished training, saving final model")
    torch.save(model, checkpoint_path.format(epoch=args.epoch, net="final"))
    log.info("loading best model for evaluation")
    model = torch.load(checkpoint_path.format(epoch=bestepoch, net="best"))
    
    log.info("finish loading model, testing and saving results")
    if not os.path.exists(args.result):
        os.makedirs(args.result)
    res1, res2 = functions.CalculateAllResults(model, dsTest, args.baseIndex, args.result, args.evalpositions)
    log.info("test results: pearson "+str(res1)+" RMSE "+str(res2))
    writeConfig(args, checkpoint_path.format(epoch=bestepoch, net="best"), os.path.join(args.checkpoints, "config.ini"))
    log.info("finshed!")

if __name__ == '__main__':
    main()
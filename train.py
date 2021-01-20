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

device = None

def Args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-gpu", action="store_true", help="whether to use gpu to train")
    parser.add_argument("-models", action="extend", nargs="+", type=int, default=None, help="input dimensions of every base models in the full model")
    parser.add_argument("-log", default=".\\log\\", type=str, help="the path to the logfile folder")
    parser.add_argument("-checkpoints", type=str, default="./checkpoints/", help="the path to the folder for saving models")
    parser.add_argument("-ds", required=True, type=str, help="path to the dataset")
    parser.add_argument("-savefreq", default=-1, type=int, help="saving the model every ? epoches")
    parser.add_argument("-epoch", default=400, type=int, help="number of epoches to train")
    parser.add_argument("-result", default=".\\results\\", type=str, help="path to save the results")
    parser.add_argument("-evalpositions", action="extend", nargs="+", type=int, default=None, help="the position for pearson calculation, intergers[0-20]")
    parser.add_argument("-lr", type=float, default=0.00002, help="learning rate")
    parser.add_argument("-weight_decay", type=float, default=0.0005, help="weight decay of parameters")
    parser.add_argument("-dropout", type=float, default=0.3, help="dropout rates")
    parser.add_argument("-baseIndex", type=int, default=2, help="which base to predict (T:1, G:2)")
    parser.add_argument("-tensorboard", action="store_true", help="whether to use tensorboard to record training information")
    parser.add_argument("-stop", type=int, default=30, help="the number of epoches to stopping training without better results")
    args = parser.parse_args()
    if not os.path.exists(args.log):
        os.makedirs(args.log)

    dsname = os.path.basename(args.ds)
    dsname = dsname[:dsname.find(".")]
    
    logname = log.init_logs(logPath=args.log, label=dsname) 
    if args.evalpositions == None:
        args.evalpositions = [3,4,5,6,7]
    if args.models == None:
        args.models = [3,3,3,4,4,4,5,5,5]
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
    dirName = os.path.join(args.checkpoints, logname+dsname[:dsname.find(".")])
    if not os.path.exists(dirName):
        os.makedirs(dirName)
    args.checkpoints = dirName
    log.info("models will be saved under "+dirName)

    if args.tensorboard:
        dirName = dirName = os.path.join(".\\tensorboard", logname+dsname[:dsname.find(".")])
        # if not os.path.exists(dirName):
        #     os.makedirs(dirName)
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=dirName)
        args.writer = writer
        log.info("using tensorboard, data stored under "+dirName)
    else:
        args.writer = None


    dirName = args.result + logname+dsname[:dsname.find(".")]+"\\"
    # if not os.path.exists(dirName):
    #     os.makedirs(dirName)
    args.result = dirName
    log.info("test results will be saved under "+dirName)

    log.info("model uses "+str(args.evalpositions)+" for evaluation")
    return args

def GetDataset(path):
    log.info("reading dataset " + path)
    with open(path, "rb") as f:
        data = pickle.load(f)
    ds = dataset.BaseEditingDataset(data['mapping'], data['seq'], data["indel"], editBase = 3, rawSequence=False)
    dsTrain, dsValid, dsTest = dataset.SplitDataset(ds)
    log.info("finish dataset construction")
    return dsTrain, dsValid, dsTest


def main():
    args = Args()
    dsTrain, dsValid, dsTest = GetDataset(args.ds)

    log.info("build model")
    model = models.FullModel(lengthlist=args.models, dropout=args.dropout)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    bestval = -2.0
    bestepoch = -1

    bestindel = -2.0 
    bestindelepoch = -1

    cri = nn.MSELoss()
    checkpoint_path = os.path.join(args.checkpoints, '{epoch}-{net}.pth')
    log.info("start training\n-------------------")
    
    cnt = 0

    for i in range(args.epoch):

        cnt += 1

        totalLoss = functions.trainonce(model, dsTrain, optim, cri, device, args.baseIndex)
        log.info("epoch " + str(i)+": Total Loss: "+str(totalLoss))

        log.info("results on training set")
        pre, tru, indelpre, indeltruth = functions.test(model, dsTrain, args.baseIndex, device)
        res1, res2 = functions.eval(pre, tru, args.evalpositions)
        indelpearson, indelRMSE = functions.CalculatePearson(indelpre, indeltruth)
        log.info("training results: pearson "+str(res1)+" RMSE "+str(res2))
        log.info("training indel results: pearson "+str(indelpearson)+" RMSE "+str(indelRMSE))

        if args.tensorboard:
            args.writer.add_scalar("Loss/train", totalLoss, i)
            args.writer.add_scalar("Efficiency/train/pearson", res1, i)
            args.writer.add_scalar("Efficiency/train/RMSE", res2, i)
            args.writer.add_scalar("Indel/train/pearson", indelpearson, i)
            args.writer.add_scalar("Indel/train/RMSE", indelRMSE, i)

        log.info("testing on validation set")
        totalLoss = functions.CalculateLoss(model, dsTest, cri, device, args.baseIndex)
        pre, tru, indelpre, indeltruth = functions.test(model, dsValid, args.baseIndex, device)
        res1, res2 = functions.eval(pre, tru, args.evalpositions)
        indelpearson, indelRMSE = functions.CalculatePearson(indelpre, indeltruth)

        log.info("validation results: pearson "+str(res1)+" RMSE "+str(res2))
        log.info("validation indel results: pearson "+str(indelpearson)+" RMSE "+str(indelRMSE))

        if args.tensorboard:
            args.writer.add_scalar("Loss/test", totalLoss, i)
            args.writer.add_scalar("Efficiency/test/pearson", res1, i)
            args.writer.add_scalar("Efficiency/test/RMSE", res2, i)
            args.writer.add_scalar("Indel/test/pearson", indelpearson, i)
            args.writer.add_scalar("Indel/test/RMSE", indelRMSE, i)

        if indelpearson > bestindel:
            cnt = 0
            bestindel = indelpearson
            bestindelepoch = i 
            log.info("Best model for indel, saving...")
            torch.save(model, checkpoint_path.format(epoch=i, net="bestindel"))


        if res1 > bestval:
            cnt = 0
            bestval = res1 
            bestepoch = i
            log.info("Best model for efficiency, saving....")
            torch.save(model, checkpoint_path.format(epoch=i, net="besteffi"))

        if args.savefreq > 0:
            if (i+1) % args.savefreq == 0:
                log.info("regularly saving model...")
                torch.save(model, checkpoint_path.format(epoch=i, net="regular"))
        log.info("finish epoch "+str(i)+"\n-------------------")

        if cnt > args.stop:
            break

    log.info("finished training, saving final model")
    torch.save(model, checkpoint_path.format(epoch=args.epoch, net="final"))
    log.info("loading best model for efficiency evaluation")
    model = torch.load(checkpoint_path.format(epoch=bestepoch, net="besteffi"))
    
    log.info("finish loading model, testing and saving results")
    if not os.path.exists(args.result):
        os.makedirs(args.result)
    res1, res2 = functions.CalculateAllResults(model, dsTest, args.baseIndex, args.result, args.evalpositions, device)
    log.info("test efficiency results: pearson "+str(res1)+" RMSE "+str(res2))

    log.info("loading best model for indel")
    model = torch.load(checkpoint_path.format(epoch=bestindelepoch, net="bestindel")) 
    pre, tru, indelpre, indeltruth = functions.test(model, dsTest, args.baseIndex, device)
    indelpearson, indelRMSE = functions.CalculatePearson(indelpre, indeltruth)
    log.info("test indel results: pearson" + str(indelpearson)+" RMSE "+str(indelRMSE))
    log.info("finshed!")

if __name__ == '__main__':
    main()
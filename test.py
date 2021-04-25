import torch 
import log 
import argparse
import os
import pickle
import models
import time
import functions
import numpy as np 
import dataset
import bayesianNetwork

def Args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-baseIndex", type=int, default=2, help="which base to predict (T:1, G:2)")
    parser.add_argument("-config", default=None, type=str, help="path to a training config file")
    parser.add_argument("-model", default=None, type=str, help="path to the model")
    parser.add_argument("-ds", default=None, type=str, help="path to the test set")
    parser.add_argument("-result", default=".\\results\\", type=str, help="path to save the results")
    parser.add_argument("-seq", default=None, type=str, help="the input sequence")
    parser.add_argument("-predictBase", type=str, default="C", help="which base to predict (default \"C\")")
    parser.add_argument("-editres", type=str, default="G", help="edit result of the base")
    parser.add_argument("-bN", default=None, type=str, help="the base for the bayesianNetwork")
    parser.add_argument("-evalpositions", action="extend", nargs="+", type=int, default=None, help="the position for pearson calculation, intergers[0-20]")
    parser.add_argument("-split", type=str, default=None, help="path to the split file of the dataset")
    parser.add_argument("-splitSave", type=str, default=None, help="name of the splitfile name to save. Used only when split file is not given")
    args = parser.parse_args()

    if args.config == None and args.model == None:
        print("model not given")
        exit()

    if not os.path.exists(args.result):
        os.makedirs(args.result)
    
    return args

# you may want to rewrite this function to read your own data
def ReadFromDS(path):
    with open(path, "rb") as f:
        seqs = pickle.load(f)

    return seqs

def GetDataset(args, path, path2=None):
    
    #log.info("reading dataset " + path)
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    if path2 != None:
        #log.info("using split file " + path2)
        with open(path2, "rb") as f:
            indices = pickle.load(f)
    else:
        indices = None
    
    indel = np.array(data['indel'], dtype=np.float)/np.array(data['cnts'], dtype=np.float)
    ds = dataset.BaseEditingDataset(data['mapping'], data['seq'], indel, data['allp'], data['proportion'], editBase = 3, rawSequence=False)

    #log.info("length of the dataset "+str(len(data["indel"])))

    dsTrain, dsValid, dsTest, indices = dataset.SplitDataset(ds, split=indices, savepath=args.splitSave)
    #log.info("finish dataset construction")

    #log.info("initializing BayesianNetwork")
    if "score" not in data.keys():
        bN = bayesianNetwork.BayesianNetwork(path, [12,13,14,15,16,17,18,19,20], indices=indices[:7*len(indices)//10])
        data["score"] = bN.score 
        with open(path, "wb") as f:
            pickle.dump(data, f) 
    else:
        score = data["score"] 
        bN = bayesianNetwork.BayesianNetwork(path, [12,13,14,15,16,17,18,19,20], score=score, indices=indices[:7*len(indices)//10])
    #log.info("finish BayesianNetwork initialization")

    return dsTrain, dsValid, dsTest, bN
from sklearn.metrics import mean_squared_error
from math import sqrt
def eval(pre1, real1, positions):

    pre = [] 
    real = []
    allres1 = [] 
    allres2 = []
    for i in positions:
        pre.extend(pre1[i])
        real.extend(real1[i])
        allres2.append(sqrt(mean_squared_error(pre1[i], real1[i])))
        allres1.append(np.corrcoef(np.asarray(pre1[i]), np.asarray(real1[i]))[0, 1])
        print(np.array(pre1[i]) - np.array(real1[i]), real1[i])
    print(allres1)
    print(allres2)

    res1 = np.corrcoef(np.asarray(pre), np.asarray(real))[0, 1]
    res2 = sqrt(mean_squared_error(pre, real))
    return (res1, res2)

def main():
    args = Args()
    dsTrain, dsValid, dsTest, bN = GetDataset(args, args.ds, path2=args.split)
    if args.model != None:
        model = torch.load(args.model, map_location=torch.device('cpu'))
    
    pre, tru = functions.test(model, dsTest, args.baseIndex)
    res1, res2 = eval(pre, tru, [3,4,5,6,7])
    print(res1, res2) 
    exit()
    name = os.path.basename(args.model)
    f = open(args.result+name[:name.find(".")]+".tsv", "w")
    f.write("seq\t\pos\t\pre\n")
    

    if args.bN != None:
        with open(args.bN, "rb") as f:
            bayesianNetwork = pickle.load(f)

    if args.seq != None:
        res = functions.CalculateOneSeq(model, args.seq)
        cpos = []
        for i in range(10,30):
            if args.seq[i] == args.predictBase:
                cpos.append(i)
                f.write(args.seq+"\t")
                f.write(str(i-10)+"\t")
                f.write(str(round(res[i-10], 8))+"\n")
        bn = bayesianNetwork.fit(cpos, res, args.seq, args.editres, 10)
        bn.printres()

    if args.ds != None:
        seqs = ReadFromDS(args.ds)
        for seq in seqs:
            res = functions.CalculateOneSeq(model, seq)
            for i in range(10,30):
                if seq[i] == args.predictBase:
                    f.write(args.seq+"\t")
                    f.write(str(i-10)+"\t")
                    f.write(str(round(res[i-10], 8))+"\n")

if __name__=="__main__":
    main()

        

from bayesianNetwork import BayesianNetwork
import pickle 
import configparser
import argparse
import numpy as np
import os
import models
import torch
import functions

mapping = {0:"A", 1:"T", 2:"G", 3:"C"}

def Args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", default=None, type=str, help="path to the model") 
    parser.add_argument("-seq", default=None, type=str, help="the input sequence") 
    parser.add_argument("-start", default=None, type=int, help="proportion start")
    parser.add_argument("-end", default=None, type=int, help="proportion end")
    #parser.add_argument("-batch", default=None, type=str, help="path to the batch file")
    args = parser.parse_args()

    return args 



def main():
    args = Args() 
    
    config = configparser.ConfigParser() 
    config.read(os.path.join(args.model, "config.ini"))

    #with open("./YE1-FNLS-BE3/bayesianNetwork.pkl", "rb") as f:
    # with open(os.path.join(args.model, "bayesianNetwork.pkl"), "rb") as f:
    #     bN = pickle.load(f)

    score = np.load(os.path.join(args.model, "score.npy"))
    bN = BayesianNetwork(score=score, give=True)
    #print(bN.positions)
    predictBase = config.getint("meta", "predictbase") 
    editBase = config.getint("meta", "editbase")  

    windowStart = config.getint("meta", "window_start")
    if args.start != None:
        windowStart = args.start 
        if windowStart <= 0:
            windowStart = 1
    
    windowEnd = config.getint("meta", "window_end")
    if args.end != None:
        windowEnd = args.end 
        if windowEnd < windowStart:
            windowEnd = windowStart 
        if windowEnd > 20:
            windowEnd = 20


    model = torch.load(os.path.join(args.model, "model.pth"))
    
    res, seq = functions.CalculateOneSeq(model, args.seq)
    cpos = []
    #print(res,seq)
    for i in range(9 + windowStart, 9 + windowEnd):
        if seq[i] == editBase:
            cpos.append(i)
        else:
            res[i-10] = 0
    
    bn = bN.fit(cpos, res, args.seq, mapping[predictBase], 10)
    print(res[windowStart-1:windowEnd-1])
    bn.printres()
    

if __name__ == "__main__":
    main()
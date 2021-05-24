from bayesianNetwork import BayesianNetwork
import pickle 
import configparser
import argparse
import numpy as np
import os
import models
import torch
import functions

mapping = {'A':0,'T':1,'G':2,'C':3}

def test(model_path,input_seq,start,end):    
    config = configparser.ConfigParser() 
    config.read(os.path.join(model_path, "config.ini"))

    score = np.load(os.path.join(model_path, "score.npy"))
    bN = BayesianNetwork(score=score, give=True)
    #print(bN.positions)
    predictBase = config.getint("meta", "predictbase") 
    editBase = config.getint("meta", "editbase")  
    
    start=int(start)
    windowStart = config.getint("meta", "window_start")
    if start != None:
        windowStart = start
        if windowStart <= 0:
            windowStart = 1
    end=int(end)
    windowEnd = config.getint("meta", "window_end")
    if end != None:
        windowEnd = end 
        if windowEnd < windowStart:
            windowEnd = windowStart 
        if windowEnd > 20:
            windowEnd = 20

    model = torch.load(os.path.join(model_path, "model.pth"))   
    #res, seq = functions.CalculateOneSeq(model, args.seq)
    input_seq = input_seq.upper()
    for c in input_seq:
        if c not in mapping:
            return ["invalid input", "contain unrecognized base " + str(c)] 

    if input_seq[31] != "G" or input_seq[32] != "G":
        return ["invalid input", "invalid PAM sequence, should be NGG"]

    res, seq = functions.CalculateOneSeq(model, input_seq)
    cpos = []
    #print(res,seq)
    for i in range(9 + windowStart, 9 + windowEnd):
        if seq[i] == editBase:
            cpos.append(i)
        else:
            res[i-10] = 0

    if len(cpos) == 0:
        return ["invalid input", "no editable base in edit window"]
    
    bn = bN.fit(cpos, res, input_seq, mapping[predictBase], 10)
    #print(res[windowStart-1:windowEnd-1])
    #bn.printres()
    e=",".join(str(i) for i in res[windowStart-1:windowEnd-1])
    m = str(bn.res)
    a=[e,m]
    return a
    
    
    
    
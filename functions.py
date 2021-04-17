import torch 
import torch.nn as nn
import log
import numpy as np 
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd

start = 10 
length = 20

def test(model, dsTest, baseIndex):
    predict = []
    truth = []

    indelpredict = []
    indeltruth = []

    for _ in range(20):
        predict.append([]) 
        truth.append([])

    model = model.eval().cpu()
    with torch.no_grad():
        for _, j in enumerate(dsTest):
            seq, mask, target, indel, allp = j 
            seq = seq.long() 
            mask = mask.float()
            target = target.float()
            indel = indel.float()
            out, _ = model(seq.long())

            target = target.numpy()
            out = out.numpy()
            #indelpre = indelpre.numpy()

            for l in range(target.shape[0]):
                #indelpredict.append(indelpre[l])
                indeltruth.append(indel[l])
                for m in range(20):
                    if mask[l][m+10] > 0.5:
                        #print("here")
                        predict[m].append(out[l][m])
                        truth[m].append(target[l][m+start][baseIndex])
                        #total.append(out[l][m])
                        #totalres.append(target[l][m+10][base])

    return predict, truth#, indelpredict, indeltruth

def trainonce(model, ds, optimizer, criterion, device, baseIndex):
    
    model = model.train().to(device)
    totalloss = 0.0
    for _, j in enumerate(ds):
        seq, mask, target, indel, allp = j 
        seq = seq.long().to(device)
        mask = mask.float().to(device)
        target = target.float().to(device)
        indel = indel.float().to(device)
        allp = allp.float().to(device)
        
        out, editproportion = model(seq)  
        loss = (out*100 - target[:, start:start+length, baseIndex]*100) 
        loss = loss * mask[:, start:start+length]
        
        lossr = criterion(loss, torch.zeros_like(loss)) + criterion(editproportion.squeeze(), allp)
        totalloss += lossr.item()

        optimizer.zero_grad() 
        lossr.backward() 
        optimizer.step() 

    return totalloss

def eval(pre1, real1, positions):

    pre = [] 
    real = []
    for i in positions:
        pre.extend(pre1[i])
        real.extend(real1[i])

    res1 = np.corrcoef(np.asarray(pre), np.asarray(real))[0, 1]
    res2 = sqrt(mean_squared_error(pre, real))
    return (res1, res2)

def CalculatePearson(pre, real):

    ret = np.corrcoef(np.asarray(pre), np.asarray(real))[0, 1]
    return ret

def CalculateAllResults(model, dsTest, baseIndex, savepath, positions):
    predict = []
    truth = []
    mapping = {0:"A", 1:"T", 2:"G", 3:"C"}

    for _ in range(20):
        predict.append([]) 
        truth.append([])

    f = open(savepath+"data.tsv", "w")
    f.write("sequence\tposition\tpredict\ttruth\n")

    g = open(savepath+"metrics.tsv", "w")
    for i in range(20):
        g.write("\t"+str(i))
    
    g.write("\t"+"+".join(map(lambda x: str(x), positions))+"\n")

    model = model.eval().cpu()
    for _, j in enumerate(dsTest):
        seq, mask, target, indel, _ = j 
        seq = seq.long() 
        mask = mask.float()
        target = target.float()
        indel = indel.float()
        out, _ = model(seq.long())


        seq = seq.detach().numpy()
        target = target.detach().numpy()
        out = out.detach().numpy()

        for l in range(target.shape[0]):
            originalSeq = seq[l]
            originalSeq = "".join(list(map(lambda x: mapping[x], originalSeq)))

            for m in range(20):
                if mask[l][m+10] > 0.5:
                    predict[m].append(out[l][m])
                    truth[m].append(target[l][m+10][baseIndex])
                    f.write(originalSeq+"\t")
                    f.write(str(m)+"\t")
                    f.write(str(round(out[l][m], 8))+"\t")
                    f.write(str(round(target[l][m+10][baseIndex], 8))+"\n")
    
    res1, res2 = eval(predict, truth, positions)
    g.write("RMSE\t")
    for i in range(20):
        if len(predict[i])>2:
            a = sqrt(mean_squared_error(predict[i], truth[i]))
            g.write(str(round(a, 8))+"\t")
        else:
            g.write("0.0"+"\t")
    g.write(str(round(res2, 8))+"\n")

    g.write("pearson\t")
    for i in range(20):
        if len(predict[i])>2:
            a = np.corrcoef(predict[i], truth[i])[0,1]
            g.write(str(round(a, 8))+"\t")
        else:
            g.write("0.0"+"\t")
    g.write(str(round(res1, 8))+"\n")

    f.close()
    g.close()

    return res1, res2

def CalculateOneSeq(model, sequence):
    assert(len(sequence) == 40)
    mapping = {'A':0,'T':1,'G':2,'C':3}
    a = []
    for j in range(40):
        a.append(mapping[sequence[j]])
    a = torch.tensor(a)
    a = a.long().unsqueeze(0)
    with torch.no_grad():
        ret = model(a).squeeze().numpy()
    return ret


import torch 
import torch.nn as nn
import log
import numpy as np 
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd

start = 10 
length = 20

mapping2 = {0:'A',1:'T',2:'G',3:'C'}
# def getkl(model, dsTest, bN, base1="C", base2="G"):
#     ret = 0 
#     kl = nn.KLDivLoss(reduction="sum")
#     model = model.eval().cpu()
#     with torch.no_grad(): 
#         for _, j in enumerate(dsTest):
#             seq, mask, target, indel, allp, proportion = j 
#             batchsize = seq.shape[0]
#             pro = proportion
#             #pro = getmargin(proportion, 12, 20, 13, 18)
#             seq = seq.long()
#             res, _ = model(seq) 
#             res = res.numpy()
#             allres = [] 
#             seq = seq.numpy()
#             for i in range(batchsize):
#                 pos = []
#                 for k in bN.positions:
#                     if seq[i][k] == 3:
#                         pos.append(k)
#                 #print(seq[i])
#                 #print(list(map(lambda x: mapping2[x], seq[i])))
#                 e = bN.fit(pos, res[i], list(map(lambda x: mapping2[x], seq[i])), base2, 10)
#                 #print(pos, bN.positions)
#                 allres.append(e.getdistribution(12, 20))
#             allres = np.stack(allres) 
#             ret += kl(torch.tensor(allres+0.0001).log(), torch.tensor(pro)) + \
#             kl(torch.tensor(pro).log(), torch.tensor(allres+0.0001))  
#     # print(pos)
#     # print(allres) 
#     # print(pro)
#     return ret


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
            seq, mask, target= j 
            seq = seq.long() 
            mask = mask.float()
            target = target.float()
            out, _ = model(seq.long())

            #print(seq)

            #print(mask)

            target = target.numpy()
            out = out.numpy()

            for l in range(target.shape[0]):
                for m in range(20):
                    if mask[l][m+10] > 0.5:
                        
                        predict[m].append(out[l][m])
                        truth[m].append(target[l][m+start][baseIndex])
                        #total.append(out[l][m])
                        #totalres.append(target[l][m+10][base])
    
    #print(predict, truth)

    return predict, truth#, indelpredict, indeltruth

def trainonce(model, ds, optimizer, criterion, device, baseIndex):
    
    model = model.train().to(device)
    totalloss = 0.0
    for _, j in enumerate(ds):
        seq, mask, target= j 
        seq = seq.long().to(device)
        mask = mask.float().to(device)
        target = target.float().to(device)
        out, editproportion = model(seq) 
        target = target[:, start:start+length, baseIndex]#/(1-allp) 
        
        loss = (out*100 - target*100) 
        loss = loss * mask[:, start:start+length]
        
        lossr = criterion(loss, torch.zeros_like(loss))# + criterion(editproportion, 1-allp)
        totalloss += lossr.item()

        optimizer.zero_grad() 
        lossr.backward() 
        optimizer.step() 

    return totalloss

def eval(pre1, real1, positions):

    pre = [] 
    real = []
    allres1 = [] 
    allres2 = []
    for i in positions:
        pre.extend(pre1[i])
        real.extend(real1[i])
        allres2.append(np.mean(real1[i]))

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
        seq, mask, target = j 
        seq = seq.long() 
        mask = mask.float()
        target = target.float()
        
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


def getmargin(data, start1, end1, start2, end2):

    ret = np.zeros((data.shape[0], 2**(end2-start2)))
    for k in range(data.shape[0]):
        for i in range(len(data[k])):
            idx = 0 
            for j in range(end1-start1):
                if ((1<<j) & i) > 0:
                    if j+start1 >= start2 and j+start1<end2:
                        idx += 1 << (j+start1-start2)
            ret[k][idx] += data[k][i] 
    return ret 

def CalculateOneSeq(model, sequence):
    assert(len(sequence) == 40)
    mapping = {'A':0,'T':1,'G':2,'C':3}
    a = []
    for j in range(40):
        a.append(mapping[sequence[j]])
    a = torch.tensor(a)
    a = a.long().unsqueeze(0)
    with torch.no_grad():
        ret, _ = model(a)
        ret = ret.squeeze().numpy()
    return ret, a.squeeze()


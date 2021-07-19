import os
import pickle
import dataset 
import numpy as np 
import math
import log
import pickle
import time
import os

class BayesianNetworkResult():
    def __init__(self, seq, probability, cpos, base, allvalues, bias=0, z=None) -> None:
        self.base = base 
        seq = list(seq) 
        self.probability = probability 
        self.cpos = cpos 
        
        self.res = {}
        self.allvalues = allvalues
        self.bias = bias
        if len(cpos) > 0:
            self.ori = seq[cpos[0]]
        else:
            self.ori = "C"
            return
        #self.array = np.zeros((2**len(cpos)))
        #self.array[0]+=1-z
        for i in range(2**len(cpos)):
            if 1 & i > 0:
                pre = 1
                pro = probability[0][1]
                seq[cpos[0]] = self.base
            else:
                pre = 0
                pro = 1-probability[0][0] 
                seq[cpos[0]] = self.ori
            for j in range(1, len(cpos)):
                if i&(1<<j) > 0:
                    pro *= probability[j][pre]
                    seq[cpos[j]] = self.base
                    pre = 1 
                else:
                    pro *= 1 - probability[j][pre]
                    seq[cpos[j]] = self.ori
                    pre = 0
            self.res["".join(seq)] = pro

    def printres(self):
        print(self.res)   

    def getdistribution(self, start, end):
        ret = np.zeros((2**(end-start)))
        subset = [] 
        for i in range(start, end):
            if i in self.cpos:
                subset.append(i) 
        subset.sort() 
        if len(subset)==0:
            ret[0] = 1.0
            return ret 
        #print(subset, self.allvalues)
        for i in range(2**len(subset)):
            idx = 0
            if i & 1 > 0:
                pro = self.allvalues[subset[0]-self.bias]
                pre = 1 
                idx += 1<<(subset[0]-start)
            else:
                pro = 1-self.allvalues[subset[0]-self.bias]
                pre = 0 
            for j in range(1, len(subset)):
                if i & (1<<j) > 0:
                    pro *= self.probability[self.cpos.index(subset[j])][pre]
                    idx += 1 << (subset[j] - start) 
                    pre = 1 
                else:
                    pro *= 1 - self.probability[self.cpos.index(subset[j])][pre]
                    pre = 0
            #print(idx, pro)
            ret[idx] = pro 
        return ret
            



def solvecorr(a, b, c, d):
    x = (d*math.sqrt(a*b*c*(1-c))+a*c)/a 
    y = (c-a*x)/b 
    return x, y

def solve(a, b, c, d):
    t1 = b*(d-1)/a 
    t2 = d+b/a-c/a*(d-1) 
    t3 = -c/a 
    y = (-t2+math.sqrt(t2*t2-4*t1*t3))/(2*t1)
    x = (c-b*y)/a 
    
    return x, y 

class BayesianNetwork():
    def __init__(self, score, positions=None, metric="cross") -> None:
        if positions is not None:
            self.positions = positions
        else:
            self.positions = list(range(11, 30))
        #self.pos, self.neg = countfrequency(data, positions)
                
        
        self.score = score 
        # self.solve = solve
        
        if metric == "corr": 
            self.score[np.isnan(self.score)] = 0 
            self.solve = solvecorr 
        elif metric == "cross":
            self.score[np.isnan(self.score)] = 1 
            self.solve = solve
        else: 
            raise NotImplementedError

    def fit(self, positions, values, seq, base, bias):
        subset = [] 
        for i in positions:
            if i in self.positions:
                subset.append(i) 
        subset.sort() 
        pro = np.zeros((len(subset),2))
        if len(subset)==0:
            pro = np.zeros((1,2))
            pro[0] = 1.0
            return BayesianNetworkResult(seq, pro, subset, base, values, bias)
         

        pro[0][1] = values[subset[0]-bias] 
        pro[0][0] = pro[0][1] 

        for i in range(1, len(subset)):
            x, y = self.solve(values[subset[i-1]-bias], 1-values[subset[i-1]-bias], values[subset[i]-bias], 
            self.score[self.positions.index(subset[i-1])][self.positions.index(subset[i])])      
            # print(values[subset[i-1]]*x+(1-values[subset[i-1]])*y, values[subset[i]])   
            # print((values[subset[i-1]]*x-values[subset[i-1]]*values[subset[i]])/math.sqrt(values[subset[i-1]]*(1-values[subset[i-1]])*values[subset[i]]*(1-values[subset[i]])),
            #  self.score[self.positions.index(subset[i-1])][self.positions.index(subset[i])])   
            pro[i][1] = x 
            pro[i][0] = y 


        res = BayesianNetworkResult(seq, pro, subset, base, values, bias)
        return res

    def drawHeatMap(self, path):
        raise NotImplementedError

if __name__ == "__main__":
    
    t = np.load("./datasetsample_YE1-FNLS-GBE3/score.npy")
    a = BayesianNetwork(t)
    bn = a.fit([13,14,15], [0.4]*20, "AAAAAAAAAACCCCCCCCCCCCCCCCCCCCCGGCCCCCCC", "G", 10)
    print(bn.res)
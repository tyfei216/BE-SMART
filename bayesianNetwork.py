import pickle
import dataset 
import numpy as np 
import math
import log

def countfrequency(data, positions, indices=None, metric="cross"):
    length = len(positions)  
    #pseudo counts
    pos0 = np.ones((length, length))
    pos1 = np.ones((length, length))
    neg0 = np.ones((length, length))
    neg1 = np.ones((length, length))
    cnts = data["cnts"]
    indel = data["indel"]
    allp = data["allp"]
    cpos = data["cpos"]
    if indices == None:
        indices = range(len(cnts))

    for i in indices:
        totalcnts = cnts[i] * (1-indel[i])+1 
        subset = [] 
        for j in cpos[i]:
            if j in positions:
                subset.append(j) 
        if len(subset) < 2:
            continue 
        for j in range(0, len(allp[i])): 
            for k in range(len(subset)-1):
                for l in range(k+1, len(subset)):
                    if (j & (1<<(cpos[i].index(subset[k])))>0) ^ (j & (1<<(cpos[i].index(subset[l])))>0):
                        if j & (1<<(cpos[i].index(subset[k])))>0:
                            neg0[positions.index(subset[k])][positions.index(subset[l])] += totalcnts*allp[i][j]
                            neg0[positions.index(subset[l])][positions.index(subset[k])] = neg0[positions.index(subset[k])][positions.index(subset[l])]
                        else:
                            neg1[positions.index(subset[k])][positions.index(subset[l])] += totalcnts*allp[i][j]
                            neg1[positions.index(subset[l])][positions.index(subset[k])] = neg1[positions.index(subset[k])][positions.index(subset[l])]
                    else:
                        if j & (1<<(cpos[i].index(subset[k])))>0:
                            pos0[positions.index(subset[k])][positions.index(subset[l])] += totalcnts*allp[i][j]
                            pos0[positions.index(subset[l])][positions.index(subset[k])] = pos0[positions.index(subset[k])][positions.index(subset[l])]
                        else:
                            pos1[positions.index(subset[k])][positions.index(subset[l])] += totalcnts*allp[i][j]
                            pos1[positions.index(subset[l])][positions.index(subset[k])] = pos1[positions.index(subset[k])][positions.index(subset[l])]
    # print(pos0)
    # print(pos1)
    # print(neg0)
    # print(neg1)
    s = pos0+pos1+neg0+neg1
    corr = (pos0*s-(neg0+pos0)*(neg1+pos0))/np.sqrt((neg1+pos0)*(neg1+pos1)*(neg0+pos1)*(neg0+pos0)) 
    if metric == "corr":
        return corr 
    elif metric == "cross":
        return pos1*pos0/(neg0*neg1)
    else: 
        raise NotImplementedError

class BayesianNetworkResult():
    def __init__(self, seq, probability, cpos, base, allvalues, bias=0, z=None) -> None:
        self.base = base 
        seq = list(seq) 
        self.probability = probability 
        self.cpos = cpos 
        self.ori = seq[cpos[0]]
        self.res = {}
        self.allvalues = allvalues
        self.bias = bias
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

def solve(a,b,c,d):
    t1 = b*(d-1)/a 
    t2 = d+b/a-c/a*(d-1) 
    t3 = -c/a 
    y = (-t2+math.sqrt(t2*t2-4*t1*t3))/(2*t1)
    x = (c-b*y)/a 
    
    return x, y 

class BayesianNetwork():
    def __init__(self, path, positions, indices=None, metric="cross") -> None:
        with open(path, "rb") as f:
            data = pickle.load(f) 
        self.positions = positions 
        #self.pos, self.neg = countfrequency(data, positions)
        self.score = countfrequency(data, positions, indices=indices, metric=metric)
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

if __name__ == "__main__":
    # print(solvecorr(0.5, 0.5, 0.9, 0.9))
    # a = BayesianNetwork("../proportion3/YE1-FNLS-CGBE.pkl", [11,12,13,14,15,16,17,18,19,20])
    a = BayesianNetwork("../proportion3/BE4max.pkl", [11,12,13,14,15,16,17,18,19,20], metric="cross")
    e = a.fit([13,14,16,17],[1,1,1,0.5,0.6,0.3,0.5,0.2],"CCCCCCCCCCCCCCCTCCCCCCCCCCCCCC","G", 10)
    e.printres()
    print(e.getdistribution(13, 18))
    # print(e.array)
    # print(a.score)
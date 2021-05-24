import os
import pickle
import dataset 
import numpy as np 
import math
import log
import pickle
import time
import os

def countfrequency(data, positions, indices=None, metric="cross"):
    length = len(positions)  
    # pseudo counts
    pos0 = np.zeros((length, length))+0.0000000001
    pos1 = np.zeros((length, length))+0.0000000001
    neg0 = np.zeros((length, length))+0.0000000001
    neg1 = np.zeros((length, length))+0.0000000001
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

    s = pos0+pos1+neg0+neg1
    corr = (pos0*s-(neg0+pos0)*(neg1+pos0))/np.sqrt((neg1+pos0)*(neg1+pos1)*(neg0+pos1)*(neg0+pos0)) 
    if metric == "corr":
        return corr 
    elif metric == "cross":
        return pos1*pos0/(neg0*neg1), (pos0, pos1, neg0, neg1)
    else: 
        raise NotImplementedError

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
    def __init__(self, path=None, positions=None, indices=None, score=None, give = False,metric="cross") -> None:
        if positions is not None:
            self.positions = positions
        else:
            self.positions = list(range(11, 30))
        #self.pos, self.neg = countfrequency(data, positions)
        if not give:
            if path is None:
                print("path not given")
                exit()
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.score, self.raw = countfrequency(data, positions, indices=indices, metric=metric)
        if give:
            print("used given score")
            self.score = score 
            self.solve = solve
        
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
    # print(solvecorr(0.5, 0.5, 0.9, 0.9))
    # l = os.listdir("../proportion3")
    # for i in l:
    #     print(i)
    #     a = BayesianNetwork("../proportion3/"+i, 
    #     [11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29])
    #     os.mkdir("./trainedmodels/"+i[:-4])
    #     np.save("./trainedmodels/"+i[:-4]+"/score.npy", a.score)
    # exit()
    
    mask = np.zeros((19,19),dtype=np.bool)
    for i in range(19):
        for j in range(i, 19):
            mask[i][j] = True 

    # with open("./YE1-FNLS-BE3/bayesianNetwork.pkl", "rb") as f:
    #     b = pickle.load(f)
        #pickle.dump(a, f)
    # exit()
    import matplotlib.pyplot as plt

    import seaborn as sns 

    names = os.listdir("../proportion3/")

    for i in names:
        print(i)
        #filename = "./trainedmodels/"+i[:-4]+"/score.npy"
        #a = BayesianNetwork("../proportion3/"+i, [11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29], metric="cross")
        #a = np.load(filename)
        
        #name = os.path.basename(filename)
        sns.set()
        print(a.score)
        fig = plt.figure() 
        #print(a.shape, mask.shape)
        np.save("./trainedmodels/"+i[:-4]+"/score.npy", a.score)
        np.savetxt("./heatmap20/"+i[:-4]+".txt", a.score)
        np.savetxt("./heatmap20/"+i[:-4]+"_log.txt", np.log(a.score))
        with open("./heatmap20/raw/"+i, "wb") as f:
            pickle.dump(a.raw, f)
        sns_plot = sns.heatmap(np.log(a.score), mask=mask, vmax=4, vmin=-4,xticklabels=range(2,21),yticklabels=range(2,21))
        plt.title(i[:-4])
        plt.savefig("./heatmap20/"+i[:-4]+".pdf")
        plt.close()
    exit()

    e = a.fit([13,14,16,17],[1,1,1,0.5,0.6,0.3,0.5,0.2],"CCCCCCCCCCCCCCCTCCCCCCCCCCCCCCCCCCCCCCCC","G", 10)
    e.printres()
    print(e.getdistribution(13, 18))
    time_end=time.time()
    print('totally cost',time_end-time_start)
    # print(e.array)
    # print(a.score)

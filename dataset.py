from models import Proportion
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import log
import torch

mapping = {'A':0,'T':1,'G':2,'C':3}

class BaseEditingDataset(Dataset):
    def __init__(self, data, sequence, indel, allp, proportion, editBase = 3, rawSequence = True):

        super(BaseEditingDataset, self).__init__()
        data = torch.tensor(data).float()
        proportion = torch.tensor(proportion).float()
        self.data = data
        self.allp = allp
        #log.debug("building base editing dataset with dimensions: " + str(data.shape))
        
        assert(data.shape[1] == 40)
        assert(data.shape[0] == len(sequence))

        assert(data.shape[0] == proportion.shape[0]) 
        assert(proportion.shape[1] == 256)
        
        #print('data_dim', self.data.shape)
        
        if rawSequence:
            #log.debug("encoding sequencing")
            sequences = []
            for i in range(len(sequence)):
                assert(len(sequence[i]) == 40)
                sequences.append(np.array(list(map(lambda x: mapping[x], sequence[i]))))
        else:
            sequences = sequence

        self.indel = indel

        self.proportion = proportion

        self.sequence = np.array(sequences).astype(np.int)
        
        #log.debug("labeling editing position")
        self.Cpos = np.zeros((self.data.shape[0], 40))
        for i in range(len(self.data)):
            for j in range(10,30):
                if sequence[i][j] == editBase:
                    self.Cpos[i][j] = 1.0

    def __len__(self):
        return self.data.shape[0]


    def __getitem__(self, index):
        seq = self.sequence[index]
        cpos = self.Cpos[index]
        data = self.data[index]
        indel = self.indel[index]
        allp = self.allp[index][0]
        proportion = self.proportion[index]

        return seq, cpos, data, indel, allp, proportion

def SplitDataset(ds:BaseEditingDataset, sizes=None, split=None, savepath=None):
    
    if sizes == None:
        sizes = [6, 1, 3]

    if len(sizes) != 3:
        log.warning("invalid split of "+str(sizes))
        sizes = [6,1,3]
        log.warning("returning to default split of "+str(sizes))
    
    if sum(sizes)!=10 or sizes[0] < 0 or sizes[1] < 0 or sizes[2] < 0:
        log.warning("invalid split of "+str(sizes))
        sizes = [6,1,3]
        log.warning("returning to default split of "+str(sizes))
    
    splits1 = len(ds)*sizes[0]//10
    splits2 = len(ds)*(sizes[0]+sizes[1])//10
    if split == None:
        log.info("randomizing new splits")        
        indices = list(range(len(ds)))
        random.shuffle(indices)
        if savepath != None:
            with open(savepath, "wb") as f: 
                pickle.dump(indices, f)
    else:
        indices = split

    trainSampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:splits1])
    dsTrain = DataLoader(ds, sampler=trainSampler, batch_size=20)

    validSampler = torch.utils.data.sampler.SubsetRandomSampler(indices[splits1:splits2])
    dsValid = DataLoader(ds, sampler=validSampler, batch_size=100)

    testSampler = torch.utils.data.sampler.SubsetRandomSampler(indices[splits2:])
    dsTest = DataLoader(ds, sampler=testSampler, batch_size=100)

    return dsTrain, dsValid, dsTest, indices


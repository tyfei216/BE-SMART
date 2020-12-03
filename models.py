import torch 
import copy
import torch.nn as nn
import torch.nn.functional as F

class BaseUnit(nn.Module):
    def __init__(self, src_vocab=4, inputDim=5,embedDim=16,hiddenDim=256,dropout=0.3):
        super(BaseUnit, self).__init__()
        self.embedding = nn.Embedding(src_vocab, embedDim)
        #self.dropout0 = nn.Dropout(0.3)
        self.fc1 = nn.Conv1d(embedDim, hiddenDim,inputDim*2+1,padding=0)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hiddenDim, hiddenDim//2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hiddenDim//2, 1)

    def forward(self, x):
        x = self.embedding(x)
        # x = self.dropout0(x)
        x = x.transpose(-1, -2)
        x = F.relu(self.fc1(x)).squeeze()
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc3(x))
        return x.squeeze()

class BaseModel(nn.Module):
    def __init__(self, length=5, dropout=0.3):
        super(BaseModel, self).__init__()
        self.length = length
        single = BaseUnit(inputDim=length, dropout=dropout)
        self.allmodels = nn.ModuleList([copy.deepcopy(single) for _ in range(20)])

    def forward(self, x):
        results = []
        for i in range(len(self.allmodels)):
            results.append(self.allmodels[i](x[:,10+i-self.length:11+i+self.length]).view(-1, 1))
        res = torch.cat(results, 1)
        return res

class FullModel(nn.Module):
    def __init__(self, lengthlist = None, dropout=0.3):
        super(FullModel, self).__init__() 
        if lengthlist == None:
            lengthlist = [3,4,5]

        models = [] 
        for i in range(len(lengthlist)):
            models.append(BaseModel(length=lengthlist[i], dropout=dropout))

        self.models = nn.ModuleList(models)

        weights = torch.ones((len(models),20)).float()
        weights = torch.nn.Parameter(weights)
        self.register_parameter("final_weights", weights)

    def forward(self, x):
        allres = []
        for i in range(len(self.models)):
            allres.append(self.models[i](x))

        weights = torch.softmax(self.final_weights, 0)
        ret = torch.zeros_like(allres[0])
 
        for i in range(len(self.models)):
            ret += weights[i]*allres[i]

        return ret
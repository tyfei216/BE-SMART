import torch 
import copy
import torch.nn as nn
import torch.nn.functional as F

class BaseUnit(nn.Module):
    def __init__(self, src_vocab=4, inputDim=5, embedDim=16, hiddenDim=128, dropout=0.3, outputLatent=False, globalDim=16):
        super(BaseUnit, self).__init__()
        self.outputLatent = outputLatent
        
        self.embedding = nn.Embedding(src_vocab, embedDim)
        #self.dropout0 = nn.Dropout(0.3)
        self.fc1 = nn.Conv1d(embedDim, hiddenDim, inputDim*2+1, padding=0)
        self.dropout1 = nn.Dropout(dropout)
        #self.fc2 = nn.Linear(hiddenDim, hiddenDim)
        self.fc2 = nn.Linear(hiddenDim+globalDim, hiddenDim)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hiddenDim, 1)

    def forward(self, x, y):
        x = self.embedding(x)
        # x = self.dropout0(x)
        x = x.transpose(-1, -2)
        x = F.relu(self.fc1(x)).squeeze()
        x = self.dropout1(x)
        x = torch.cat((x, y), dim=1)
        x = F.relu(self.fc2(x))
        out = self.dropout2(x)
        out = torch.sigmoid(self.fc3(out))
        
        if self.outputLatent:
            return out.squeeze(), x
        
        return out.squeeze()

class BaseModel(nn.Module):
    def __init__(self, length=5, dropout=0.3, outputLatent=False):
        super(BaseModel, self).__init__()
        self.length = length
        single = BaseUnit(inputDim=length, dropout=dropout, outputLatent=outputLatent)
        self.allmodels = nn.ModuleList([copy.deepcopy(single) for _ in range(20)])

        self.outputLatent = outputLatent

    def forward(self, x, y):
        results = []
        if self.outputLatent:
            intermediate = []
        for i in range(len(self.allmodels)):
            
            if self.outputLatent:
                res, out = self.allmodels[i](x[:,10+i-self.length:11+i+self.length])#.view(-1, 1)
                res = res.unsqueeze(1)
                results.append(res)
                intermediate.append(out)
            
            else:
                res = self.allmodels[i](x[:,10+i-self.length:11+i+self.length], y).view(-1, 1)
                results.append(res)
            
        res = torch.cat(results, 1)
        if self.outputLatent:
            inter = torch.stack(intermediate, 1)
            return res, inter

        return res

class Decoder(nn.Module):
    def __init__(self, inputDim=16):
        super(Decoder, self).__init__()
        self.L1 = nn.Linear(inputDim, 1) 
        #self.L2 = nn.Linear(inputDim//2, 1)

    def forward(self, x):
        #x = F.relu(self.L1(x))
        x = torch.sigmoid(self.L1(x))
        return x#.squeeze()

class Encoder(nn.Module):
    def __init__(self, src_vocab=4, inputDim=40, embedDim=16, hiddenDim=256, outputDim=16, dropout=0.3):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(src_vocab, embedDim)
        #self.dropout0 = nn.Dropout(0.3)
        self.fc1 = nn.Conv1d(embedDim, 128, 5, padding=0)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128*(inputDim-4), hiddenDim)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hiddenDim, outputDim)

    def forward(self, x):
        x = self.embedding(x)
        # x = self.dropout0(x)
        x = x.transpose(-1, -2)
        x = F.relu(self.fc1(x)).squeeze()
        x = self.dropout1(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc3(x))
        return x

class Proportion(nn.Module):
    
    def __init__(self, inputDim = 128*9, hiddenDim = 512, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Conv1d(inputDim, hiddenDim, 10, padding=0)
        self.d1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hiddenDim, hiddenDim)
        self.d2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hiddenDim, 255) 

    def forward(self, x):
        x = x.transpose(-1, -2)
        x = F.relu(self.fc1(x)).squeeze()
        x = self.d1(x) 
        x = F.relu(self.fc2(x))
        x = self.d2(x)
        x = torch.softmax(self.fc3(x))
        return x 

class FullModel(nn.Module):
    def __init__(self, lengthlist = None, dropout=0.3):
        super(FullModel, self).__init__() 
        if lengthlist == None:
            lengthlist = [3,4,5]

        models = [] 
        for i in range(len(lengthlist)):
            models.append(BaseModel(length=lengthlist[i], dropout=dropout))

        self.models = nn.ModuleList(models)

        #self.proportion = Proportion(inputDim=len(lengthlist)*128) 

        weights = torch.ones((len(models),20)).float()
        weights = torch.nn.Parameter(weights)
        self.register_parameter("final_weights", weights)

        self.encoder = Encoder() 
        self.decoder = Decoder(inputDim=16)

    def forward(self, x):

        y = self.encoder(x)

        allres = []
        #features = []
        for i in range(len(self.models)):
            res = self.models[i](x, y)
            #features.append(q[:, :10, :])
            allres.append(res)

        #features = torch.cat(features, dim=2)
        #pro = self.proportion(features.detach())
        #print(allres)
        z = self.decoder(y)

        # adjust = z * pro
        # res = torch.cat([1-z, adjust], dim=1)



        weights = torch.softmax(self.final_weights, 0)
        ret = torch.zeros_like(allres[0])
 
        for i in range(len(self.models)):
            ret += weights[i]*allres[i]

        return ret, z

if __name__ == "__main__":
    a = torch.randint(0, 4, (5, 40))
    b = FullModel(lengthlist=[3,4,5])
    c = b(a)
    print(c)
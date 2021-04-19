import torch 
import log 
import argparse
import os
import pickle
import models
import time
import functions

def Args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", default=None, type=str, help="path to a training config file")
    parser.add_argument("-model", default=None, type=str, help="path to the model")
    parser.add_argument("-ds", default=None, type=str, help="path to the test set")
    parser.add_argument("-result", default=".\\results\\", type=str, help="path to save the results")
    parser.add_argument("-seq", default=None, type=str, help="the input sequence")
    parser.add_argument("-predictBase", type=str, default="C", help="which base to predict (default \"C\")")
    parser.add_argument("-editres", type=str, default="G", help="edit result of the base")
    parser.add_argument("-bN", default=None, type=str, help="the base for the bayesianNetwork")
    parser.add_argument("-evalpositions", action="extend", nargs="+", type=int, default=None, help="the position for pearson calculation, intergers[0-20]")
    args = parser.parse_args()

    if args.config == None and args.model == None:
        print("model not given")
        exit()

    if not os.path.exists(args.result):
        os.makedirs(args.result)
    
    return args

# you may want to rewrite this function to read your own data
def ReadFromDS(path):
    with open(path, "rb") as f:
        seqs = pickle.load(f)

    return seqs


def main():
    args = Args()
    name = os.path.basename(args.model)
    f = open(args.result+name[:name.find(".")]+".tsv", "w")
    f.write("seq\t\pos\t\pre\n")
    if args.model != None:
        model = torch.load(args.model, map_location=torch.device('cpu'))

    if args.bN != None:
        with open(args.bN, "rb") as f:
            bayesianNetwork = pickle.load(f)

    if args.seq != None:
        res = functions.CalculateOneSeq(model, args.seq)
        cpos = []
        for i in range(10,30):
            if args.seq[i] == args.predictBase:
                cpos.append(i)
                f.write(args.seq+"\t")
                f.write(str(i-10)+"\t")
                f.write(str(round(res[i-10], 8))+"\n")
        bn = bayesianNetwork.fit(cpos, res, args.seq, args.editres, 10)
        bn.printres()

    if args.ds != None:
        seqs = ReadFromDS(args.ds)
        for seq in seqs:
            res = functions.CalculateOneSeq(model, seq)
            for i in range(10,30):
                if seq[i] == args.predictBase:
                    f.write(args.seq+"\t")
                    f.write(str(i-10)+"\t")
                    f.write(str(round(res[i-10], 8))+"\n")

if __name__=="__main__":
    main()

        

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
    parser.add_argument("-model", required=True, type=str, help="path to the model")
    parser.add_argument("-ds", default=None, type=str, help="path to the test set")
    parser.add_argument("-result", default=".\\results\\", type=str, help="path to save the results")
    parser.add_argument("-seq", default=None, type=str, help="the input sequence")
    parser.add_argument("-predictBase", type=str, default="C", help="which base to predict (default \"C\")")
    args = parser.parse_args()

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
    model = torch.load(args.model, map_location=torch.device('cpu'))

    if args.seq != None:
        res = functions.CalculateOneSeq(model, args.seq)
        for i in range(10,30):
            if args.seq[i] == args.predictBase:
                f.write(args.seq+"\t")
                f.write(str(i-10)+"\t")
                f.write(str(round(res[i-10], 8))+"\n")

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

        

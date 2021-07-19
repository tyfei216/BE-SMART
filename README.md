# BE-SMART



BE-SMART is a deep learning tool for CRISPR base editing efficiency prediction.



## Dependencies



The model is build with



- Python 3.8.3

- standard packages: numpy, pandas, pickle, sklearn

- pytorch 1.6.0



##  Usage



For training:



```powershell

python train.py -ds [path-to-dataset]

python train.py -ds ./datasetsample_YE1-FNLS-GBE3

```



For testing:



```powershell

python example.py -model [path-to-saved-model] -seq [raw-input-sequence]

python example.py -model ./trainedmodels/YE1-FNLS-CGBE -seq CCGCATGCGGGCGCTCCGGGCCCATCCTGAGGGCCCGGCC
```



The input dataset contains 3 files, `seq.txt` contains all target site sequences. `outcome.npy` is a numpy array file containing the editing results, Each target corresponds with a 40 $\times$ 4 array, referring the probability of each base being edited to A, T, G, C. `score.npy` is the scoring matrix for the Bayesian Network. This matrix is not used for the training of the model. It is only used for outputting the proportion of all outcomes. 




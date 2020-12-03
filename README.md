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

```



For testing:



```powershell

python test.py -model [path-to-saved-model] -seq [raw-input-sequence]

```



or



```powershell

python test.py -model [path-to-saved-model] -ds [path-to-dataset]

```




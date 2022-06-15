# IRREGULAR TIME SERIES
## THIRD STEP: Running the code using the Command-Line
Command Line Arguments:
1. PATH1: path of the dataset for patients WITH cancer 
2. PATH2: path of the dataset for patients WITHOUT cancer

***Scripts to be run***:

Before remember to change directory *cd Model_1* and then launch the scripts
```bash
    python main.py ../dataset/cancer.csv ../dataset/nocancer.csv 
    python main.py ../dataset/cancer.csv ../dataset/nocancer.csv 
    python main.py ../dataset/cancer.csv ../dataset/nocancer.csv 
    python main.py ../dataset/cancer.csv ../dataset/nocancer.csv 
```
***
## FOURTH STEP: Results to download
* The folders *lstm* contain 4 sub-folders (one for each type of input). In the single folder there are the following outputs:
  1. the confusion matrix (conf_mat.png)
  2. the ROC curve (roc_curve.png)
  3. the model 
  4. the metrics (results.txt)
  5. a table with accuracy/loss curve

* *boxplot.png* 

### TYPE OF PROBLEM

1. Binary classification problem (label 0: no cancer, label 1: cancer)

2. Total number of patients: 84496

3. Balanced classes (50% patients with cancer, 50% patients without cancer)


4. Training set: 80% Test set: 20% 


5. Minimum time series' length for each patient = 4


6. 50 <= age < 100


### DIFFERENT TYPES OF INPUT
INPUT 1: resampling, no interpolation, NaN values --> -1

INPUT 2: resampling, no interpolation, NaN values --> -1, binary indicator

INPUT 3: resampling, interpolation

INPUT 4: resampling, interpolation, binary indicator

### FEATURES
*PSA*: real value of the PSA

*BINARY VALUE*: 0 means real value while 1 means missing value


### MODEL

*RNN*
*LSTM*
*GRU*




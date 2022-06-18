# IRREGULAR TIME SERIES
## THIRD STEP: Running the code using the Command-Line
Command Line Arguments:
1. PATH1: path of the dataset for patients WITH cancer 
2. PATH2: path of the dataset for patients WITHOUT cancer


***Scripts to be run***:

Before remember to change directory *cd Model_2* and then launch the script
```bash
    python main.py ../dataset/cancer.csv ../dataset/nocancer.csv 
```
***
## FOURTH STEP: Results to download
The folders *cnn1d*, *lstm*, *mlp* contain 4 sub-folders (one for each type of input). In the single folder there are the following outputs:
1. the confusion matrix (conf_mat.png)
2. the ROC curve (roc_curve.png)
3. the model 
4. the metrics (results.txt)
5. a table with accuracy/loss curve

In the folder *boxplot* there are the bloxplots for each type of input.

### TYPE OF PROBLEM

1. Binary classification problem (label 0: no cancer, label 1: cancer)

2. Total number of patients: 84496


3. Balanced classes (50% patients with cancer, 50% patients without cancer)


4. Training set: 80% Test set: 20% 


5. Minimum time series' length for each patient = 5


6. 50 <= age < 100


### DIFFERENT TYPES OF INPUT

| age           | psa           | -             | -              |
|---------------|---------------|---------------|----------------|
| **age**       | **psa**       | **delta_psa** | **delta_time** |
| **hoe(age)*** | **hoe(psa)*** | -             | -              |
| **hoe(age)*** | **hoe(psa)*** | **delta_psa** | **delta_time** |

*AGE*: age of the patient

*PSA*: real value of the PSA

*DELTA_PSA*: different between PSA at time t and PSA at time t-1

*DELTA_TIME*: different between time t and time t-1 (in months)

*hoe means One Hot Encoding

### DIFFERENT MODELS

*LSTM* 

*CNN1D*

*MLP*



# BASELINES
###If you want to run only the baselines:
## THIRD STEP: Running the code using the Command-Line
Command Line Arguments:
1. PATH1: path of the dataset for patients WITH cancer 
2. PATH2: path of the dataset for patients WITHOUT cancer

***Scripts to be run***:

Before remember to change directory *cd Baseline* and then launch the script
```bash
    python main.py ../dataset/cancer.csv ../dataset/nocancer.csv 
```
***
## FOURTH STEP: Results to download
* The folders *results* contain 3 sub-folders (one for each type of input). In the single folder there are the following outputs:
  1. the boxplot (boxplot.png)
  2. the heatmap 
  3. the feature importance (for each type of model) 
  4. the metrics (results.txt)
  5. the isomap



### TYPE OF PROBLEM

1. Binary classification problem (label 0: no cancer, label 1: cancer)

2. Total number of patients: 84496

3. Balanced classes (50% patients with cancer, 50% patients without cancer) only for the training part


4. Training set: 80% Test set: 20% 


5. Minimum time series' length for each patient = 5


6. 30 <= age < 100


### DIFFERENT TYPES OF INPUT
*INPUT 1*: age and psa (second last values)

*INPUT 2*: mean, median, quantile of delta time

*INPUT 3*: age, psa and velocity 


### MODELS

*SVM*

*Random Forest*

*Decision Tree*

*Ada Boost*

*Naive Bayes*

*KNN*



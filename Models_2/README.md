##FIRST STEP: Building database


***PATIENTS WITH CANCER***:
***
    SELECT psaresults.ss_number_id, (extract ( year from ambiguous_date) - extract ( year from ss_numbers.date_of_birth_15)) as age,
    (extract( year from ambiguous_date) * 12 + extract( month from ambiguous_date)) as months,
    date_of_birth_15, result_numeric as psa
    
    FROM ss_numbers, psaresults, kreftreg_data
    
    WHERE diagnosis_date >= ambiguous_date and psaresults.ss_number_id = kreftreg_data.ss_number_id and
        ss_numbers.ss_number_id = psaresults.ss_number_id and (extract ( year from ambiguous_date) - extract ( year from ss_numbers.date_of_birth_15)) >= 50
***



***PATIENTS WITHOUT CANCER***:
***
    SELECT psaresults.ss_number_id, (extract ( year from ambiguous_date) - extract ( year from ss_numbers.date_of_birth_15)) as age,
    (extract( year from ambiguous_date) * 12 + extract( month from ambiguous_date)) as months,
    date_of_birth_15, result_numeric as psa
    
    FROM (
    SELECT distinct(psaresults.ss_number_id) as id

        FROM psaresults
        
        LEFT JOIN kreftreg_data
        
        ON psaresults.ss_number_id = kreftreg_data.ss_number_id
        
        WHERE kreftreg_data.ss_number_id is NULL) as tab, ss_numbers, psaresults

    WHERE tab.id =  psaresults.ss_number_id and 
        ss_numbers.ss_number_id = psaresults.ss_number_id and (extract ( year from ambiguous_date) - extract ( year from ss_numbers.date_of_birth_15)) >= 50
***
##SECOND STEP: Save the dataset

The datasets should be saved in *Models2/db* with the names *cancer.csv* and  *nocancer.csv*

##THIRD STEP: Running the code using the Command-Line
Command Line Arguments:
1. PATH1: path of the dataset for patients WITH cancer 
2. PATH2: path of the dataset for patients WITHOUT cancer
3. DUMMIES: True / False
4. DELTA_FEATURES: True / False
5. TYPE: LSTM / CNN1D / MLP
 
***Scripts to be run***:
```bash
    python main.py db/cancer.csv db/nocancer.csv True False LSTM
    python main.py db/cancer.csv db/nocancer.csv False True LSTM
    python main.py db/cancer.csv db/nocancer.csv False False LSTM
    python main.py db/cancer.csv db/nocancer.csv True True LSTM
    
    python main.py db/cancer.csv db/nocancer.csv True False CNN1D
    python main.py db/cancer.csv db/nocancer.csv False True CNN1D
    python main.py db/cancer.csv db/nocancer.csv False False CNN1D
    python main.py db/cancer.csv db/nocancer.csv True True CNN1D
    
    python main.py db/cancer.csv db/nocancer.csv True False MLP
    python main.py db/cancer.csv db/nocancer.csv False True MLP
    python main.py db/cancer.csv db/nocancer.csv False False MLP
    python main.py db/cancer.csv db/nocancer.csv True True MLP
```
***

###TYPE OF PROBLEM

1. Binary classification problem (label 0: no cancer, label 1: cancer)


2. Balanced classes (50% patients with cancer, 50% patients without cancer)


3. Training set: 95% → 80.271 patients Test set: 5% → 4.225 patients


4. Minimum time series' length for each patient = 4


5. 50 <= age < 100


###DIFFERENT TYPES OF INPUT

|age              | psa              | -             | -              |
|-----------------|------------------|---------------|----------------|
|  **age**        | **psa**          | **delta_psa** | **delta_time** |
| **dummies(age)** | **dummies(psa)** | -             | -              |
| **dummies(age)** | **dummies(psa)** | **delta_psa** | **delta_time** |

*AGE*: age of the patient

*PSA*: real value of the PSA

*DELTA_PSA*: different between PSA at time t and PSA at time t-1

*DELTA_TIME*: different between time t and time t-1 (in months)


###DIFFERENT MODELS

*LSTM* 

*CNN1D*

*MLP*


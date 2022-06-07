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
3. DUMMIES: True/False
4. DELTA_FEATURES: True/False
 
***Scripts to be run***:
```bash
    python main.py db/cancer.csv db/nocancer.csv True False
    python main.py db/cancer.csv db/nocancer.csv False True
    python main.py db/cancer.csv db/nocancer.csv False False
    python main.py db/cancer.csv db/nocancer.csv True True
```
***

# prostate-cancer-prediction
BASELINES

MODELS 1: Time series with interpolation and discretization

MODELS 2: Irregular time series

## FIRST STEP: Building database


***PATIENTS WITH CANCER***:
***
    SELECT psaresults.ss_number_id,
    (extract ( year from ambiguous_date) - extract ( year from ss_numbers.date_of_birth_15)) * 12 * 30 +
    (extract ( month from ambiguous_date) - extract ( month from ss_numbers.date_of_birth_15)) * 30 +
    (extract ( day from ambiguous_date) - extract ( day from ss_numbers.date_of_birth_15)) as days,
    ambiguous_date, date_of_birth_15, result_numeric as psa, npcc_risk_class_group_1 as category
    
    FROM ss_numbers, psaresults, kreftreg_data
    
    WHERE diagnosis_date >= ambiguous_date and psaresults.ss_number_id = kreftreg_data.ss_number_id and
        ss_numbers.ss_number_id = psaresults.ss_number_id and (extract ( year from ambiguous_date) - extract ( year from ss_numbers.date_of_birth_15)) >= 30
***



***PATIENTS WITHOUT CANCER***:
***
    SELECT psaresults.ss_number_id, 
    (extract ( year from ambiguous_date) - extract ( year from ss_numbers.date_of_birth_15)) * 12 * 30 +
    (extract ( month from ambiguous_date) - extract ( month from ss_numbers.date_of_birth_15)) * 30 +
    (extract ( day from ambiguous_date) - extract ( day from ss_numbers.date_of_birth_15)) as days,
    ambiguous_date, date_of_birth_15, result_numeric as psa
    
    FROM (
    SELECT distinct(psaresults.ss_number_id) as id

        FROM psaresults
        
        LEFT JOIN kreftreg_data
        
        ON psaresults.ss_number_id = kreftreg_data.ss_number_id
        
        WHERE kreftreg_data.ss_number_id is NULL) as tab, ss_numbers, psaresults

    WHERE tab.id =  psaresults.ss_number_id and 
        ss_numbers.ss_number_id = psaresults.ss_number_id and (extract ( year from ambiguous_date) - extract ( year from ss_numbers.date_of_birth_15)) >= 30
***
## SECOND STEP: Save the dataset

The datasets should be saved in the folder *dataset* with the names *cancer.csv* and  *nocancer.csv*

## THIRD STEP: Run the code
1. CD prostate-cancer-prediction
2. Launch the bash script *commands.sh*
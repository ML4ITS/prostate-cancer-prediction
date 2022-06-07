#FIRST STEP: QUERIES


***PATIENT WITH CANCER***:
***
    SELECT psaresults.ss_number_id, (extract ( year from ambiguous_date) - extract ( year from ss_numbers.date_of_birth_15)) as age,
    (extract( year from ambiguous_date) * 12 + extract( month from ambiguous_date)) as months,
    date_of_birth_15, result_numeric as psa
    
    FROM ss_numbers, psaresults, kreftreg_data
    
    WHERE diagnosis_date >= ambiguous_date and psaresults.ss_number_id = kreftreg_data.ss_number_id and
        ss_numbers.ss_number_id = psaresults.ss_number_id and (extract ( year from ambiguous_date) - extract ( year from ss_numbers.date_of_birth_15)) >= 50
***



***PATIENT WITHOUT CANCER***:
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


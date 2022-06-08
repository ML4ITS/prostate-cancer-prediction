import numpy as np

def get_random_numbers(layers, trial, min, max, element, int = True, desc = True):
    random_float_list = []
    for i in range(layers+1):
        element = element + str(i)
        if int is True:
            x = trial.suggest_int(element, min, max)
        else:
            x= trial.suggest_uniform(element, min, max)
        random_float_list.append(x)
    return -np.sort(np.array(random_float_list)) if desc else np.sort(np.array(random_float_list))

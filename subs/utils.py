import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
from scipy.spatial import distance_matrix

def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter Notebooks or Qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal Running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python Interpreter

status = isnotebook()

if status==True:
    from IPython.display import clear_output
    def clr_output():
        clear_output(wait=True)

    class color:
        PURPLE = '\033[95m'
        CYAN = '\033[96m'
        DARKCYAN = '\033[36m'
        BLUE = '\033[94m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
        END = '\033[0m'
        
elif status==False:
    import os
    def clr_output():
        os.system('cls')
        
    class color:
        PURPLE = ''
        CYAN = ''
        DARKCYAN = ''
        BLUE = ''
        GREEN = ''
        YELLOW = ''
        RED = ''
        BOLD = ''
        UNDERLINE = ''
        END = ''
    
def describe_DataFrame(df):
    desc = df.describe().T
    desc_count = pd.DataFrame(desc['count'])
    print(color.BOLD + color.RED+"\n *** Data Description: \n \n" + color.END, df.describe().T) 
    print(color.BOLD + color.RED+'\n *** Check for Missing Values!' + color.END)
    if desc_count.value_counts().shape[0]==1:
        print(color.BOLD + color.GREEN + "\n No Missing Data! Continue ... \n" + color.END)
    else:
        print(color.BOLD + color.YELLOW + "*** There Are Some Columns With Missing Values! Checking ...\n" + color.END)
        columns_list = []    
        for i in df.columns:
            if desc_count['count'].loc[i]!=desc['count'].value_counts().index[0]:
                columns_list.append(i)
        print(color.BOLD + color.YELLOW + '\n *** Here is the list of columns which have missing values! Check them please! \n' + color.END, columns_list)
    print(color.BOLD + color.RED + '\n *** Let\'s Print DataFrame Info! \n' + color.END)
    df.info()
    if df.dtypes.value_counts().shape[0]>1:
        print(color.BOLD + color.GREEN + '\n + We Have A Mixture of Datatypes in This DataFrame! \n' + color.END)   
    for val, cnt in df.dtypes.value_counts().iteritems():
        print(color.BOLD + color.GREEN + f'    - There are {cnt} columns in which we have got {val} datatype! \n' + color.END)
       
    features = list(df.columns)
    print(color.BOLD + color.RED + "*** The Original Dataset Shape is: " + color.GREEN + str(df.shape) + color.END)
    print(color.BOLD + color.RED + "*** The Number of Distinct Patients in the Dataset is: " + color.GREEN + str(len(pd.unique(df['subject#']))) + color.END)
    print(color.BOLD + color.RED + "*** List of Features: \n\n" + color.GREEN + str(features) + color.END)
    
    
def data_preprocess(input_data, train_test_ratio, random_seed=30):
    
    np.random.seed(random_seed)
    data_shuffled = input_data.sample(frac=1, replace=False, random_state=random_seed, axis=0)
    n_patients, n_regressors = input_data.shape
    n_train = int(n_patients*train_test_ratio)
    n_test = n_patients - n_train

    data_train = data_shuffled[0:n_train]

    mean_X = data_train.mean()
    std_X = data_train.std()
    mean_Y = mean_X['total_UPDRS']
    std_Y = std_X['total_UPDRS']
    X_norm = (data_shuffled - mean_X) / std_X
    Y_norm = X_norm['total_UPDRS']
    X_norm = X_norm.drop(['total_UPDRS','subject#', 'Jitter:DDP', 'Shimmer:DDA'], axis=1)

    regressors = list(X_norm.columns)
    print("The Regressors Are: \n", regressors)

    X_norm = X_norm.values
    Y_norm = Y_norm.values

    X_train = X_norm[0:n_train]
    X_test = X_norm[n_train:]
    y_train = Y_norm[0:n_train]
    y_test = Y_norm[n_train:]

    print(f"\nTraining Data Shape: {X_train.shape} \nTest Data Shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, mean_X, std_X, mean_Y, std_Y, regressors
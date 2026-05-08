import numpy as np
from imblearn.under_sampling import RandomUnderSampler

def balance_set(X, y, times, random_state=42):    
    """
    Use undersampling technique for balancing the set.
    """
    classes = np.unique(y)
    if len(classes) > 1:
        rus = RandomUnderSampler(sampling_strategy="auto", random_state=random_state)
        idx = np.arange(len(y)).reshape(-1, 1)
        idx_rus, _ = rus.fit_resample(idx, y)
        
        idx_rus = idx_rus.flatten()
        
        X_rus = X[idx_rus]
        y_rus = y[idx_rus]
        t_rus = times[idx_rus]
        
        return X_rus, y_rus, t_rus
    
    return X, y, times

def split_and_balance(X, y, times, train_prop=0.7, val_prop=0.15, random_state=42):
    """
    Split the set in train, validation and test set, 
    while balancing the train and  validation set
    """
    y = y.astype(int)
    
    n_windows = X.shape[0]
    
    train_end = int(n_windows*train_prop)
    val_end = train_end + int(n_windows*val_prop)
    
    X_train, y_train, t_train = X[:train_end], y[:train_end], times[:train_end]
    X_val, y_val, t_val = X[train_end:val_end], y[train_end:val_end], times[train_end:val_end]
    X_test, y_test, t_test = X[val_end:], y[val_end:], times[val_end:]
    
    X_train, y_train, t_train = balance_set(X_train, y_train, t_train, random_state)
    X_val, y_val, t_val = balance_set(X_val, y_val, t_val, random_state)

    sets = {}
    
    cls_train, ct_train = np.unique(y_train, return_counts=True)
    cls_val, ct_val = np.unique(y_val, return_counts=True)
    cls_test, ct_test = np.unique(y_test, return_counts=True)
    
    sets["Train"] = {"Size": X_train.shape[0], "Classes": cls_train, "Count": ct_train}
    sets["Val"] = {"Size": X_val.shape[0], "Classes": cls_val, "Count": ct_val}
    sets["Test"] = {"Size": X_test.shape[0], "Classes": cls_test, "Count": ct_test}


    print(f"{'Set':<10} | {'Size':<10} | {'Class 0':<10} | {'Class 1':<10}")
    print("-" * 50)
    for set, values in sets.items():
        if len(values['Classes']) > 1:
            print(f"{set:<10} | {values['Size']:<10} | {values['Count'][0]:<10} | {values['Count'][1]:<10}")
        
        elif 0 in values['Classes']:
            print(f"{set:<10} | {values['Size']:<10} | {values['Count'][0]:<10} | {0:<10}")
            
        elif 1 in values["Classes"]:
            print(f"{set:<10} | {values['Size']:<10} | {0:<10} | {values['Count'][1]:<10}")

    return (X_train, y_train, t_train), (X_val, y_val, t_val), (X_test, y_test, t_test)



def task2_preprocess(X, y_task2, times, range_tts=(60, 600)):
    """
    Reassign labels to prepare it for task2
    """
    # Remove full in crisis
    X_clean=X[y_task2 != 0]
    tts_clean=y_task2[y_task2 != 0]
    t_clean=times[y_task2 != 0]    

    y_pre = np.zeros(len(tts_clean))
    # Pre-crisis from range tts selected (defaults 1-10 minutes)
    y_pre[(tts_clean >= range_tts[0]) & (tts_clean <=range_tts[1])] = 1
    
    return X_clean, y_pre, t_clean
    


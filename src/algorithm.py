import math
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

def get_params(num_parms):
    if num_parms == "4":
        return {'eta': [math.exp(-7), math.exp(0)],
                'gamma': [math.exp(-10), math.exp(2)],
                'lambda': [math.exp(-7), math.exp(7)],
                'alpha': [math.exp(-7), math.exp(7)]}
    elif num_parms == "3":
        return {'eta': [math.exp(-7), math.exp(0)],
                'gamma': [math.exp(-10), math.exp(2)],
                'lambda': [math.exp(-7), math.exp(7)],}
    elif num_parms == "2":
        return {'eta': [math.exp(-7), math.exp(0)],
                'gamma': [math.exp(-10), math.exp(2)]}
    elif num_parms == "1":
        return {'eta': [math.exp(-7), math.exp(0)]}
    

def lgbm_model(num_parms, params):
    if num_parms == "4":
        return LGBMClassifier(learning_rate=params['eta'],
                              min_split_gain=params['gamma'],
                              reg_lambda=params['lambda'],
                              reg_alpha=params['alpha'],
                              verbose=-1)
    elif num_parms == "3":
        return LGBMClassifier(learning_rate=params['eta'],
                              min_split_gain=params['gamma'],
                              reg_lambda=params['lambda'],
                              verbose=-1)
    elif num_parms == "2":
        return LGBMClassifier(learning_rate=params['eta'],
                              min_split_gain=params['gamma'],
                              verbose=-1)
    elif num_parms == "1":
        return LGBMClassifier(learning_rate=params['eta'],
                              verbose=-1)
    

def xgb_model(num_parms, params):
    if num_parms == "4":
        return XGBClassifier(learning_rate=params['eta'],
                             min_split_gain=params['gamma'],
                             reg_lambda=params['lambda'],
                             reg_alpha=params['alpha'],
                             )
    elif num_parms == "3":
        return XGBClassifier(learning_rate=params['eta'],
                             min_split_gain=params['gamma'],
                             reg_lambda=params['lambda'],
                             )
    elif num_parms == "2":
        return XGBClassifier(learning_rate=params['eta'],
                             min_split_gain=params['gamma'],
                             )
    elif num_parms == "1":
        return XGBClassifier(learning_rate=params['eta'],
                             )
    
    
def rf_model(num_parms, params):
    return RandomForestClassifier(n_estimators=100,
                                  min_samples_split=params['gamma'],
                                  min_samples_leaf=params['lambda'],
                                  max_features=params['eta'],
                                  verbose=-1)

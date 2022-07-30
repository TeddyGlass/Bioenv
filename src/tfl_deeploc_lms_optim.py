import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from machine_learing.core.neural_network import NNClassifier
from machine_learing.core.suppor_tvector_machines import SVMClassifier
from machine_learing.core.optimazer import optuna_search, Objective
from machine_learing.core.trainer import Trainer
from machine_learing.settings.parameters import init_fit_params
import os
import json


if __name__ == '__main__':

    root = os.getcwd()

    f_names = [
        'DeepLocEmbedd_BERT_BFD.csv',
        'DeepLocEmbedd_Albert_BFD.csv',
        'DeepLocEmbedd_T5_BFD.csv',
        'DeepLocEmbedd_T5_FT.csv'
    ]
    out_dirs = [
        'DeepLocBERT',
        'DeepLocAlbert',
        'DeepLocT5',
        'DeepLocT5FT'
    ]

    for f_name, out_dir in zip(f_names, out_dirs):
        # parameters for nected cross validation 
        n_splits_ncv = 5
        seed_ncv = 1712
        # parameters for beyes optimization
        n_splits_optim = 5
        n_trials = 60
        n_jobs = 16
        seed_optim = 1307
        esr_optim = 100
        out_dir = os.path.join('../results/models/', out_dir)
        print('out_dir', out_dir)

        # load data set
        df = pd.read_csv(f'../data/DeepLoc/{f_name}')
        M_idx = df.iloc[:,1]=='M'
        S_idx = df.iloc[:,1]=='S'
        df = pd.concat([df[M_idx], df[S_idx]], axis=0)
        X = np.array(df.iloc[:,2:])
        y = np.concatenate([np.array([1]*sum(M_idx)), np.array([0]*sum(S_idx))]).flatten()
        print('X_shape', X.shape)
        print('y_shape', y.shape)

        # nested cross validation
        skf = StratifiedKFold(n_splits=n_splits_ncv, random_state=seed_ncv, shuffle=True) 
        idxes = list(skf.split(X, y))
        for i, (tr_idx, te_idx) in enumerate(idxes):
            if i >= 1:
                break
            # initialize fit params
            path_to_config = os.path.join(root, 'machine_learing/settings/config.ini')
            if os.path.exists(path_to_config):    
                PARAMS = {
                    # 'lgb':init_fit_params('lgb_params', path_to_config),
                    # 'xgb':init_fit_params('xgb_params', path_to_config),
                    # 'rf':init_fit_params('rf_params', path_to_config),
                    'svm':init_fit_params('svm_params', path_to_config),
                    # 'nn':init_fit_params('nn_params', path_to_config),
                }
            # start hyper paramneter optimization for each model
            for k, fit_params in PARAMS.items():
                if k == 'lgb':
                    model = Trainer(LGBMClassifier(**fit_params))
                elif k == 'xgb':
                    model = Trainer(XGBClassifier(**fit_params))
                elif k == 'rf':
                    model = Trainer(RandomForestClassifier(**fit_params))
                elif k == 'svm':
                    model = Trainer(SVMClassifier(**fit_params))
                elif k == 'nn':
                    model = Trainer(NNClassifier(**fit_params))
                model_type = type(model.get_model()).__name__
                
                print('-'*100)
                print(f'outer fold {i}, {out_dir}')
                print(f'Beyesian optimization for {model_type} model...')
                obj = Objective(
                    model,
                    path_to_config,
                    X[tr_idx],
                    y[tr_idx],
                    n_splits_optim,
                    esr_optim,
                    seed_optim
                )
                print('-'*100)
                best_params = optuna_search(obj, n_trials, n_jobs, seed_optim)
                print('completed')
                print('best params:', best_params)

                f_out = os.path.join(
                    out_dir,
                    os.path.join(model_type, f'{model_type}_{i}.json')
                    )
                with open(f_out, 'w') as f:
                    json.dump(best_params, f)

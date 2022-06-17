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
        'DeepPPIDescriptorAAindex.csv',
    ]
    out_dirs = [
        'DeepPPIAAindex'
    ]

    for f_name, out_dir in zip(f_names, out_dirs):
        # parameters for nected cross validation 
        n_splits_ncv = 5
        seed_ncv = 1712
        # parameters for beyes optimization
        n_splits_optim = 3
        n_trials = 60
        seed_optim = 1307
        esr_optim = 20
        # out dir path
        out_dir = os.path.join('../results/models/', out_dir)
        print('out_dir', out_dir)

        # load core data set for PPI classification
        df = pd.read_csv('../data/DeepPPI/DeepPPIAll.csv')
        protein_a = np.array(df['proteinA'])
        protein_b = np.array(df['proteinB'])
        y = np.array(df['interaction'])

        # load features
        df_feature = pd.read_csv(f'../data/DeepPPI/{f_name}')
        # pre-processing of protein features
        feature_dict = {
            Id: np.array(df_feature[df_feature.iloc[:,0]==Id].iloc[:,1:])
            for Id in df_feature.iloc[:,0].tolist()
        }
        feature_a, feature_b = [], []
        for a, b in zip(protein_a, protein_b):
            feature_a.append(feature_dict[a])
            feature_b.append(feature_dict[b])
        X_a, X_b = np.concatenate(feature_a), np.concatenate(feature_b)
        X = np.concatenate([X_a, X_b], axis=1)

        print('X_shape', X.shape)
        print('y_shape', y.shape)

        # nested cross validation
        skf = StratifiedKFold(n_splits=n_splits_ncv, random_state=seed_ncv, shuffle=True) 
        idxes = list(skf.split(X, y))
        for i, (tr_idx, te_idx) in enumerate(idxes):
            if i == 0:
                # initialize fit params
                path_to_config = os.path.join(root, 'machine_learing/settings/config.ini')
                if os.path.exists(path_to_config):    
                    PARAMS = {
                        'lgb':init_fit_params('lgb_params', path_to_config),
                        # 'xgb':init_fit_params('xgb_params', path_to_config),
                        'rf':init_fit_params('rf_params', path_to_config),
                        # 'svm':init_fit_params('svm_params', path_to_config),
                        'nn':init_fit_params('nn_params', path_to_config),
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
                    if k == 'lgb' or k == 'nn':
                        n_jobs = 8
                    else:
                        n_jobs = 32
                    # define objective function
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
                    # bayeseian optimization
                    best_params = optuna_search(obj, n_trials, n_jobs, seed_optim)
                    print('completed')
                    print('best params:', best_params)

                    f_out = os.path.join(
                        out_dir,
                        os.path.join(model_type, f'{model_type}_{i}.json')
                        )
                    with open(f_out, 'w') as f:
                        json.dump(best_params, f)

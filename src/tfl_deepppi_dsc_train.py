import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from machine_learing.settings.parameters import init_fit_params
from machine_learing.core.trainer import Trainer
from machine_learing.core.neural_network import NNClassifier
from machine_learing.core.suppor_tvector_machines import SVMClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from machine_learing.core.utils import roc_cutoff, evaluate_clf


if __name__ == '__main__':

    root = os.getcwd()

    f_names = [
        'DeepPPIDescriptorAAindex.csv'
    ]
    out_dirs = [
        'DeepPPIAAindex'
    ]

    for f_name, out_dir in zip(f_names, out_dirs):

       # parameters for nected cross validation 
        n_splits_ncv = 5
        seed_ncv = 1712
        early_stopping_rounds = 100
        # out dir path
        subroot = os.path.join('../results/models/', out_dir)
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
        skf_outer = StratifiedKFold(n_splits=n_splits_ncv, random_state=seed_ncv, shuffle=True) 
        outer_idxes = list(skf_outer.split(X, y))
        for i, (inner_idx, te_idx) in enumerate(outer_idxes):
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
            # update parameters
            if i == 0:
                for k, fit_params in PARAMS.items():
                    if k == 'lgb':
                        path_to_best_params = os.path.join(subroot, f'LGBMClassifier/LGBMClassifier_{i}.json')
                        with open (path_to_best_params, 'r') as f:
                            best_params = json.load(f)
                        fit_params.update(best_params)
                        fit_params['min_child_samples'] = int(fit_params['min_child_samples'])
                        print('train with parames:', fit_params)
                    elif k == 'xgb':
                        path_to_best_params = os.path.join(subroot, f'XGBClassifier/XGBClassifier_{i}.json')
                        with open (path_to_best_params, 'r') as f:
                            best_params = json.load(f)
                        fit_params.update(best_params)
                        print('train with parames:', fit_params)
                    elif k == 'rf':
                        path_to_best_params = os.path.join(subroot, f'RandomForestClassifier/RandomForestClassifier_{i}.json')
                        with open (path_to_best_params, 'r') as f:
                            best_params = json.load(f)
                        fit_params.update(best_params)
                        print('train with parames:', fit_params)
                    elif k == 'svm':
                        path_to_best_params = os.path.join(subroot, f'SVMClassifier/SVMClassifier_{i}.json')
                        with open (path_to_best_params, 'r') as f:
                            best_params = json.load(f)
                        fit_params.update(best_params)
                        print('train with parames:', fit_params)
                    elif k == 'nn':
                        path_to_best_params = os.path.join(subroot, f'NNClassifier/NNClassifier_{i}.json')
                        with open (path_to_best_params, 'r') as f:
                            best_params = json.load(f)
                        fit_params.update(best_params)
                        print('train with parames:', fit_params)
                    # inner validation
                    skf_inner = StratifiedKFold(n_splits=n_splits_ncv, random_state=seed_ncv, shuffle=True)
                    inner_idxes = list(skf_outer.split(X[inner_idx], y[inner_idx]))
                    METRICS = []
                    for j, (tr_idx, va_idx) in enumerate(inner_idxes):
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
                        print(f'Outer forld: {i}')
                        print('data', out_dir)
                        print(f'Training of {model_type} (fold {j+1}/{n_splits_ncv}) has begin with following parameters')
                        print(fit_params)
                        model.fit(
                                X[inner_idx][tr_idx],
                                y[inner_idx][tr_idx],
                                X[inner_idx][va_idx],
                                y[inner_idx][va_idx],
                                early_stopping_rounds
                            )
                        print(f'Training (fold {j+1}) has been been completed.')
                        print('-'*100)
                        # prediction and evaluation
                        y_pred = model.predict(X[inner_idx][va_idx])
                        cutoff = roc_cutoff(y[inner_idx][va_idx], y_pred)
                        metrics = evaluate_clf(y[inner_idx][va_idx], y_pred, cutoff) # check!
                        METRICS.append(pd.DataFrame(metrics))
                        print(metrics)
                        print('-'*100)
                        # saving learning curve
                        out_root = subroot + f'/{model_type}'
                        try:
                            model.get_learning_curve(
                                os.path.join(out_root, f'{model_type}_ij{i}{j}_learning_curve.png')
                                )
                        except Exception as e:
                            pass
                        # saving model
                        model = model.get_model()
                        if 'LGB' in model_type or 'XGB' in model_type or 'RandomForest' in model_type or 'SVM' in model_type: # LGB model or XGB model
                            out_name = os.path.join(
                                out_root,
                                f'{model_type}_ij{i}{j}_trainedmodel.pkl'
                                )
                            with open(out_name, 'wb') as f:
                                pickle.dump(model, f)
                            del model
                        elif 'NN' in model_type: # Keras model
                            # saving transfomer for feature standardization
                            if fit_params['standardization']:
                                transformer = model.get_transformer()
                                out_file = os.path.join(
                                    out_root,
                                    f'{model_type}_ij{i}{j}_transformer.pkl'
                                    )
                                with open(out_file, 'wb') as f:
                                    pickle.dump(transformer, f)
                            # saving weight
                            weight_name = os.path.join(
                                out_root,
                                f'{model_type}_ij{i}{j}_trainedweight.h5'
                                )
                            model.model.save(weight_name)
                            # saving model architecture
                            archit_name = os.path.join(
                                out_root,
                                f'{model_type}_i{i}_architecture.json'
                                )
                            json_string = model.model.to_json()
                            with open(archit_name, 'w') as f:
                                f.write(json_string)
                            del model
                    # saving metrics
                    out_neme = os.path.join(out_root, f'{model_type}_i{i}_metrics.csv')
                    df_METRICS = pd.concat(METRICS)
                    df_METRICS.to_csv(out_neme)
from machine_learing.core.trainer import Trainer
from sklearn.metrics import log_loss
from machine_learing.core.utils import root_mean_squared_error
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from machine_learing.core.neural_network import NNRegressor, NNClassifier
from machine_learing.core.suppor_tvector_machines import SVMClassifier, SVMRegressor
from machine_learing.settings.parameters import init_fit_params
import numpy as np
import optuna


class Objective:

    '''
    # Usage
    obj = Objective(LGBMRegressor(), X, y)
    study = optuna.create_study(
        sampler=optuna.samplers.RandomSampler(seed=123))
    study.optimize(obj, n_trials=10, n_jobs=-1)
    '''

    def __init__(self, model, path_to_config, x, y, n_splits, early_stopping_rounds, random_state):
        self.model = model
        self.path_to_config = path_to_config
        self.model_type = type(self.model.get_model()).__name__
        self.x = x
        self.y = y
        self.n_splits = n_splits
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
    
    def __call__(self, trial):
        if 'LGBM' in self.model_type:
            self.SPACE = {
                'num_leaves': trial.suggest_int('num_leaves', 8, 31),
                'subsample': trial.suggest_uniform('subsample', 0.60, 0.80),
                'colsample_bytree': trial.suggest_uniform(
                    'colsample_bytree', 0.60, 0.80),
                'bagging_freq': trial.suggest_int(
                    'bagging_freq', 1, 51, 5),
                'min_child_weight': trial.suggest_loguniform(
                    'min_child_weight', 1, 32),
                'min_child_samples': int(trial.suggest_discrete_uniform(
                    'min_child_samples', 5, 50, 5)),
                'min_split_gain': trial.suggest_loguniform(
                    'min_split_gain', 1e-5, 1e-1),
                'learning_rate': 0.05,
                'n_estimators': init_fit_params('lgb_params', self.path_to_config)['n_estimators'],
                'random_state': init_fit_params('lgb_params', self.path_to_config)['random_state'],
                'max_depth':init_fit_params('lgb_params', self.path_to_config)['max_depth']
            }
        elif 'XGB' in self.model_type:
            self.SPACE = {
                'subsample': trial.suggest_uniform(
                    'subsample', 0.65, 0.85),
                'colsample_bytree': trial.suggest_uniform(
                    'colsample_bytree', 0.65, 0.80),
                'gamma': trial.suggest_loguniform(
                    'gamma', 1e-8, 1.0),
                'min_child_weight': trial.suggest_loguniform(
                    'min_child_weight', 1, 32),
                'learning_rate': 0.05,
                'max_depth':init_fit_params('xgb_params', self.path_to_config)['max_depth'],
                'n_estimators': init_fit_params('xgb_params', self.path_to_config)['n_estimators'],
                'random_state': init_fit_params('xgb_params', self.path_to_config)['random_state']
            }
        elif 'NN' in self.model_type:
            self.SPACE = {
                "input_dropout": trial.suggest_uniform(
                    "input_dropout", 0.001, 0.01),
                "hidden_layers": trial.suggest_int(
                    "hidden_layers", 3, 5),
                'hidden_units': int(trial.suggest_discrete_uniform(
                    'hidden_units', 256, 1024, 256)),
                'hidden_dropout': trial.suggest_uniform(
                    'hidden_dropout', 0.001, 0.01),
                'batch_norm': trial.suggest_categorical(
                    'batch_norm', ['before_act', 'non']),
                'batch_size': int(trial.suggest_discrete_uniform(
                    'batch_size', 32, 128, 16)),
                'learning_rate': 1e-4,
                'epochs': init_fit_params('nn_params', self.path_to_config)['epochs']
            }
        elif 'RandomForest' in self.model_type:
            self.SPACE = {
                'max_depth':trial.suggest_int(
                    'max_depth', 4, 10),
                'min_samples_split':trial.suggest_int(
                    'min_samples_split', 16, 64),
                'max_features': trial.suggest_uniform(
                    'max_features', 0.5, 0.8),
                'random_state': init_fit_params('rf_params', self.path_to_config)['random_state']
            }
        elif 'SVM' in self.model_type:
            self.SPACE = {
                'C': trial.suggest_loguniform(
                   'C', 1e-2, 1e3 ),
                'gamma': trial.suggest_loguniform(
                   'gamma', 1e-2, 1e3 ),
                'random_state': init_fit_params('svm_params', self.path_to_config)['random_state']
            }
        # splitting type of cross validation
        if 'Classifier' in self.model_type or 'SVC' in self.model_type:
            cv = StratifiedKFold(n_splits=self.n_splits, random_state=self.random_state, shuffle=True)
        elif 'Regressor' in self.model_type or 'SVR' in self.model_type:
            cv = KFold(n_splits=self.n_splits, random_state=self.random_state, shuffle=True)
        # validate average loss in K-Fold CV on a set of parameters.
        LOSS = []
        for tr_idx, va_idx in cv.split(self.x, self.y):
            if 'LGBM' in self.model_type:
                if 'Classifier' in self.model_type:
                    model_ = Trainer(LGBMClassifier(**self.SPACE))
                elif 'Regressor' in self.model_type:
                    model_ = Trainer(LGBMRegressor(**self.SPACE))
            elif 'XGB' in self.model_type:
                if 'Classifier' in self.model_type:
                    model_ = Trainer(XGBClassifier(**self.SPACE))
                elif 'Regressor' in self.model_type:
                    model_ = Trainer(XGBRegressor(**self.SPACE))
            elif 'NN' in self.model_type:
                if 'Classifier' in self.model_type:
                    model_ = Trainer(NNClassifier(**self.SPACE))
                elif 'Regressor' in self.model_type:
                    model_ = Trainer(NNRegressor(**self.SPACE))
            elif 'RandomForest' in self.model_type:
                if 'Classifier' in self.model_type:
                    model_ = Trainer(RandomForestClassifier(**self.SPACE))
                elif 'Regressor' in self.model_type:
                    model_ = Trainer(RandomForestRegressor(**self.SPACE))
            elif 'SVM' in self.model_type:
                if 'Classifier' in self.model_type:
                    model_ = Trainer(SVMClassifier(**self.SPACE))
                elif 'Regressor' in self.model_type:
                    model_ = Trainer(SVMRegressor(**self.SPACE))
            model_.fit(
                self.x[tr_idx],
                self.y[tr_idx],
                self.x[va_idx],
                self.y[va_idx],
                self.early_stopping_rounds
            )
            y_pred = model_.predict(self.x[va_idx])  # best_iteration
            if 'Classifier' in self.model_type:
                loss = log_loss(self.y[va_idx], y_pred)
            elif 'Regressor' in self.model_type:
                loss = root_mean_squared_error(self.y[va_idx], y_pred)
            LOSS.append(loss)
        return np.mean(LOSS)

            
def optuna_search(obj, n_trials, n_jobs, random_state):
    study = optuna.create_study(
        sampler=optuna.samplers.RandomSampler(seed=random_state))
    study.optimize(obj, n_trials=n_trials, n_jobs=n_jobs)
    return study.best_params


if __name__ == "__main__":
    pass
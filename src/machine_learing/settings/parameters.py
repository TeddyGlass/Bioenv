import configparser


def init_fit_params(section, path_to_config):
    config = configparser.ConfigParser()
    config.read(path_to_config)
    FIT_PARAMS = {}
    if section == 'lgb_params':
        FIT_PARAMS['learning_rate'] = float(config.get(section, 'learning_rate'))
        FIT_PARAMS['n_estimators'] = int(config.get(section, 'n_estimators'))
        FIT_PARAMS['max_depth'] = int(config.get(section, 'max_depth'))
        FIT_PARAMS['num_leaves'] = int(config.get(section, 'num_leaves'))
        FIT_PARAMS['subsample'] = float(config.get(section, 'subsample'))
        FIT_PARAMS['colsample_bytree'] = float(config.get(section, 'colsample_bytree'))
        FIT_PARAMS['bagging_freq'] = int(config.get(section, 'bagging_freq'))
        FIT_PARAMS['min_child_weight'] = float(config.get(section, 'min_child_weight'))
        FIT_PARAMS['min_child_samples'] = int(config.get(section, 'min_child_samples'))
        FIT_PARAMS['min_split_gain'] = float(config.get(section, 'min_split_gain'))
        FIT_PARAMS['n_jobs'] = int(config.get(section, 'n_jobs'))
        FIT_PARAMS['random_state'] = int(config.get(section, 'random_state'))
    elif section == 'xgb_params':
        FIT_PARAMS['learning_rate'] = float(config.get(section, 'learning_rate'))
        FIT_PARAMS['n_estimators'] = int(config.get(section, 'n_estimators'))
        FIT_PARAMS['max_depth'] = int(config.get(section, 'max_depth'))
        FIT_PARAMS['subsample'] = float(config.get(section, 'subsample'))
        FIT_PARAMS['colsample_bytree'] = float(config.get(section, 'colsample_bytree'))
        FIT_PARAMS['gamma'] = float(config.get(section, 'gamma'))
        FIT_PARAMS['min_child_weight'] = float(config.get(section, 'min_child_weight'))
        FIT_PARAMS['n_jobs'] = int(config.get(section, 'n_jobs'))
        FIT_PARAMS['random_state'] = int(config.get(section, 'random_state'))
    elif section == 'rf_params':
        FIT_PARAMS['max_depth'] = int(config.get(section, 'max_depth'))
        FIT_PARAMS['min_samples_split'] = float(config.get(section, 'min_samples_split'))
        FIT_PARAMS['max_features'] = float(config.get(section, 'max_features'))
        FIT_PARAMS['n_jobs'] = int(config.get(section, 'n_jobs'))
        FIT_PARAMS['random_state'] = int(config.get(section, 'random_state'))
    elif section == 'svm_params':
        FIT_PARAMS['kernel'] = config.get(section, 'kernel')
        FIT_PARAMS['C'] = float(config.get(section, 'C'))
        FIT_PARAMS['gamma'] = float(config.get(section, 'gamma'))
        FIT_PARAMS['random_state'] = int(config.get(section, 'random_state'))
    elif section == 'nn_params':
        FIT_PARAMS['standardization'] = config.getboolean(section, 'standardization')
        FIT_PARAMS['learning_rate'] = float(config.get(section, 'learning_rate'))
        FIT_PARAMS['epochs'] = int(config.get(section, 'epochs'))
        FIT_PARAMS['hidden_units'] = int(config.get(section, 'hidden_units'))
        FIT_PARAMS['batch_size'] = int(config.get(section, 'batch_size'))
        FIT_PARAMS['input_dropout'] = float(config.get(section, 'input_dropout'))
        FIT_PARAMS['hidden_dropout'] = float(config.get(section, 'hidden_dropout'))
        FIT_PARAMS['hidden_layers'] = int(config.get(section, 'hidden_layers'))
        FIT_PARAMS['batch_norm'] = config.get(section, 'batch_norm')
    return FIT_PARAMS
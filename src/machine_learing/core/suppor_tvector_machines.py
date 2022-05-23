from sklearn.svm import SVC, SVR
from sklearn.preprocessing import QuantileTransformer


class SVMClassifier():

    def __init__(self, kernel='rbf', C=1.0, gamma='auto'):
        self.params = {
            'kernel': kernel,
            'C':C,
            'gamma':gamma,
            'probability':True
        }
    
    def fit(self, X, y):
        self.transformer = QuantileTransformer(
            n_quantiles=100,
            random_state=0, 
            output_distribution='normal'
            )
        X = self.transformer.fit_transform(X) 
        self.model = SVC(**self.params)
        self.model.fit(X, y)
    
    def predict_proba(self, X):
        X = self.transformer.transform(X)
        return self.model.predict_proba(X)
    

class SVMRegressor():

    def __init__(self, kernel='rbf', C=1.0, gamma='auto'):
        self.params = {
            'kernel': kernel,
            'C':C,
            'gamma':gamma,
        }

    def fit(self, X, y):
        self.transformer = QuantileTransformer(
            n_quantiles=100,
            random_state=0, 
            output_distribution='normal'
            )
        X = self.transformer.fit_transform(X) 
        self.model = SVR(**self.params)
        self.model.fit(X, y)
    
    def predict(self, X):
        X = self.transformer.transform(X)
        return self.model.predict(X)
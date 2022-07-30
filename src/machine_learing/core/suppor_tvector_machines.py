from sklearn.svm import SVC, SVR
from sklearn.preprocessing import QuantileTransformer, StandardScaler


class SVMClassifier():

    def __init__(self, kernel='rbf', C=1.0, gamma='auto', random_state=0):
        self.params = {
            'kernel': kernel,
            'C':C,
            'gamma':gamma,
            'cache_size':10000,
            'random_state':random_state,
            'probability': False
        }
    
    def fit(self, X, y):
        # self.transformer = QuantileTransformer(
        #     n_quantiles=100,
        #     random_state=0,
        #     output_distribution='normal'
        #     )
        self.transformer = StandardScaler()
        X = self.transformer.fit_transform(X) 
        self.model = SVC(**self.params)
        self.model.fit(X, y)
    
    def predict_proba(self, X):
        X = self.transformer.transform(X)
        return self.model.predict_proba(X)

    def predict_score(self, X):
        X = self.transformer.transform(X)
        return self.model.decision_function(X)
    

class SVMRegressor():

    def __init__(self, kernel='rbf', C=1.0, gamma='auto', random_state=0):
        self.params = {
            'kernel': kernel,
            'C':C,
            'gamma':gamma,
            'random_state':random_state,
        }

    def fit(self, X, y):
        # self.transformer = QuantileTransformer(
        #     n_quantiles=100,
        #     random_state=0, 
        #     output_distribution='normal'
        #     )
        self.transformer = StandardScaler()
        X = self.transformer.fit_transform(X) 
        self.model = SVR(**self.params)
        self.model.fit(X, y)
    
    def predict(self, X):
        X = self.transformer.transform(X)
        return self.model.predict(X)
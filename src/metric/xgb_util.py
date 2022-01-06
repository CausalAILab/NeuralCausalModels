import numpy as np
import pandas as pd
from xgboost import XGBClassifier


class XGBProb:
    def __init__(self, Y, X):
        if isinstance(X, str):
            X = [X]
        if isinstance(Y, str):
            self.binary = True
            self.Y = Y
        else:
            if len(Y) == 1:
                self.binary = True
                self.Y = Y[0]
            else:
                self.binary = False
                self.Y = sorted(Y)
        self.X = sorted(X)
        self.xgbr = None

    def fit(self, dat, regval=None):
        labels = self._transform_multi_labels(dat)

        if regval is None:
            regval = 100.0 / np.sqrt(len(dat))

        self.xgbr = XGBClassifier(n_estimators=200, max_depth=10,
                                  reg_alpha=regval / 2, reg_lambda=regval,
                                  objective='binary:logistic' if self.binary else 'multi:softprob',
                                  eval_metric='logloss' if self.binary else 'mlogloss',
                                  nthread=-1, use_label_encoder=False)
        self.xgbr.fit(dat[self.X], labels)

    def predict(self, y, x):
        if self.xgbr is None:
            raise UnfittedModelException("Model must be fit on training data before being used for predictions.")
        if not isinstance(y, dict):
            raise InvalidInputException("Value for Y must be stored in a dict.")
        for Y in self.Y:
            if Y not in y:
                raise InvalidInputException("Every value of Y must be specified.")
        if not isinstance(x, dict):
            raise InvalidInputException("Value for X must be stored in a dict.")
        for X in self.X:
            if X not in x:
                raise InvalidInputException("Every value of X must be specified.")

        index = self._transform_multi_labels(y)
        return self.xgbr.predict_proba(pd.Series(x).to_frame().T.sort_index(axis=1))[0, index]

    def predict_dat(self, dat, y_fix=None, x_fix=None):
        if self.xgbr is None:
            raise UnfittedModelException("Model must be fit on training data before being used for predictions.")

        new_dat_y = dat[self.Y].copy(deep=True)
        new_dat_x = dat[self.X].copy(deep=True)
        if y_fix is not None:
            if not isinstance(y_fix, dict):
                raise InvalidInputException("Fixed values for Y must be stored in a dict.")
            for Y in y_fix:
                new_dat_y[Y] = y_fix[Y]
        if x_fix is not None:
            if not isinstance(x_fix, dict):
                raise InvalidInputException("Fixed values for X must be stored in a dict.")
            for X in x_fix:
                new_dat_x[X] = x_fix[X]

        new_dat_x = np.ascontiguousarray(new_dat_x)

        index = self._transform_multi_labels(new_dat_y)
        preds = self.xgbr.predict_proba(new_dat_x)
        return preds[np.arange(len(preds)), index]

    def _transform_multi_labels(self, row):
        if self.binary:
            return row[self.Y]

        val = 0
        for i, y in enumerate(self.Y):
            val += (2 ** i) * row[y]
        return val


class CountingProb:
    def __init__(self, X):
        if isinstance(X, str):
            X = [X]
        self.X = sorted(X)
        self.prob_dict = None

    def fit(self, dat):
        self.prob_dict = (dat.groupby(self.X).size() / len(dat)).to_dict()

    def predict(self, x):
        if self.prob_dict is None:
            raise UnfittedModelException("Model must be fit on training data before being used for predictions.")
        if not isinstance(x, dict):
            raise InvalidInputException("Value for X must be stored in a dict.")

        key = tuple([x[var] for var in self.X])
        if len(key) == 1:
            key = key[0]
        if key in self.prob_dict:
            return self.prob_dict[key]
        else:
            return 0

    def _predict_dat_helper(self, row):
        key = tuple(row[var] for var in self.X)
        if len(key) == 1:
            key = key[0]
        if key in self.prob_dict:
            return self.prob_dict[key]
        else:
            return 0

    def predict_dat(self, dat):
        return dat.apply(self._predict_dat_helper, axis=1)


class UnfittedModelException(Exception):
    pass


class InvalidInputException(Exception):
    pass


if __name__ == "__main__":
    pyx0z0 = 0.8
    pyx0z1 = 0.6
    pyx1z0 = 0.4
    pyx1z1 = 0.2
    x_list = np.random.binomial(1, 0.5, 100000)
    z_list = np.random.binomial(1, 0.5, 100000)
    y_list = []
    y2_list = []
    for i in range(len(x_list)):
        x = x_list[i]
        z = z_list[i]
        if x == 0 and z == 0:
            y_list.append(np.random.binomial(1, pyx0z0, 1))
            y2_list.append(np.random.binomial(1, pyx0z0, 1))
        elif x == 0 and z == 1:
            y_list.append(np.random.binomial(1, pyx0z1, 1))
            y2_list.append(np.random.binomial(1, pyx0z1, 1))
        elif x == 1 and z == 0:
            y_list.append(np.random.binomial(1, pyx1z0, 1))
            y2_list.append(np.random.binomial(1, pyx1z0, 1))
        else:
            y_list.append(np.random.binomial(1, pyx1z1, 1))
            y2_list.append(np.random.binomial(1, pyx1z1, 1))
    y_list = np.array(y_list).squeeze()
    y2_list = np.array(y2_list).squeeze()
    df = pd.DataFrame(np.vstack((x_list, z_list, y_list, y2_list)).T)
    df.columns = ['X', 'Z', 'Y', 'Y2']

    xgb_model = XGBProb(['Y', 'Y2'], ['X', 'Z'])
    xgb_model.fit(df)
    print(xgb_model.predict_dat(df, y_fix={'Y': 0, 'Y2': 0}, x_fix={'X': 0, 'Z': 0}))
    print(xgb_model.predict({'Y': 0, 'Y2': 0}, {'X': 0, 'Z': 0}))
    #df['*'] = 0
    #pv = (df.groupby(['Y', 'Y2', 'X', 'Z'])['*'].count() / len(df)).rename('P(V)').to_frame().reset_index()
    #pcond = (df.groupby(['X', 'Z'])['*'].count() / len(df)).rename('P(cond)').to_frame().reset_index()
    #merged = pd.merge(pv, pcond)
    #print(merged["P(V)"] / merged["P(cond)"])

    count_model_all = CountingProb(['Y', 'Y2', 'X', 'Z'])
    count_model_all.fit(df)
    count_model_cond = CountingProb(['X', 'Z'])
    count_model_cond.fit(df)
    print(df)
    print(count_model_all.predict_dat(df))
    print(count_model_all.predict({'Y': 0, 'Y2': 0, 'X': 0, 'Z': 0}) /
          count_model_cond.predict({'X': 0, 'Z': 0}))

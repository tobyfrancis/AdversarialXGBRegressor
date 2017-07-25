import sys
import os
import ctypes
import collections
import re
import numpy as np
import scipy.sparse

from xgboost import rabit
from xgboost import callback
from xgboost.libpath import find_lib_path
from xgboost.sklearn import XGBModel
from xgboost.compat import (STRING_TYPES, PY3, DataFrame, py_str, PANDAS_INSTALLED, XGBStratifiedKFold,
                     SKLEARN_INSTALLED, XGBModelBase, XGBClassifierBase, XGBRegressorBase, XGBLabelEncoder)
from adversarial_training import *
class XGBAdversarialRegressor(XGBModel):
    # pylint: disable=too-many-arguments, too-many-instance-attributes, invalid-name
    """Implementation of the Scikit-Learn API for XGBoost.
    ----
    A custom objective function can be provided for the ``objective``
    parameter. In this case, it should have the signature
    ``objective(y_true, y_pred) -> grad, hess``:
    y_true: array_like of shape [n_samples]
        The target values
    y_pred: array_like of shape [n_samples]
        The predicted values
    grad: array_like of shape [n_samples]
        The value of the gradient for each sample point.
    hess: array_like of shape [n_samples]
        The value of the second derivative for each sample point
    """

    def __init__(self, max_depth=3, learning_rate=1.0, n_estimators=1000,
                 silent=True, objective="reg:linear", booster='gbtree',
                 n_jobs=1, nthread=4, gamma=0, min_child_weight=1, max_delta_step=0,
                 subsample=1, colsample_bytree=1, colsample_bylevel=1,
                 reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                 base_score=0.5, random_state=0, seed=0, missing=None, **kwargs):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.silent = silent
        self.objective = objective
        self.booster = booster
        self.nthread = nthread
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.base_score = base_score
        self.missing = missing if missing is not None else np.nan
        self.kwargs = kwargs

        self._Booster = None

        self.seed = seed
        self.random_state = random_state
        self.nthread = nthread
        self.n_jobs = n_jobs

    def get_booster(self):
        """Get the underlying xgboost Booster of this model.
        This will raise an exception when fit was not called
        Returns
        -------
        booster : a xgboost booster of underlying model
        """
        if not self.booster:
            raise XGBoostError('need to call fit beforehand')
        return self.booster

    def fit(self, X, y, eval_set=None, eval_metric=None,
            early_stopping_rounds=None, verbose=True):
        # pylint: disable=missing-docstring,invalid-name,attribute-defined-outside-init
        if callable(self.objective):
            obj = _objective_decorator(self.objective)
            # Use default value. Is it really not used ?
            xgb_options["objective"] = "binary:logistic"
        else:
            obj = None
        prior_copy = X
        prior = DMatrix(X)
        if len(y.shape) == 1:
            distribution = np.concatenate((np.expand_dims(y,axis=-1),X),axis=-1)
        elif len(y.shape) == 2:
            distribution = np.concatenate((y,X),axis=-1)
        else: 
            raise ValueError('Numpy Array input must have a maximum of 2 Dimensions.')

        distribution = DMatrix(distribution)
        posterior = y

        
        evals_result = {}
        if eval_set is not None:
            evals = list(DMatrix(x[0], label=x[1], missing=self.missing) for x in eval_set)
            evals = list(zip(evals, ["validation_{}".format(i) for i in
                                     range(len(evals))]))
        else:
            evals = ()

        params = self.get_xgb_params()

        if callable(self.objective):
            obj = _objective_decorator(self.objective)
            params["objective"] = "reg:linear"
        else:
            obj = None

        feval = eval_metric if callable(eval_metric) else None
        if eval_metric is not None:
            if callable(eval_metric):
                eval_metric = None
            else:
                params.update({'eval_metric': eval_metric})

        self.booster = train(params, prior, prior_copy, distribution, posterior,
                            self.n_estimators, evals=evals,
                            early_stopping_rounds=early_stopping_rounds,
                            evals_result=evals_result, obj=None, feval=feval,
                            verbose_eval=verbose)
        if evals_result:
            for val in evals_result.items():
                evals_result_key = list(val[1].keys())[0]
                evals_result[val[0]][evals_result_key] = val[1][evals_result_key]
            self.evals_result_ = evals_result

        if early_stopping_rounds is not None:
            self.best_score = self.booster.best_score
            self.best_iteration = self.booster.best_iteration
            self.best_ntree_limit = self.booster.best_ntree_limit
        return self

    def predict(self, data, output_margin=False, ntree_limit=0):
        # pylint: disable=missing-docstring,invalid-name
        test_dmatrix = DMatrix(data, missing=self.missing)
        return self.get_booster().predict(test_dmatrix,
                                        output_margin=output_margin,
                                        ntree_limit=ntree_limit)

    def apply(self, X, ntree_limit=0):
        """Return the predicted leaf every tree for each sample.
        Parameters
        ----------
        X : array_like, shape=[n_samples, n_features]
            Input features matrix.
        ntree_limit : int
            Limit number of trees in the prediction; defaults to 0 (use all trees).
        Returns
        -------
        X_leaves : array_like, shape=[n_samples, n_trees]
            For each datapoint x in X and for each tree, return the index of the
            leaf x ends up in. Leaves are numbered within
            ``[0; 2**(self.max_depth+1))``, possibly with gaps in the numbering.
        """
        test_dmatrix = DMatrix(X, missing=self.missing)
        return self.get_booster().predict(test_dmatrix,
                                          pred_leaf=True,
                                          ntree_limit=ntree_limit)

    def predict_proba(self, data, output_margin=False, ntree_limit=0):
        ''' 
        Returns
        ------- 
        predict_proba: array of shape = [n_features]
            (probability that the estimation lies within probability distribution)
        '''
        test_dmatrix = DMatrix(X, missing=self.missing)
        return self.get_booster().predict_proba(test_dmatrix,
                                          pred_leaf=True,
                                          ntree_limit=ntree_limit)

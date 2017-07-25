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
from xgboost.core import Booster, DMatrix, c_array, _check_call, c_str, ctypes2numpy
from xgboost.compat import (STRING_TYPES, PY3, DataFrame, py_str, PANDAS_INSTALLED, XGBStratifiedKFold,
                     SKLEARN_INSTALLED, XGBModelBase, XGBClassifierBase, XGBRegressorBase, XGBLabelEncoder)

def _load_lib():
    """Load xgboost Library."""
    lib_path = find_lib_path()
    if len(lib_path) == 0:
        return None
    lib = ctypes.cdll.LoadLibrary(lib_path[0])
    lib.XGBGetLastError.restype = ctypes.c_char_p
    return lib

# load the XGBoost library globally
_LIB = _load_lib()
c_bst_ulong = ctypes.c_uint64

class AdversarialBooster(Booster):
    def __init__(self, params=None, cache=(), model_file=None):
        # pylint: disable=invalid-name
        """Initialize the Booster.
        Parameters
        ----------
        params : dict
            Parameters for boosters.
        cache : list
            List of cache items.
        model_file : string
            Path to the model file.
        """
        self.generator_feature_names = None
        self.discriminator_feature_names = None

        dmats = c_array(ctypes.c_void_p, [d.handle for d in cache])
        self.discriminator_handle = ctypes.c_void_p()
        self.handle = ctypes.c_void_p()
        self.discriminator_handle = ctypes.c_void_p()
        _check_call(_LIB.XGBoosterCreate(dmats, c_bst_ulong(len(cache)),
                                         ctypes.byref(self.discriminator_handle)))
        _check_call(_LIB.XGBoosterCreate(dmats, c_bst_ulong(len(cache)),
                                         ctypes.byref(self.handle)))
        self.set_param({'seed': 0})
        self.set_param(params or {})
        if model_file is not None:
            self.load_model(model_file)

    def _fobj_gen(self,pred,_):
        g_fake = self._discriminate(pred)
        grad = g_fake - np.ones(len(g_fake))
        hess = g_fake * (1.0 - g_fake)
        return grad,hess

    def _fobj_disc(self,pred,dtrain):
        d_real = self._discriminate(dtrain)
        d_fake = self._discriminate(pred)
        grad = d_real - np.ones(len(d_real)) + d_fake
        hess = d_real*(1.0-d_real) + d_fake*(1.0-d_fake)
        return grad,hess

    def _sigmoid(self,x):
        return 1 / np.maximum(1 + np.exp(-x),1e-9)

    def _L2(self,pred,posterior):
        return np.sum(np.mean(np.sqrt(np.square(pred - posterior)),axis=-1))

    def update(self, prior, prior_copy, distribution, posterior, iteration, fobj=None):
        """
        Update for one iteration, with objective function calculated internally.
        Parameters
        ----------
        dtrain : DMatrix
            Training data.
        iteration : int
            Current iteration number.
        fobj : function
            Customized objective function.
        """
        pred = self.predict(prior)
        pred = np.expand_dims(pred.flatten(),axis=-1)
        dpred = DMatrix(np.concatenate((pred,prior_copy),axis=-1))
        grad, hess = self._fobj_disc(dpred, distribution)
        self.boost_discriminator(distribution, grad, hess)
 
        grad, hess = self._fobj_gen(dpred,distribution)
        loss = self._L2(pred.flatten(),posterior)
        self.boost_generator(prior, grad, hess)
        print('Iteration {}: Loss - {}'.format(iteration,loss))

    def boost_discriminator(self, distribution, grad, hess):
        if len(grad) != len(hess):
            raise ValueError('grad / hess length mismatch: {} / {}'.format(len(grad), len(hess)))
        if not isinstance(distribution, DMatrix):
            raise TypeError('invalid training matrix: {}'.format(type(distribution).__name__))
        self._validate_discriminator_features(distribution)

        _check_call(_LIB.XGBoosterBoostOneIter(self.discriminator_handle,distribution.handle,
                                               c_array(ctypes.c_float, grad),
                                               c_array(ctypes.c_float, hess),
                                               c_bst_ulong(len(grad))))

    def boost_generator(self, distribution, grad, hess):
        if len(grad) != len(hess):
            raise ValueError('grad / hess length mismatch: {} / {}'.format(len(grad), len(hess)))
        if not isinstance(distribution, DMatrix):
            raise TypeError('invalid training matrix: {}'.format(type(distribution).__name__))
        #self._validate_generator_features(distribution)

        _check_call(_LIB.XGBoosterBoostOneIter(self.handle, distribution.handle,
                                               c_array(ctypes.c_float, grad),
                                               c_array(ctypes.c_float, hess),
                                               c_bst_ulong(len(grad))))

    def _discriminate(self, data, output_margin=False, ntree_limit=0, pred_leaf=False,
                pred_contribs=False):
        option_mask = 0x00
        if output_margin:
            option_mask |= 0x01
        if pred_leaf:
            option_mask |= 0x02
        if pred_contribs:
            option_mask |= 0x04

        self._validate_discriminator_features(data)

        length = c_bst_ulong()
        disc = ctypes.POINTER(ctypes.c_float)()
        _check_call(_LIB.XGBoosterPredict(self.discriminator_handle, data.handle,
                                          ctypes.c_int(option_mask),
                                          ctypes.c_uint(ntree_limit),
                                          ctypes.byref(length),
                                          ctypes.byref(disc)))
        disc = self._sigmoid(ctypes2numpy(disc, length.value, np.float32))
        return disc

    def predict_proba(self, data, output_margin=False, ntree_limit=0, pred_leaf=False,
                pred_contribs=False):
        option_mask = 0x00
        if output_margin:
            option_mask |= 0x01
        if pred_leaf:
            option_mask |= 0x02
        if pred_contribs:
            option_mask |= 0x04

        self._validate_generator_features(data)

        length = c_bst_ulong()
        preds = ctypes.POINTER(ctypes.c_float)()
        _check_call(_LIB.XGBoosterPredict(self.handle, data.handle,
                                          ctypes.c_int(option_mask),
                                          ctypes.c_uint(ntree_limit),
                                          ctypes.byref(length),
                                          ctypes.byref(preds)))
        preds = ctypes2numpy(preds, length.value, np.float32)
        preds = DMatrix(np.expand_dims(pred.flatten(),axis=-1))
        pred_probas = ctypes.POINTER(ctypes.c_float)()
        _check_call(_LIB.XGBoosterPredict(self.discriminator_handle, preds.handle,
                                          ctypes.c_int(option_mask),
                                          ctypes.c_uint(ntree_limit),
                                          ctypes.byref(length),
                                          ctypes.byref(preds)))
        pred_probas = ctypes2numpy(pred_probas, length.value, np.float32)
        return self._sigmoid(pred_probas)

    def predict(self, data, output_margin=False, ntree_limit=0, pred_leaf=False,
                pred_contribs=False):
        option_mask = 0x00
        if output_margin:
            option_mask |= 0x01
        if pred_leaf:
            option_mask |= 0x02
        if pred_contribs:
            option_mask |= 0x04

        #self._validate_generator_features(data)

        length = c_bst_ulong()
        preds = ctypes.POINTER(ctypes.c_float)()
        _check_call(_LIB.XGBoosterPredict(self.handle, data.handle,
                                          ctypes.c_int(option_mask),
                                          ctypes.c_uint(ntree_limit),
                                          ctypes.byref(length),
                                          ctypes.byref(preds)))
        preds = ctypes2numpy(preds, length.value, np.float32)
        return preds

    def set_param(self, params, value=None):
        """Set parameters into the Booster.
        Parameters
        ----------
        params: dict/list/str
           list of key,value paris, dict of key to value or simply str key
        value: optional
           value of the specified parameter, when params is str key
        """
        if isinstance(params, collections.Mapping):
            params = params.items()
        elif isinstance(params, STRING_TYPES) and value is not None:
            params = [(params, value)]
        for key, val in params:
            _check_call(_LIB.XGBoosterSetParam(self.handle, c_str(key), c_str(str(val))))

    def _validate_generator_features(self, data):
        """
        Validate Booster and data's feature_names are identical.
        Set feature_names and feature_types from DMatrix
        """
        if self.generator_feature_names is None:
            self.generator_feature_names = data.feature_names
            self.generator_feature_types = data.feature_types
        else:
            # Booster can't accept data with different feature names
            if self.feature_names != data.feature_names:
                dat_missing = set(self.generator_feature_names) - set(data.feature_names)
                my_missing = set(data.feature_names) - set(self.generator_feature_names)

                msg = 'feature_names mismatch: {0} {1}'

                if dat_missing:
                    msg += ('\nexpected ' + ', '.join(str(s) for s in dat_missing) +
                            ' in input data')

                if my_missing:
                    msg += ('\ntraining data did not have the following fields: ' +
                            ', '.join(str(s) for s in my_missing))

                raise ValueError(msg.format(self.generator_feature_names,
                                            data.feature_names))

    def _validate_discriminator_features(self, data):
        """
        Validate Booster and data's feature_names are identical.
        Set feature_names and feature_types from DMatrix
        """
        if self.discriminator_feature_names is None:
            self.discriminator_feature_names = data.feature_names
            self.discriminator_feature_types = data.feature_types
        else:
            # Booster can't accept data with different feature names
            if self.discriminator_feature_names != data.feature_names:
                dat_missing = set(self.discriminator_feature_names) - set(data.feature_names)
                my_missing = set(data.feature_names) - set(self.discriminator_feature_names)

                msg = 'feature_names mismatch: {0} {1}'

                if dat_missing:
                    msg += ('\nexpected ' + ', '.join(str(s) for s in dat_missing) +
                            ' in input data')

                if my_missing:
                    msg += ('\ntraining data did not have the following fields: ' +
                            ', '.join(str(s) for s in my_missing))

                raise ValueError(msg.format(self.discriminator_feature_names,
                                            data.feature_names))


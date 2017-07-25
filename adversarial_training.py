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
from xgboost.core import *
from xgboost.compat import (STRING_TYPES, PY3, DataFrame, py_str, PANDAS_INSTALLED, XGBStratifiedKFold,
                     SKLEARN_INSTALLED, XGBModelBase, XGBClassifierBase, XGBRegressorBase, XGBLabelEncoder)
from adversarial_booster import AdversarialBooster

def train(params, prior, prior_copy, distribution, posterior, num_boost_round=100, evals=(), obj=None, feval=None,
          maximize=False, early_stopping_rounds=None, evals_result=None,
          verbose_eval=True, xgb_model=None, callbacks=None, learning_rates=None):
    callbacks = [] if callbacks is None else callbacks

    # Most of legacy advanced options becomes callbacks
    if isinstance(verbose_eval, bool) and verbose_eval:
        callbacks.append(callback.print_evaluation())
    else:
        if isinstance(verbose_eval, int):
            callbacks.append(callback.print_evaluation(verbose_eval))

    if early_stopping_rounds is not None:
        callbacks.append(callback.early_stop(early_stopping_rounds,
                                             maximize=maximize,
                                             verbose=bool(verbose_eval)))
    if evals_result is not None:
        callbacks.append(callback.record_evaluation(evals_result))

    if learning_rates is not None:
        warnings.warn("learning_rates parameter is deprecated - use callback API instead",
                      DeprecationWarning)
        callbacks.append(callback.reset_learning_rate(learning_rates))

    return _train_internal(params, prior, prior_copy, distribution, posterior,
                           num_boost_round=num_boost_round,
                           evals=evals,
                           obj=obj, feval=feval,
                           xgb_model=xgb_model, callbacks=callbacks)

def _train_internal(params, prior, prior_copy, distribution, posterior,
                    num_boost_round=100, evals=(),
                    obj=None, feval=None,
                    xgb_model=None, callbacks=None):
    callbacks = [] if callbacks is None else callbacks
    evals = list(evals)
    if isinstance(params, dict) \
            and 'eval_metric' in params \
            and isinstance(params['eval_metric'], list):
        params = dict((k, v) for k, v in params.items())
        eval_metrics = params['eval_metric']
        params.pop("eval_metric", None)
        params = list(params.items())
        for eval_metric in eval_metrics:
            params += [('eval_metric', eval_metric)]

    bst = AdversarialBooster(params, [distribution] + [d[0] for d in evals])
    nboost = 0
    num_parallel_tree = 1

    if xgb_model is not None:
        if not isinstance(xgb_model, STRING_TYPES):
            xgb_model = xgb_model.save_raw()
        bst = GANBooster(params, [d[0] for d in evals], model_file=xgb_model)
        nboost = len(bst.get_dump())

    _params = dict(params) if isinstance(params, list) else params

    if 'num_parallel_tree' in _params:
        num_parallel_tree = _params['num_parallel_tree']
        nboost //= num_parallel_tree
    if 'num_class' in _params:
        nboost //= _params['num_class']

    # Distributed code: Load the checkpoint from rabit.
    version = bst.load_rabit_checkpoint()
    assert(rabit.get_world_size() != 1 or version == 0)
    rank = rabit.get_rank()
    start_iteration = int(version / 2)
    nboost += start_iteration

    callbacks_before_iter = [
        cb for cb in callbacks if cb.__dict__.get('before_iteration', False)]
    callbacks_after_iter = [
        cb for cb in callbacks if not cb.__dict__.get('before_iteration', False)]

    for i in range(start_iteration, num_boost_round):
        for cb in callbacks_before_iter:
            cb(CallbackEnv(model=bst,
                           cvfolds=None,
                           iteration=i,
                           begin_iteration=start_iteration,
                           end_iteration=num_boost_round,
                           rank=rank,
                           evaluation_result_list=None))
        if version % 2 == 0:
            bst.update(prior, prior_copy, distribution, posterior, i, obj)
            bst.save_rabit_checkpoint()
            version += 1

        assert(rabit.get_world_size() == 1 or version == rabit.version_number())

        nboost += 1
        evaluation_result_list = []
        if len(evals) != 0:
            bst_eval_set = bst.eval_set(evals, i, feval)
            if isinstance(bst_eval_set, STRING_TYPES):
                msg = bst_eval_set
            else:
                msg = bst_eval_set.decode()
            res = [x.split(':') for x in msg.split()]
            evaluation_result_list = [(k, float(v)) for k, v in res[1:]]
        try:
            for cb in callbacks_after_iter:
                cb(CallbackEnv(model=bst,
                               cvfolds=None,
                               iteration=i,
                               begin_iteration=start_iteration,
                               end_iteration=num_boost_round,
                               rank=rank,
                               evaluation_result_list=evaluation_result_list))
        except EarlyStopException:
            break
        bst.save_rabit_checkpoint()
        version += 1

    if bst.attr('best_score') is not None:
        bst.best_score = float(bst.attr('best_score'))
        bst.best_iteration = int(bst.attr('best_iteration'))
    else:
        bst.best_iteration = nboost - 1
    bst.best_ntree_limit = (bst.best_iteration + 1) * num_parallel_tree
    return bst
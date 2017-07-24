from xgboost.core import *
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
        for d in cache:
            if not isinstance(d, DMatrix):
                raise TypeError('invalid cache item: {}'.format(type(d).__name__))
            self._validate_features(d)

        dmats = c_array(ctypes.c_void_p, [d.handle for d in cache])
        self.discriminator_handle = ctypes.c_void_p()
        self.handle = ctypes.c_void_p()
        _check_call(_LIB.XGBoosterCreate(dmats, c_bst_ulong(len(cache)),
                                         ctypes.byref(self.handle_disc)))
        _check_call(_LIB.XGBoosterCreate(dmats, c_bst_ulong(len(cache)),
                                         ctypes.byref(self.handle)))
        self.set_param({'seed': 0})
        self.set_param(params or {})
        if model_file is not None:
            self.load_model(model_file)

    def _fobj_gen(self,pred,dtrain):
        g_fake = self._discriminate(pred)
        grad = 0.5 * (np.log(g_fake) - np.log(1 - g_fake))**2
        hess = (np.log(1-g_fake)-log(g_fake))/(g_fake*(1-g_fake))
        return grad,hess

    def _fobj_disc(self,pred,dtrain):
        d_real = self._discriminate(distribution)
        d_fake = self._discriminate(pred)
        grad = -(np.log(d_real) - np.log(d_fake))
        hess = 1/d_fake
        return grad,hess

    def _L2(self,pred,posterior):
        return np.sqrt(np.square(pred - posterior))

    def update(self, prior, distribution, posterior, iteration, fobj=None):
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
        pred = self._generate(prior)
        dpred = DMatrix(pred)
        grad, hess = self._fobj_disc(dpred, distribution)
        self.boost_discriminator(dtrain, grad, hess)

        grad, hess = self._fobj_gen(dpred,dtrain)
        self.boost_generator(dtrain, grad, hess)
        
        loss = self._L2(pred,posterior)
        print('Iteration {}: Loss - {}'.format(iteration,np.mean(np.abs(loss))))

    def boost_discriminator(self, distribution, grad, hess):
        if len(grad) != len(hess):
            raise ValueError('grad / hess length mismatch: {} / {}'.format(len(grad), len(hess)))
        if not isinstance(dtrain, DMatrix):
            raise TypeError('invalid training matrix: {}'.format(type(dtrain).__name__))
        self._validate_features(dtrain)

        _check_call(_LIB.XGBoosterBoostOneIter(self.discriminator_handle, dtrain.handle,
                                               c_array(ctypes.c_float, grad),
                                               c_array(ctypes.c_float, hess),
                                               c_bst_ulong(len(grad))))

    def boost_generator(self, dposterior, grad, hess):
        if len(grad) != len(hess):
            raise ValueError('grad / hess length mismatch: {} / {}'.format(len(grad), len(hess)))
        if not isinstance(dtrain, DMatrix):
            raise TypeError('invalid training matrix: {}'.format(type(dtrain).__name__))
        self._validate_features(dtrain)

        _check_call(_LIB.XGBoosterBoostOneIter(self.handle, dtrain.handle,
                                               c_array(ctypes.c_float, grad),
                                               c_array(ctypes.c_float, hess),
                                               c_bst_ulong(len(grad))))
    
    def _generate(self, data, output_margin=False, ntree_limit=0, pred_leaf=False,
                pred_contribs=False):
        option_mask = 0x00
        if output_margin:
            option_mask |= 0x01
        if pred_leaf:
            option_mask |= 0x02
        if pred_contribs:
            option_mask |= 0x04

        self._validate_features(data)

        length = c_bst_ulong()
        gen = ctypes.POINTER(ctypes.c_float)()
        _check_call(_LIB.XGBoosterPredict(self.handle, data.handle,
                                          ctypes.c_int(option_mask),
                                          ctypes.c_uint(ntree_limit),
                                          ctypes.byref(length),
                                          ctypes.byref(gen)))
        gen = ctypes2numpy(gen, length.value, np.float32)
        return gen

    def _discriminate(self, data, output_margin=False, ntree_limit=0, pred_leaf=False,
                pred_contribs=False):
        option_mask = 0x00
        if output_margin:
            option_mask |= 0x01
        if pred_leaf:
            option_mask |= 0x02
        if pred_contribs:
            option_mask |= 0x04

        #self._validate_features(data)

        length = c_bst_ulong()
        disc = ctypes.POINTER(ctypes.c_float)()
        _check_call(_LIB.XGBoosterPredict(self.handle_disc, data.handle,
                                          ctypes.c_int(option_mask),
                                          ctypes.c_uint(ntree_limit),
                                          ctypes.byref(length),
                                          ctypes.byref(disc)))
        disc = ctypes2numpy(disc, length.value, np.float32)
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

        self._validate_features(data)

        length = c_bst_ulong()
        preds = ctypes.POINTER(ctypes.c_float)()
        _check_call(_LIB.XGBoosterPredict(self.handle, data.handle,
                                          ctypes.c_int(option_mask),
                                          ctypes.c_uint(ntree_limit),
                                          ctypes.byref(length),
                                          ctypes.byref(preds)))
        preds = DMatrix(ctypes2numpy(preds, length.value, np.float32))
        pred_probas = ctypes.POINTER(ctypes.c_float)()
        _check_call(_LIB.XGBoosterPredict(self.handle_disc, preds.handle,
                                          ctypes.c_int(option_mask),
                                          ctypes.c_uint(ntree_limit),
                                          ctypes.byref(length),
                                          ctypes.byref(preds)))
        pred_probas = ctypes2numpy(pred_probas, length.value, np.float32)
        return preds

    def predict(self, data, output_margin=False, ntree_limit=0, pred_leaf=False,
                pred_contribs=False):
        option_mask = 0x00
        if output_margin:
            option_mask |= 0x01
        if pred_leaf:
            option_mask |= 0x02
        if pred_contribs:
            option_mask |= 0x04

        self._validate_features(data)

        length = c_bst_ulong()
        preds = ctypes.POINTER(ctypes.c_float)()
        _check_call(_LIB.XGBoosterPredict(self.handle, data.handle,
                                          ctypes.c_int(option_mask),
                                          ctypes.c_uint(ntree_limit),
                                          ctypes.byref(length),
                                          ctypes.byref(preds)))
        preds = ctypes2numpy(preds, length.value, np.float32)
        return preds

    def load_rabit_checkpoint(self):
        """Initialize the model by load from rabit checkpoint.
        Returns
        -------
        version: integer
            The version number of the model.
        """
        version_gen = ctypes.c_int()
        version_disc = ctypes.c_int()
        _check_call(_LIB.XGBoosterLoadRabitCheckpoint(
            self.handle, ctypes.byref(version_gen)))
        _check_call(_LIB.XGBoosterLoadRabitCheckpoint(
            self.discriminator_handle, ctypes.byref(version_disc)))
        if version_gen.value == version_disc.value:
            return version_gen.value
        else:
            raise AssertionError('Discriminator and Generator not of same version.')

    def save_rabit_checkpoint(self):
        """Save the current booster to rabit checkpoint."""
        _check_call(_LIB.XGBoosterSaveRabitCheckpoint(self.handle))
        _check_call(_LIB.XGBoosterSaveRabitCheckpoint(self.discriminator_handle))

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
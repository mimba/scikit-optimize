import numbers
import time
import warnings

import numpy as np
import scipy.sparse as sp
from sklearn.base import is_classifier, clone
from sklearn.exceptions import FitFailedWarning
from sklearn.externals.joblib import Parallel, delayed, logger
from sklearn.gaussian_process.kernels import Kernel as GPKernel
from sklearn.metrics.scorer import check_scoring
from sklearn.utils import indexable, safe_indexing
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _is_arraylike, _num_samples


def weighted_safe_split(estimator, X, y, sample_weight, indices, train_indices=None):
    """Create subset of dataset and properly handle kernels."""
    if hasattr(estimator, 'kernel') and callable(estimator.kernel) \
       and not isinstance(estimator.kernel, GPKernel):
        # cannot compute the kernel values with custom function
        raise ValueError("Cannot use a custom kernel function. "
                         "Precompute the kernel matrix instead.")

    if not hasattr(X, "shape"):
        if getattr(estimator, "_pairwise", False):
            raise ValueError("Precomputed kernels or affinity matrices have "
                             "to be passed as arrays or sparse matrices.")
        X_subset = [X[idx] for idx in indices]
    else:
        if getattr(estimator, "_pairwise", False):
            # X is a precomputed square kernel matrix
            if X.shape[0] != X.shape[1]:
                raise ValueError("X should be a square kernel matrix")
            if train_indices is None:
                X_subset = X[np.ix_(indices, indices)]
            else:
                X_subset = X[np.ix_(indices, train_indices)]
        else:
            X_subset = safe_indexing(X, indices)

    if y is not None:
        y_subset = safe_indexing(y, indices)
    else:
        y_subset = None

    if sample_weight is not None:
        sample_weight_subset = safe_indexing(sample_weight, indices)
    else:
        sample_weight_subset = None

    return X_subset, y_subset, sample_weight_subset


def _fit_and_score(estimator,
                   X, y,
                   sample_weight, sample_weight_steps,
                   scorer, train, test, verbose,
                   parameters, fit_params, return_train_score=False,
                   return_parameters=False, return_n_test_samples=False,
                   return_times=False, error_score='raise'):
    """Fit estimator and compute scores for a given dataset split.

        Parameters
        ----------
        estimator : estimator object implementing 'fit'
            The object to use to fit the data.

        X : array-like of shape at least 2D
            The data to fit.

        y : array-like, optional, default: None
            The target variable to try to predict in the case of
            supervised learning.

        sample_weight: array-like, optional, default:None
            Also known as sample_weight

        sample_weight_steps: array-like, optional, default:None
            Must be provided if sample_weight are given.
            Indicates all steps of the pipeline which apply the sample_weight.
            E.g: steps_l1_xgb = [
                ('nan', MissingNumbersTransformer(..)),
                ('lgbm', LGBMRegressor(..))
            ]
            Then one may provide the steps ['lgbm'] such that only the pipeline step named 'lgbm' becomes passed the sample_weight as fit_param

        scorer : callable
            A scorer callable object / function with signature
            ``scorer(estimator, X, y)``.

        train : array-like, shape (n_train_samples,)
            Indices of training samples.

        test : array-like, shape (n_test_samples,)
            Indices of test samples.

        verbose : integer
            The verbosity level.

        error_score : 'raise' (default) or numeric
            Value to assign to the score if an error occurs in estimator fitting.
            If set to 'raise', the error is raised. If a numeric value is given,
            FitFailedWarning is raised. This parameter does not affect the refit
            step, which will always raise the error.

        parameters : dict or None
            Parameters to be set on the estimator.

        fit_params : dict or None
            Parameters that will be passed to ``estimator.fit``.

        return_train_score : boolean, optional, default: False
            Compute and return score on training set.

        return_parameters : boolean, optional, default: False
            Return parameters that has been used for the estimator.

        Returns
        -------
        train_score : float, optional
            Score on training set, returned only if `return_train_score` is `True`.

        test_score : float
            Score on test set.

        n_test_samples : int
            Number of test samples.

        fit_time : float
            Time spent for fitting in seconds.

        score_time : float
            Time spent for scoring in seconds.

        parameters : dict or None, optional
            The parameters that have been evaluated.
        """

    if verbose > 1:
        if parameters is None:
            msg = ''
        else:
            msg = '%s' % (', '.join('%s=%s' % (k, v)
                                    for k, v in parameters.items()))
        print("[CV] %s %s" % (msg, (64 - len(msg)) * '.'))

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = dict([(k, _index_param_value(X, v, train))
                       for k, v in fit_params.items()])

    if parameters is not None:
        estimator.set_params(**parameters)

    start_time = time.time()

    X_train, y_train, sample_weight_train = weighted_safe_split(estimator, X, y, sample_weight, train)
    X_test, y_test, sample_weight_test = weighted_safe_split(estimator, X, y, sample_weight, test, train)

    try:
        if y_train is None and sample_weight_train is None:
            estimator.fit(X_train, **fit_params)
        elif sample_weight_train is None:
            estimator.fit(X_train, y_train, **fit_params)
        else:
            for step in sample_weight_steps:
                fit_params[step + '__sample_weight'] = sample_weight_train

            estimator.fit(X_train, y_train, **fit_params)

    except Exception as e:
        # Note fit time as time until error
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == 'raise':
            raise
        elif isinstance(error_score, numbers.Number):
            test_score = error_score
            if return_train_score:
                train_score = error_score
            warnings.warn("Classifier fit failed. The score on this train-test"
                          " partition for these parameters will be set to %f. "
                          "Details: \n%r" % (error_score, e), FitFailedWarning)
        else:
            raise ValueError("error_score must be the string 'raise' or a"
                             " numeric value. (Hint: if using 'raise', please"
                             " make sure that it has been spelled correctly.)")

    else:
        fit_time = time.time() - start_time
        test_score = _score(estimator, X_test, y_test, sample_weight_test, scorer)
        score_time = time.time() - start_time - fit_time
        if return_train_score:
            train_score = _score(estimator, X_train, y_train, sample_weight_train, scorer)

    if verbose > 2:
        msg += ", score=%f" % test_score
    if verbose > 1:
        total_time = score_time + fit_time
        end_msg = "%s, total=%s" % (msg, logger.short_format_time(total_time))
        print("[CV] %s %s" % ((64 - len(end_msg)) * '.', end_msg))

    ret = [train_score, test_score] if return_train_score else [test_score]

    if return_n_test_samples:
        ret.append(_num_samples(X_test))
    if return_times:
        ret.extend([fit_time, score_time])
    if return_parameters:
        ret.append(parameters)
    return ret


def _index_param_value(X, v, indices):
    """Private helper function for parameter value indexing."""
    if not _is_arraylike(v) or _num_samples(v) != _num_samples(X):
        # pass through: skip indexing
        return v
    if sp.issparse(v):
        v = v.tocsr()
    return safe_indexing(v, indices)


def _score(estimator, X_test, y_test, sample_weight_test, scorer):
    """Compute the score of an estimator on a given test set."""
    if y_test is not None and sample_weight_test is not None:
        score = scorer(estimator, X_test, y_test, sample_weight_test)
    elif y_test is None:
        score = scorer(estimator, X_test)
    else:
        score = scorer(estimator, X_test, y_test)
    if hasattr(score, 'item'):
        try:
            # e.g. unwrap memmapped scalars
            score = score.item()
        except ValueError:
            # non-scalar?
            pass
    if not isinstance(score, numbers.Number):
        raise ValueError("scoring must return a number, got %s (%s) instead."
                         % (str(score), type(score)))
    return score

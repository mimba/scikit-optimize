"""Test scikit-optimize based implementation of hyperparameter
search with interface similar to those of GridSearchCV
"""

import numpy as np
import pytest
import time

from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.tree import DecisionTreeClassifier

from skopt import WeightedBayesSearchCV

from sklearn.base import clone
from sklearn.base import BaseEstimator
from joblib import cpu_count
from scipy.stats import rankdata
import numpy as np
from numpy.testing import assert_array_equal
from skopt.space import Real, Categorical, Integer
from skopt import WeightedBayesSearchCV
from skopt.tests.utils import assert_less_list

def _fit_svr(n_jobs=1, n_points=1, cv=None):
    """Utility function to fit a larger regression task with SVR
    """

    X, y = make_regression(n_samples=1000, n_features=20,
                           n_informative=18, random_state=1)
    opt = WeightedBayesSearchCV(
        SVR(),
        {
            'C': Real(1e-3, 1e+3, prior='log-uniform'),
            'gamma': Real(1e-3, 1e+1, prior='log-uniform'),
            'degree': Integer(1, 3),
        },
        scoring=make_scorer(mean_squared_error, greater_is_better=False),
        n_jobs=n_jobs, n_iter=11, n_points=n_points, cv=cv, random_state=42
    )
    opt.fit(X, y)
    assert opt.score(X,y) > -70000

def _fit_svc(n_jobs=1, n_points=1, cv=None):
    """
    Utility function to fit a larger classification task with SVC
    """

    X, y = make_classification(n_samples=1000, n_features=20, n_redundant=0,
                               n_informative=18, random_state=1,
                               n_clusters_per_class=1)

    opt = WeightedBayesSearchCV(
        SVC(),
        {
            'C': Real(1e-3, 1e+3, prior='log-uniform'),
            'gamma': Real(1e-3, 1e+1, prior='log-uniform'),
            'degree': Integer(1, 3),
        },
        n_jobs=n_jobs, n_iter=11, n_points=n_points, cv=cv, random_state=42
    )

    opt.fit(X, y)
    assert opt.score(X, y) > 0.9

    opt2 = WeightedBayesSearchCV(
        SVC(),
        {
            'C': Real(1e-3, 1e+3, prior='log-uniform'),
            'gamma': Real(1e-3, 1e+1, prior='log-uniform'),
            'degree': Integer(1, 3),
        },
        n_jobs=n_jobs, n_iter=11, n_points=n_points, cv=cv,
        random_state=42,
    )
    opt2.fit(X, y)

    assert opt.score(X, y) == opt2.score(X, y)


def _fit_class_weighted_cv(n_jobs=1, n_points=1, cv=None, random_state=13):
    """
    Utility function to fit a larger classification task with SVC and randomly filled sample weight
    :return score accuracy
    """
    score = _fit_class_weighted_cv_e(
        random_state=random_state,
        estimator=SVC(),
        search_spaces={
            'estimator__C': Real(1e-3, 1e+3, prior='log-uniform'),
            'estimator__gamma': Real(1e-3, 1e+1, prior='log-uniform'),
            'estimator__degree': Integer(1, 3),
        },
        n_jobs=n_jobs,
        n_points=n_points,
        cv=cv)
    assert score > 0.8  # we are more tolerant in weighted case as weighting may be misleading
    return score


def _fit_class_weighted_cv_e(estimator, search_spaces, n_jobs=1, n_points=1, cv=None, random_state=13):
    """
    Utility function to fit a larger classification task with the provided estimator, search space and randomly filled sample weight
    :return score accuracy
    """
    X, y = make_classification(n_samples=1000, n_features=20, n_redundant=0,
                               n_informative=18, random_state=1,
                               n_clusters_per_class=1)

    sample_weight = np.random.rand(len(y))
    pipeline = Pipeline([('estimator', estimator)])
    opt = WeightedBayesSearchCV(
        pipeline,
        random_state=random_state,
        search_spaces=search_spaces,
        n_jobs=n_jobs, n_iter=11, n_points=n_points, cv=cv
    )
    opt.fit(X, y, sample_weight=sample_weight, sample_weight_steps=['estimator'])
    score = opt.score(X, y)
    return score


def _fit_reg_weighted_cv(n_jobs=1, n_points=1, cv=None, random_state=13):
    """
    Utility function to fit a larger regression task with SVR and randomly filled sample weight
    :return score
    """
    score = _fit_reg_weighted_cv_e(
        random_state=random_state,
        estimator=SVR(),
        search_spaces={
            'estimator__C': Real(1e-3, 1e+3, prior='log-uniform'),
            'estimator__gamma': Real(1e-2, 1e+2, prior='log-uniform'),
            'estimator__degree': Integer(1, 3)
        },
        n_jobs=n_jobs,
        n_points=n_points,
        cv=cv)
    assert score > -70000  # we are more tolerant in weighted case as weighting may be misleading
    return score


def _fit_reg_weighted_cv_e(estimator, search_spaces, n_jobs=1, n_points=1, cv=None, random_state=13):
    """
    Utility function to fit a larger regression task with the provided estimator, search space and randomly filled sample weight
    :return score
    """
    X, y = make_regression(n_samples=1000, n_features=20,
                               n_informative=18, random_state=1)

    sample_weight = np.random.rand(len(y))
    pipeline = Pipeline([('estimator', estimator)])
    opt = WeightedBayesSearchCV(
        pipeline,
        scoring=make_scorer(mean_squared_error, greater_is_better=False),
        random_state=random_state,
        search_spaces=search_spaces,
        n_jobs=n_jobs, n_iter=11, n_points=n_points, cv=cv
    )
    opt.fit(X, y, sample_weight=sample_weight, sample_weight_steps=['estimator'])
    score = opt.score(X, y)
    return score


def test_raise_errors():
    # check if empty search space is raising errors
    with pytest.raises(ValueError):
        WeightedBayesSearchCV(SVC(), {})

    # check if invalid dimensions are raising errors
    with pytest.raises(ValueError):
        WeightedBayesSearchCV(SVC(), {'C': '1 ... 100.0'})

    with pytest.raises(TypeError):
        WeightedBayesSearchCV(SVC(), ['C', (1.0, 1)])


@pytest.mark.parametrize("surrogate", ['gp', None])
@pytest.mark.parametrize("n_jobs", [1, -1])  # test sequential and parallel
@pytest.mark.parametrize("n_points", [1, 3])  # test query of multiple points
def test_searchcv_runs(surrogate, n_jobs, n_points, cv=None):
    """
    Test whether the cross validation search wrapper around sklearn
    models runs properly with available surrogates and with single
    or multiple workers and different number of parameter settings
    to ask from the optimizer in parallel.

    Parameters
    ----------

    * `surrogate` [str or None]:
        A class of the scikit-optimize surrogate used. None means
        to use default surrogate.

    * `n_jobs` [int]:
        Number of parallel processes to use for computations.

    """

    X, y = load_iris(True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.75, random_state=0
    )

    # create an instance of a surrogate if it is not a string
    if surrogate is not None:
        optimizer_kwargs = {'base_estimator': surrogate}
    else:
        optimizer_kwargs = None

    opt = WeightedBayesSearchCV(
        SVC(),
        {
            'C': Real(1e-6, 1e+6, prior='log-uniform'),
            'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
            'degree': Integer(1, 8),
            'kernel': Categorical(['linear', 'poly', 'rbf']),
        },
        n_jobs=n_jobs, n_iter=11, n_points=n_points, cv=cv,
        optimizer_kwargs=optimizer_kwargs
    )

    opt.fit(X_train, y_train)

    # this normally does not hold only if something is wrong
    # with the optimizaiton procedure as such
    assert opt.score(X_test, y_test) > 0.9


@pytest.mark.slow_test
def test_parallel_class_cv():
    """
    Test whether parallel classification jobs work
    """
    _fit_svc(n_jobs=1, cv=5)
    _fit_svc(n_jobs=2, cv=5)


@pytest.mark.slow_test
def test_parallel_reg_cv():
    """
    Test whether parallel regression jobs work
    """
    _fit_svr(n_jobs=1, cv=5)
    _fit_svr(n_jobs=2, cv=5)


@pytest.mark.slow_test
def test_parallel_class_weighted_cv():
    """
    Test whether parallel classification jobs work with sample weights. For each cv run it is expected to produce
    different accuracy scores because of different sample_weight.
    Attention: This test may fail under extreme random sampling conditions
    """
    score_1 = _fit_class_weighted_cv(n_jobs=1, cv=5, random_state=13)
    score_2 = _fit_class_weighted_cv(n_jobs=1, cv=5, random_state=15)
    score_3 = _fit_class_weighted_cv(n_jobs=1, cv=5, random_state=17)
    assert score_1 == pytest.approx(score_2, 0.1)
    assert score_1 == pytest.approx(score_3, 0.1)

    dummy_estimator = DummyClassifier(strategy="stratified")
    score_dummy_1 = _fit_class_weighted_cv_e(estimator=dummy_estimator, search_spaces={"estimator__random_state": Integer(1, 1000)}, n_jobs=1, cv=5)
    assert_less_list(score_dummy_1, [score_1, score_2, score_3])
    dummy_estimator_2 = DummyClassifier(strategy="most_frequent")
    score_dummy_2 = _fit_class_weighted_cv_e(estimator=dummy_estimator_2, search_spaces={"estimator__random_state": Integer(1, 1000)}, n_jobs=1, cv=5)
    assert_less_list(score_dummy_2, [score_1, score_2, score_3])
    dummy_estimator_3 = DummyClassifier(strategy="uniform")
    score_dummy_3 = _fit_class_weighted_cv_e(estimator=dummy_estimator_3, search_spaces={"estimator__random_state": Integer(1, 1000)}, n_jobs=1, cv=5)
    assert_less_list(score_dummy_3, [score_1, score_2, score_3])


def test_parallel_reg_weighted_cv():
    """
    Test whether parallel regression jobs work with sample weights. For each cv run it is expected to produce
    different scores because of different sample_weight.
    Attention: This test may fail under extreme random sampling conditions
    """
    score_1 = _fit_reg_weighted_cv(n_jobs=1, cv=5, random_state=13)
    score_2 = _fit_reg_weighted_cv(n_jobs=1, cv=5, random_state=15)
    score_3 = _fit_reg_weighted_cv(n_jobs=1, cv=5, random_state=17)
    assert score_1 != score_2 or score_1 != score_3

    dummy_estimator = DummyRegressor(strategy="mean")
    score_dummy_1 = _fit_reg_weighted_cv_e(estimator=dummy_estimator, search_spaces={"estimator__constant": Integer(1, 1000)}, random_state=13)
    assert_less_list(score_dummy_1, [score_1, score_2, score_3])
    dummy_estimator_2 = DummyRegressor(strategy="median")
    score_dummy_2 = _fit_reg_weighted_cv_e(estimator=dummy_estimator_2, search_spaces={"estimator__constant": Integer(1, 1000)}, random_state=15)
    assert_less_list(score_dummy_2, [score_1, score_2, score_3])


def test_searchcv_runs_multiple_subspaces():
    """
    Test whether the WeightedBayesSearchCV runs without exceptions when
    multiple subspaces are given.
    """

    X, y = load_iris(True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.75, random_state=0
    )

    # used to try different model classes
    pipe = Pipeline([
        ('model', SVC())
    ])

    # single categorical value of 'model' parameter sets the model class
    lin_search = {
        'model': Categorical([LinearSVC()]),
        'model__C': Real(1e-6, 1e+6, prior='log-uniform'),
    }

    dtc_search = {
        'model': Categorical([DecisionTreeClassifier()]),
        'model__max_depth': Integer(1, 32),
        'model__min_samples_split': Real(1e-3, 1.0, prior='log-uniform'),
    }

    svc_search = {
        'model': Categorical([SVC()]),
        'model__C': Real(1e-6, 1e+6, prior='log-uniform'),
        'model__gamma': Real(1e-6, 1e+1, prior='log-uniform'),
        'model__degree': Integer(1, 8),
        'model__kernel': Categorical(['linear', 'poly', 'rbf']),
    }

    opt = WeightedBayesSearchCV(
        pipe,
        [(lin_search, 1), (dtc_search, 1), svc_search],
        n_iter=2
    )

    opt.fit(X_train, y_train)

    # test if all subspaces are explored
    total_evaluations = len(opt.cv_results_['mean_test_score'])
    assert total_evaluations == 1 + 1 + 2, "Not all spaces were explored!"
    assert len(opt.optimizer_results_) == 3
    assert isinstance(opt.optimizer_results_[0].x[0], LinearSVC)
    assert isinstance(opt.optimizer_results_[1].x[0], DecisionTreeClassifier)
    assert isinstance(opt.optimizer_results_[2].x[0], SVC)


def test_searchcv_sklearn_sample_weight_compatibility():
    """
    Test whether the WeightedBayesSearchCV is compatible with sklearn methods having sample weights integrated.
    """
    X, y = load_iris(True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.75, random_state=0
    )

    # used to try different model classes
    pipe = Pipeline([
        ('model', SVC())
    ])

    # single categorical value of 'model' parameter sets the model class
    lin_search = {
        'model': Categorical([LinearSVC()]),
        'model__C': Real(1e-6, 1e+6, prior='log-uniform'),
    }

    dtc_search = {
        'model': Categorical([DecisionTreeClassifier()]),
        'model__max_depth': Integer(1, 32),
        'model__min_samples_split': Real(1e-3, 1.0, prior='log-uniform'),
    }

    svc_search = {
        'model': Categorical([SVC()]),
        'model__C': Real(1e-6, 1e+6, prior='log-uniform'),
        'model__gamma': Real(1e-6, 1e+1, prior='log-uniform'),
        'model__degree': Integer(1, 8),
        'model__kernel': Categorical(['linear', 'poly', 'rbf']),
    }

    opt = WeightedBayesSearchCV(
        pipe,
        [(lin_search, 1), svc_search],
        n_iter=2
    )

    opt_clone = clone(opt)

    params, params_clone = opt.get_params(), opt_clone.get_params()
    assert params.keys() == params_clone.keys()

    for param, param_clone in zip(params.items(), params_clone.items()):
        assert param[0] == param_clone[0]
        assert isinstance(param[1], type(param_clone[1]))

    opt.set_params(search_spaces=[(dtc_search, 1)])

    sample_weight = np.random.rand(len(y_train))

    opt.fit(X_train, y_train, sample_weight=sample_weight, sample_weight_steps=['model'])
    opt_clone.fit(X_train, y_train, sample_weight=sample_weight, sample_weight_steps=['model'])

    total_evaluations = len(opt.cv_results_['mean_test_score'])
    total_evaluations_clone = len(opt_clone.cv_results_['mean_test_score'])

    # test if expected number of subspaces is explored
    assert total_evaluations == 1
    assert total_evaluations_clone == 1 + 2


def test_searchcv_sklearn_compatibility():
    """
    Test whether the WeightedBayesSearchCV is compatible with base sklearn methods
    such as clone, set_params, get_params.
    """

    X, y = load_iris(True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.75, random_state=0
    )

    # used to try different model classes
    pipe = Pipeline([
        ('model', SVC())
    ])

    # single categorical value of 'model' parameter sets the model class
    lin_search = {
        'model': Categorical([LinearSVC()]),
        'model__C': Real(1e-6, 1e+6, prior='log-uniform'),
    }

    dtc_search = {
        'model': Categorical([DecisionTreeClassifier()]),
        'model__max_depth': Integer(1, 32),
        'model__min_samples_split': Real(1e-3, 1.0, prior='log-uniform'),
    }

    svc_search = {
        'model': Categorical([SVC()]),
        'model__C': Real(1e-6, 1e+6, prior='log-uniform'),
        'model__gamma': Real(1e-6, 1e+1, prior='log-uniform'),
        'model__degree': Integer(1, 8),
        'model__kernel': Categorical(['linear', 'poly', 'rbf']),
    }

    opt = WeightedBayesSearchCV(
        pipe,
        [(lin_search, 1), svc_search],
        n_iter=2
    )

    opt_clone = clone(opt)

    params, params_clone = opt.get_params(), opt_clone.get_params()
    assert params.keys() == params_clone.keys()

    for param, param_clone in zip(params.items(), params_clone.items()):
        assert param[0] == param_clone[0]
        assert isinstance(param[1], type(param_clone[1]))

    opt.set_params(search_spaces=[(dtc_search, 1)])

    opt.fit(X_train, y_train)
    opt_clone.fit(X_train, y_train)

    total_evaluations = len(opt.cv_results_['mean_test_score'])
    total_evaluations_clone = len(opt_clone.cv_results_['mean_test_score'])

    # test if expected number of subspaces is explored
    assert total_evaluations == 1
    assert total_evaluations_clone == 1 + 2


def test_searchcv_reproducibility():
    """
    Test whether results of WeightedBayesSearchCV can be reproduced with a fixed
    random state.
    """

    X, y = load_iris(True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.75, random_state=0
    )

    random_state = 42

    opt = WeightedBayesSearchCV(
        SVC(random_state=random_state),
        {
            'C': Real(1e-6, 1e+6, prior='log-uniform'),
            'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
            'degree': Integer(1, 8),
            'kernel': Categorical(['linear', 'poly', 'rbf']),
        },
        n_iter=11, random_state=random_state
    )

    opt.fit(X_train, y_train)
    best_est = opt.best_estimator_
    optim_res = opt.optimizer_results_[0].x

    opt2 = clone(opt).fit(X_train, y_train)
    best_est2 = opt2.best_estimator_
    optim_res2 = opt2.optimizer_results_[0].x

    assert getattr(best_est, 'C') == getattr(best_est2, 'C')
    assert getattr(best_est, 'gamma') == getattr(best_est2, 'gamma')
    assert getattr(best_est, 'degree') == getattr(best_est2, 'degree')
    assert getattr(best_est, 'kernel') == getattr(best_est2, 'kernel')
    # dict is sorted by alphabet
    assert optim_res[0] == getattr(best_est, 'C')
    assert optim_res[2] == getattr(best_est, 'gamma')
    assert optim_res[1] == getattr(best_est, 'degree')
    assert optim_res[3] == getattr(best_est, 'kernel')
    assert optim_res2[0] == getattr(best_est, 'C')
    assert optim_res2[2] == getattr(best_est, 'gamma')
    assert optim_res2[1] == getattr(best_est, 'degree')
    assert optim_res2[3] == getattr(best_est, 'kernel')


@pytest.mark.fast_test
def test_searchcv_rank():
    """
    Test whether results of WeightedBayesSearchCV can be reproduced with a fixed
    random state.
    """

    X, y = load_iris(True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.75, random_state=0
    )

    random_state = 42

    opt = WeightedBayesSearchCV(
        SVC(random_state=random_state),
        {
            'C': Real(1e-6, 1e+6, prior='log-uniform'),
            'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
            'degree': Integer(1, 8),
            'kernel': Categorical(['linear', 'poly', 'rbf']),
        },
        n_iter=11, random_state=random_state, return_train_score=True
    )

    opt.fit(X_train, y_train)
    results = opt.cv_results_

    test_rank = np.asarray(rankdata(-np.array(results["mean_test_score"]),
                                    method='min'), dtype=np.int32)
    train_rank = np.asarray(rankdata(-np.array(results["mean_train_score"]),
                                     method='min'), dtype=np.int32)

    assert_array_equal(np.array(results['rank_test_score']), test_rank)
    assert_array_equal(np.array(results['rank_train_score']), train_rank)


def test_searchcv_refit():
    """
    Test whether results of WeightedBayesSearchCV can be reproduced with a fixed
    random state.
    """

    X, y = load_iris(True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.75, random_state=0
    )

    random_state = 42

    opt = WeightedBayesSearchCV(
        SVC(random_state=random_state),
        {
            'C': Real(1e-6, 1e+6, prior='log-uniform'),
            'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
            'degree': Integer(1, 8),
            'kernel': Categorical(['linear', 'poly', 'rbf']),
        },
        n_iter=11, random_state=random_state
    )

    opt2 = WeightedBayesSearchCV(
        SVC(random_state=random_state),
        {
            'C': Real(1e-6, 1e+6, prior='log-uniform'),
            'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
            'degree': Integer(1, 8),
            'kernel': Categorical(['linear', 'poly', 'rbf']),
        },
        n_iter=11, random_state=random_state, refit=True
    )

    opt.fit(X_train, y_train)
    opt2.best_estimator_ = opt.best_estimator_

    opt2.fit(X_train, y_train)
    # this normally does not hold only if something is wrong
    # with the optimizaiton procedure as such
    assert opt2.score(X_test, y_test) > 0.9


def test_searchcv_callback():
    # Test whether callback is used in WeightedBayesSearchCV and
    # whether is can be used to interrupt the search loop

    X, y = load_iris(True)
    opt = WeightedBayesSearchCV(
        DecisionTreeClassifier(),
        {
            'max_depth': [3],  # additional test for single dimension
            'min_samples_split': Real(0.1, 0.9),
        },
        n_iter=5
    )
    total_iterations = [0]

    def callback(opt_result):
        # this simply counts iterations
        total_iterations[0] += 1

        # break the optimization loop at some point
        if total_iterations[0] > 2:
            return True  # True == stop optimization

        return False

    opt.fit(X, y, callback=callback)

    assert total_iterations[0] == 3

    # test whether final model was fit
    opt.score(X, y)


def test_searchcv_total_iterations():
    # Test the total iterations counting property of WeightedBayesSearchCV

    opt = WeightedBayesSearchCV(
        DecisionTreeClassifier(),
        [
            ({'max_depth': (1, 32)}, 10),  # 10 iterations here
            {'min_samples_split': Real(0.1, 0.9)}  # 5 (default) iters here
        ],
        n_iter=5
    )

    assert opt.total_iterations == 10 + 5


def test_search_cv_internal_parameter_types():
    # Test whether the parameters passed to the
    # estimator of the WeightedBayesSearchCV are of standard python
    # types - float, int, str

    # This is estimator is used to check whether the types provided
    # are native python types.
    class TypeCheckEstimator(BaseEstimator):
        def __init__(self, float_param=0.0, int_param=0, str_param=""):
            self.float_param = float_param
            self.int_param = int_param
            self.str_param = str_param

        def fit(self, X, y):
            assert isinstance(self.float_param, float)
            assert isinstance(self.int_param, int)
            assert isinstance(self.str_param, str)
            return self

        def score(self, X, y):
            return 0.0

    # Below is example code that used to not work.
    X, y = make_classification(10, 4)

    model = WeightedBayesSearchCV(
        estimator=TypeCheckEstimator(),
        search_spaces={
            'float_param': [0.0, 1.0],
            'int_param': [0, 10],
            'str_param': ["one", "two", "three"],
        },
        n_iter=11
    )

    model.fit(X, y)

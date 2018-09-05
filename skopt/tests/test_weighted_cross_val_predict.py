import pytest
from sklearn.datasets import make_regression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from skopt.space import Real, Integer

from skopt import WeightedBayesSearchCV

from skopt.ext import weighted_validation

import numpy as np


@pytest.mark.slow_test
def test_cross_val_predict():
    """
    Test whether cross_val_predict works as intended incl sample weight
    :return:
    """
    # test svr against dummy
    dummy_estimator = DummyRegressor(strategy="mean")
    X, y = make_regression(n_samples=1000, n_features=20,
                           n_informative=18, random_state=1,
                           bias=10)
    dummy_y_pred = weighted_validation.cross_val_predict(estimator=dummy_estimator, X=X, y=y)
    dummy_mse = mean_squared_error(y_true=y, y_pred=dummy_y_pred)
    svr_pipeline = WeightedBayesSearchCV(
        random_state=13,
        n_iter=11,
        cv=3,
        estimator=Pipeline([('estimator', SVR())]),
        search_spaces={
            'estimator__C': Real(1e-3, 1e+3, prior='log-uniform'),
            'estimator__gamma': Real(1e-3, 1e+1, prior='log-uniform'),
            'estimator__degree': Integer(1, 3)
        })
    svr_y_pred = weighted_validation.cross_val_predict(estimator=svr_pipeline, X=X, y=y)
    svr_mse = mean_squared_error(y_true=y, y_pred=svr_y_pred)

    assert svr_mse < dummy_mse

    # test sample weight aware mse of svr pipeline trained with sample weight against svr pipeline trained without
    # sample weight
    min = np.min(y)
    max = np.max(y)

    # numpy average implementation and thus also sklearn requires weights to be positive
    # introduce sw_big: sample weight that weight big y higher than small y
    sw_big = (y - min) / (max - min)
    svr_pipeline_sw = WeightedBayesSearchCV(
        random_state=13,
        n_iter=12,
        cv=3,
        estimator=Pipeline([('estimator', SVR())]),
        search_spaces={
            'estimator__C': Real(1e-3, 1e+3, prior='log-uniform'),
            'estimator__gamma': Real(1e-3, 1e+1, prior='log-uniform'),
            'estimator__degree': Integer(1, 3)
        })
    svr_pipeline_sw_y_pred = weighted_validation.cross_val_predict(estimator=svr_pipeline_sw, X=X, y=y,
                                                                   sample_weight=sw_big,
                                                                   sample_weight_steps=['estimator'])

    # introduce sw_small: sample weight that weight small y higher than big y
    sw_small = 1 - sw_big
    # evaluate model optimized for big y to small weights
    svr_pipeline_sw_mse_high = mean_squared_error(y_true=y, y_pred=svr_pipeline_sw_y_pred, sample_weight=sw_small)
    # build model optimized for small y
    svr_pipeline_sw = WeightedBayesSearchCV(
        random_state=13,
        n_iter=12,
        cv=3,
        estimator=Pipeline([('estimator', SVR())]),
        search_spaces={
            'estimator__C': Real(1e-3, 1e+3, prior='log-uniform'),
            'estimator__gamma': Real(1e-3, 1e+1, prior='log-uniform'),
            'estimator__degree': Integer(1, 3)
        })
    svr_pipeline_sw_y_pred = weighted_validation.cross_val_predict(estimator=svr_pipeline_sw, X=X, y=y, sample_weight=sw_small, sample_weight_steps=['estimator'])
    svr_pipeline_sw_mse = mean_squared_error(y_true=y, y_pred=svr_pipeline_sw_y_pred, sample_weight=sw_small)
    # expect that model optimized to fit small y evaluates better than model optimized to fit big y.
    assert svr_pipeline_sw_mse < svr_pipeline_sw_mse_high

    # test sample weight aware mse of svr estimator trained with sample weight against svr
    svr_sw = WeightedBayesSearchCV(
        random_state=13,
        n_iter=12,
        cv=3,
        estimator=SVR(),
        search_spaces={
            'C': Real(1e-3, 1e+3, prior='log-uniform'),
            'gamma': Real(1e-3, 1e+1, prior='log-uniform'),
            'degree': Integer(1, 3)
        })
    svr_sw_y_pred = weighted_validation.cross_val_predict(estimator=svr_sw, X=X, y=y, sample_weight=sw_small)
    svr_sw_mse = mean_squared_error(y_true=y, y_pred=svr_sw_y_pred, sample_weight=sw_small)

    assert svr_sw_mse < svr_pipeline_sw_mse_high



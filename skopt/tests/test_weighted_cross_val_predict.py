import numpy as np
import pytest
from lightgbm import LGBMRegressor
from sklearn.datasets import make_regression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error

from skopt.ext import weighted_validation


def test_cross_val_predict():
    """
    Test whether cross_val_predict works as intended incl sample weight.
    It uses a a lightgm regressor because sample weights usually highly influence the performance of a lightgbm model.
    :return:
    """
    # test svr against dummy
    dummy_estimator = DummyRegressor(strategy="mean")
    X, y = make_regression(n_samples=1000, n_features=20,
                           n_informative=18, random_state=1,
                           bias=10)
    dummy_y_pred = weighted_validation.cross_val_predict(estimator=dummy_estimator, X=X, y=y)
    dummy_mse = mean_squared_error(y_true=y, y_pred=dummy_y_pred)

    estimator = LGBMRegressor()
    y_pred = weighted_validation.cross_val_predict(estimator=estimator, X=X, y=y)
    mse = mean_squared_error(y_true=y, y_pred=y_pred)

    assert mse < dummy_mse

    # test sample weight aware mse of svr pipeline trained with sample weight against svr pipeline trained without
    # sample weight
    sw_big = np.where(y > 0, 1, 0)
    sw_small = 1 - sw_big

    estimator_sw = LGBMRegressor()
    y_pred_sw = weighted_validation.cross_val_predict(estimator=estimator_sw, X=X, y=y,
                                                      sample_weight=sw_big)

    # evaluate model optimized for big y to small weights
    mse_big = mean_squared_error(y_true=y, y_pred=y_pred_sw, sample_weight=sw_small)
    # build model optimized for small y
    estimator_sw_small = LGBMRegressor()
    y_pred_sw_small = weighted_validation.cross_val_predict(
        estimator=estimator_sw_small, X=X, y=y, sample_weight=sw_small)
    mse_small = mean_squared_error(y_true=y, y_pred=y_pred_sw_small, sample_weight=sw_small)
    # expect that model optimized to fit small y evaluates better than model optimized to fit big y.
    assert mse_small < mse_big


# @pytest.mark.slow_test
# def test_nested_cv():
#     X, y = make_regression(n_samples=1000, n_features=20,
#                            n_informative=18, random_state=1,
#                            bias=10)
#
#     np.random.seed(13)
#
#     # test sample weight aware mse of svr pipeline trained with sample weight against svr pipeline trained without
#     # sample weight
#     # numpy average implementation and thus also sklearn requires weights to be positive
#     # introduce sw_big: sample weight that weight big y higher than small y
#     sw_big = np.where(y > 0, 1, 0)
#     sw_small = 1 - sw_big
#
#     # test nested cv
#     y_pred_nested_small = weighted_validation.cross_val_predict(
#         estimator=WeightedBayesSearchCV(
#             refit=True,
#             random_state=13,
#             n_iter=12,
#             cv=3,
#             estimator=Pipeline([('estimator', LGBMRegressor())]),
#             search_spaces={
#                 'estimator__learning_rate': Real(0.05, 0.2, prior='uniform'),
#                 'estimator__n_estimators': Integer(70, 130)
#             }),
#         X=X, y=y, sample_weight=sw_small, sample_weight_steps=['estimator'])
#     mse_nested_small = mean_squared_error(y_true=y, y_pred=y_pred_nested_small, sample_weight=sw_small)
#
#     y_pred_nested_big = weighted_validation.cross_val_predict(
#         estimator=WeightedBayesSearchCV(
#             refit=True,
#             random_state=13,
#             n_iter=12,
#             cv=3,
#             estimator=Pipeline([('estimator', LGBMRegressor())]),
#             search_spaces={
#                 'estimator__learning_rate': Real(0.05, 0.2, prior='uniform'),
#                 'estimator__n_estimators': Integer(70, 130)
#             }),
#         X=X, y=y, sample_weight=sw_big, sample_weight_steps=['estimator'])
#     mse_nested_big = mean_squared_error(y_true=y, y_pred=y_pred_nested_big, sample_weight=sw_small)
#
#     assert mse_nested_small < mse_nested_big

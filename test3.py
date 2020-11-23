import os
import sys

import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

import statsmodels.api as sm
import xgboost as xgb
import shap

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["figure.figsize"] = (12, 5)


def load_data():
    """
    read housing.data
    1. CRIM per capita crime rate by town ; 범죄율(%), float, continuous
    2. ZN proportion of residential land zoned for lots over25,000 sq.ft. ; 주거용지비율(%), float, continuous
    3. INDUS proportion of non-retail business acres per town ; 비소매업비율(%), float, continuous
    4. CHAS Charles River dummy variable (= 1 if tract bounds river; 0 otherwise) ; 강인접여부, category, binary
    5. NOX nitric oxides concentration (parts per 10 million) ; 산화질소농도, float, continuous
    6. RM average number of rooms per dwelling ; 평균방수, float, continuous
    7. AGE proportion of owner-occupied units built prior to 1940 ; 노후주택비율(%), float, continuous
    8. DIS weighted distances to five Boston employment centres ; 일자리거리, float, continuous
    9. RAD index of accessibility to radial highways ; 도로접근지수, int, continuous
    10. TAX full-value property-tax rate per \$10,000 ; 재산세, int, continuous
    11. PTRATIO pupil-teacher ratio by town ; 학생교사비율, float, continuous
    12. B 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town ; 흑인비율지수, float, continuous
    13. LSTAT \% lower status of the population ; 저소득자비율(%), flaot, continuous
    14. MEDV Median value of owner-occupied homes in \$1000's ; 주택중앙값, float, continuous
    """
    header_list = [
        "CRIM",
        "ZN",
        "INDUS",
        "CHAS",
        "NOX",
        "RM",
        "AGE",
        "DIS",
        "RAD",
        "TAX",
        "PTRATIO",
        "B",
        "LSTAT",
        "MEDV",
    ]
    housing_df = pd.read_csv(
        "housing.data", header=None, delim_whitespace=True, names=header_list
    )

    # type casting
    dtype_dic = {"CHAS": "int64", "TAX": "int64"}
    housing_df = housing_df.astype(dtype_dic)
    housing_df.dtypes

    return housing_df


def remove_outlier(housing_df):
    """
    # multivariate outlier detection techniques. 
    # Cook's distance
    """
    # exclude binary variables
    housing_df_woB = housing_df.drop(columns=["CHAS"])

    X = housing_df_woB.drop(columns=["MEDV"])
    y = housing_df_woB["MEDV"]

    # add a constant to fit the intercept
    X["CONSTANT"] = 1

    model = sm.OLS(y, X).fit()
    ols_inf = model.get_influence()
    Di = ols_inf.summary_frame().cooks_d

    # print(Di.sort_values(ascending=False)[:10])

    housing_df_o = housing_df_woB[Di < 0.01]
    housing_df_o["CHAS"] = housing_df["CHAS"][housing_df_o.index]

    return housing_df_o


def normalize_df(housing_df_o, scaler):
    """
    normalization
    # scaler = StandardScaler()
    # scaler = MinMaxScaler()
    """
    fitted = scaler.fit(housing_df_o)
    # print(fitted.mean_)

    output = scaler.transform(housing_df_o)
    housing_df_o_n = pd.DataFrame(
        output, columns=housing_df_o.columns, index=list(housing_df_o.index.values)
    )

    return housing_df_o_n


def review_df(housing_df_o_n):
    # summary statistics
    housing_df_o_n.describe()

    # boxplot
    housing_df_o_n.boxplot(figsize=(15, 7))

    # hitsogram
    housing_df_o_n.hist(bins=50, figsize=(20, 15))

    # non-normal dist correlation - spearman
    corr = housing_df_o_n.corr(method="spearman")  #'pearson','kendall','spearman'
    corr.style.background_gradient(cmap="coolwarm").set_precision(2)

    # pairwise scatter matrix plot
    scatter_matrix(
        housing_df_o_n, diagonal="kde", color="b", alpha=0.3, figsize=(20, 15)
    )


def get_tree_based_feature_selection(housing_df_o_n):
    """
    Tree-based feature selection : ExtraTreesClassifier
    return top 10 feature list
    """

    X = housing_df_o_n.drop(columns=["MEDV"])  # .iloc[:,0:13] #independent columns
    y = housing_df_o_n["MEDV"]  # target column 'MEDV'
    y = np.round(housing_df_o_n["MEDV"])

    model = ExtraTreesClassifier()
    model.fit(X, y)

    # print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

    # Plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    # feat_importances.nlargest(10).plot(kind="barh")
    # plt.show()

    tree_target_features = feat_importances.nlargest(10).index.to_list()

    return tree_target_features


def get_rfe_feature_selection(housing_df_o_n):
    """
    # RFE recursive feature elimination
    # model : RandomForestRegressor
    # return top 6 feature list
    """
    X = housing_df_o_n.drop(columns=["MEDV"])
    y = housing_df_o_n["MEDV"]  # target column

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=0
    )

    model = RandomForestRegressor()

    r2_list = []
    for i in range(1, X.shape[1] + 1):
        rfe = RFE(model, n_features_to_select=i)
        fit = rfe.fit(X_train, y_train)

        # predict prices of X_test
        y_pred = fit.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        r2_list.append(r2)

    # plt.plot(range(1, X.shape[1] + 1), r2_list)
    # plt.xlabel("N_Features")
    # plt.ylabel("R2")
    # plt.show()

    # R2 saturated n_features
    rfe = RFE(model, n_features_to_select=5)
    fit = rfe.fit(X_train, y_train)

    rfe_target_features = X.columns[fit.support_].to_list()
    return rfe_target_features


def randomforest_fit_and_evaluate(housing_df_o_n, target_features):
    """
    Random Forest Regressor
    """
    # feat_importances
    X = housing_df_o_n[target_features]
    # X = housing_df_o_n.drop(columns=["MEDV"])
    y = housing_df_o_n["MEDV"]  # target column

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=0
    )

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_train)

    print("\n[Random Forest Regressor]")
    print("Training Accuracy:", model.score(X_train, y_train) * 100)
    print("Testing Accuracy:", model.score(X_test, y_test) * 100)
    print("Mean Absolute Error:", mean_absolute_error(model.predict(X), y))
    print()

    # plt.scatter(y_train, y_pred)
    # plt.xlabel("Prices")
    # plt.ylabel("Predicted prices")
    # plt.title("Prices vs Predicted prices")
    # plt.show()

    # #check residuals
    # plt.scatter(y_pred, y_train-y_pred)
    # plt.xlabel("Predicted")
    # plt.ylabel("Residuals")
    # plt.title("Predicted vs residuals")
    # plt.show()

    # # Checking Normality of errors
    # (y_train-y_pred).plot.hist(bins=30)
    # plt.xlabel("Residuals")
    # plt.title("Checking Normality of errors")
    # plt.show()


def radomforest_cross_validation(housing_df_o_n):
    """
    RandomForestRegressor
    5 fold cross validation
    GridSearchCV - RFE on 'n_features_to_select'
    """
    # data
    X = housing_df_o_n.drop(columns=["MEDV"])
    y = housing_df_o_n["MEDV"]  # target column

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=0
    )

    # specify regression model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # cross-validation scheme
    folds = KFold(n_splits=5, shuffle=True, random_state=100)

    # hyperparameters range
    hyper_params = [{"n_features_to_select": list(range(1, 13))}]

    # perform grid search
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    rfe = RFE(model)

    # call GridSearchCV()
    model_cv = GridSearchCV(
        estimator=rfe,
        param_grid=hyper_params,
        scoring="r2",
        cv=folds,
        verbose=1,
        return_train_score=True,
    )

    # fit the model
    model_cv.fit(X_train, y_train)

    # cv results
    cv_results = pd.DataFrame(model_cv.cv_results_)

    # # plotting cv results
    # plt.figure(figsize=(15,7))

    # plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_test_score"])
    # plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_train_score"])
    # plt.xlabel('number of features')
    # plt.ylabel('r-squared')
    # plt.title("Optimal Number of Features")
    # plt.legend(['test score', 'train score'], loc='upper left')

    print("\n[Random Forest Regressor; RFE GridSearch 5-fold cross-validation]")
    print(
        cv_results[
            ["param_n_features_to_select", "mean_train_score", "mean_test_score"]
        ]
    )


def xgboost_hyperparameters_grid_search(housing_df_o_n, params):

    print("\n[XGBoost hyperparameters GridSearch]")
    # input data
    X = housing_df_o_n.drop(columns=["MEDV"])
    y = housing_df_o_n["MEDV"]  # target column

    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # convert to DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # init hyper-parameters
    num_boost_round = 1000
    # params = {
    #     "max_depth": 5,
    #     "min_child_weight": 8,
    #     "subsample": 0.9,
    #     "colsample_bytree": 0.9,
    #     "eta": 0.3,
    #     "objective": "reg:squarederror",
    #     "learning_rate": 0.1,
    #     "eval_metric": "mae",
    # }

    #####################################################################
    # grid_search - num_boost_round
    # k-fold cross-validation
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=5,
        metrics={"mae"},
        early_stopping_rounds=10,
    )
    best_round = cv_results["test-mae-mean"].idxmin()
    best_mae = cv_results["test-mae-mean"].min()
    print("Best MAE: {:.2f} with {} rounds".format(best_mae, best_round))
    num_boost_round = best_round  # 137

    #####################################################################
    ## grid_search - max_depth, min_child_weight
    gridsearch_params = [
        (max_depth, min_child_weight)
        for max_depth in range(5, 14)
        for min_child_weight in range(1, 10)
    ]

    # Define initial best params and MAE
    min_mae = float("Inf")
    best_params = None
    for max_depth, min_child_weight in gridsearch_params:
        # Update our parameters
        params["max_depth"] = max_depth
        params["min_child_weight"] = min_child_weight
        # Run CV
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            seed=42,
            nfold=5,
            metrics={"mae"},
            early_stopping_rounds=10,
        )
        # Update best MAE
        mean_mae = cv_results["test-mae-mean"].min()
        boost_rounds = cv_results["test-mae-mean"].argmin()
        if mean_mae < min_mae:
            min_mae = mean_mae
            best_params = (max_depth, min_child_weight)

    best_max_depth = best_params[0]
    best_min_child_weight = best_params[1]
    print(
        "Best max_depth, min_child_weight: {}, {}, MAE: {}".format(
            best_max_depth, best_min_child_weight, min_mae
        )
    )
    params["max_depth"] = best_max_depth
    params["min_child_weight"] = best_min_child_weight

    #####################################################################
    ## grid_search - subsample, colsample
    gridsearch_params = [
        (subsample, colsample)
        for subsample in [i / 10.0 for i in range(5, 11)]
        for colsample in [i / 10.0 for i in range(5, 11)]
    ]

    min_mae = float("Inf")
    best_params = None
    # We start by the largest values and go down to the smallest
    for subsample, colsample in reversed(gridsearch_params):
        # We update our parameters
        params["subsample"] = subsample
        params["colsample_bytree"] = colsample
        # Run CV
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            seed=42,
            nfold=5,
            metrics={"mae"},
            early_stopping_rounds=10,
        )
        # Update best score
        mean_mae = cv_results["test-mae-mean"].min()
        boost_rounds = cv_results["test-mae-mean"].argmin()
        if mean_mae < min_mae:
            min_mae = mean_mae
            best_params = (subsample, colsample)

    best_subsample = best_params[0]
    best_colsample = best_params[1]
    print(
        "Best subsample, colsample: {}, {}, MAE: {}".format(
            best_subsample, best_colsample, min_mae
        )
    )
    params["subsample"] = best_subsample
    params["colsample_bytree"] = best_colsample

    #####################################################################
    ## grid_search - ETA
    min_mae = float("Inf")
    best_params = None
    for eta in [1.0, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.005]:
        #     print("CV with eta={}".format(eta))
        # We update our parameters
        params["eta"] = eta
        # Run and time CV
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            seed=42,
            nfold=5,
            metrics=["mae"],
            early_stopping_rounds=10,
        )
        # Update best score
        mean_mae = cv_results["test-mae-mean"].min()
        boost_rounds = cv_results["test-mae-mean"].argmin()
        #     print("\tMAE {} for {} rounds\n".format(mean_mae, boost_rounds))
        if mean_mae < min_mae:
            min_mae = mean_mae
            best_params = eta

    best_eta = best_params
    print("Best ETA: {}, MAE: {}".format(best_eta, min_mae))
    params["eta"] = best_eta
    params["num_boost_round"] = best_round

    return params


def xgboost_evluation(housing_df_o_n, params):
    """
    # Accuracy
    """
    # input data
    X = housing_df_o_n.drop(columns=["MEDV"])
    y = housing_df_o_n["MEDV"]  # target column

    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    xg_reg = xgb.XGBRegressor(**params, early_stopping_rounds=10)
    xg_reg.fit(X_train, y_train)

    print("\n[XGBoost]")
    print("Training Accuracy:", xg_reg.score(X_train, y_train) * 100)
    print("Testing Accuracy:", xg_reg.score(X_test, y_test) * 100)
    print("R Squre:", r2_score(y, xg_reg.predict(X)) * 100)
    print("Mean Absolute Error:", mean_absolute_error(xg_reg.predict(X), y))

    # # Feature importance
    # xgb.plot_importance(xg_reg)
    # plt.rcParams["figure.figsize"] = [11, 8]
    # plt.show()

    return xg_reg, X


def explain_shap(xg_reg, X):
    """
    SHAP (SHapley Additive exPlanations) 
    is a game theoretic approach to explain the output of any machine learning model.    
    https://github.com/slundberg/shap
    """
    # explain the model's predictions using SHAP
    explainer = shap.TreeExplainer(xg_reg)
    shap_values = explainer.shap_values(X)

    # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
    shap.initjs()
    shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :])

    # visualize the training set predictions
    shap.initjs()
    shap.force_plot(explainer.expected_value, shap_values, X)

    # create a dependence plot to show the effect of a single feature across the whole dataset
    shap.dependence_plot("RM", shap_values, X)

    # create a dependence plot to show the effect of a single feature across the whole dataset
    shap.dependence_plot("LSTAT", shap_values, X)

    # summary plot
    shap.summary_plot(shap_values, X)
    shap.summary_plot(shap_values, X, plot_type="bar")


if __name__ == "__main__":

    # Load housing.data
    housing_df = load_data()
    # print(housing_df.info())

    # Outlier filtering
    housing_df_o = remove_outlier(housing_df)
    print(housing_df_o.info())

    # Normalization
    housing_df_o_n = normalize_df(
        housing_df, StandardScaler()
    )  # StandardScaler(), MinMaxScaler()
    # review_df(housing_df_o_n)

    # Feature selection
    tree_target_features = get_tree_based_feature_selection(housing_df_o_n)
    rfe_target_features = get_rfe_feature_selection(housing_df_o_n)

    # RandomForest regressor
    randomforest_fit_and_evaluate(housing_df_o_n, rfe_target_features)
    radomforest_cross_validation(housing_df_o_n)

    # XGBoost
    params = {
        "max_depth": 5,
        "min_child_weight": 8,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "eta": 0.3,
        "objective": "reg:squarederror",
        "learning_rate": 0.1,
        "eval_metric": "mae",
    }
    params = xgboost_hyperparameters_grid_search(housing_df_o_n, params)
    xg_reg, X = xgboost_evluation(housing_df_o_n, params)

    # Model Explaination
    # explain_shap(xg_reg, X)


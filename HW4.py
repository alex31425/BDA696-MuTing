import sys

import numpy as np
import pandas as pd
import statsmodels.api
from plotly import express as px
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestClassifier  # noqa
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# uncomment for loading these datasets
# from sklearn.datasets import load_breast_cancer,load_diabetes, load_wine
# install openpyxl


def main():
    # data = load_wine()
    # data = load_breast_cancer()
    data = load_boston()
    # data = load_diabetes()
    X = data.data
    y = data.target

    # Determine if response is continuous or boolean

    if y.dtype is str or bool is True:
        response_type = "boolean"
        print("---Response is boolean---")
    elif np.unique(y).size / y.size < 0.05:
        response_type = "boolean"
        print("---Response is boolean---")
    else:
        response_type = "continuous"
        print("---Response is continuous---")

    # Determine if the predictor continuous or boolean &
    # create plots for each variable type

    predictor_type = []
    for idx, column in enumerate(X.T):
        feature_name = data.feature_names[idx]

        predictor = statsmodels.api.add_constant(column)
        # Get the stats & plot
        if column.dtype is str or bool is True:
            v_type = "boolean"
            print(data.feature_names[idx], "is boolean")
            if response_type == "continuous":

                logistic_regression_model = statsmodels.api.GLM(y, predictor)
                logistic_regression_model_fitted = (
                    logistic_regression_model.fit()
                )  # noqa
                print(f"Variable: {feature_name}")
                print(logistic_regression_model_fitted.summary())
                t_value = round(logistic_regression_model_fitted.tvalues[1], 6)
                p_value = "{:.6e}".format(
                    logistic_regression_model_fitted.pvalues[1]
                )  # noqa

                # Categorical Predictor by Continuous Response

                fig = px.scatter(x=column, y=y)
                fig.update_layout(
                    title=f"Variable: {feature_name}: (t-value={t_value}) (p-value={p_value})",  # noqa
                    xaxis_title=f"Variable: {feature_name}",
                    yaxis_title="y",
                )
                fig.show()

            else:
                print(data.feature_names[idx], "is boolean")

                logistic_regression_model = statsmodels.api.GLM(y, predictor)
                logistic_regression_model_fitted = (
                    logistic_regression_model.fit()
                )  # noqa
                print(f"Variable: {feature_name}")
                print(logistic_regression_model_fitted.summary())
                t_value = round(logistic_regression_model_fitted.tvalues[1], 6)
                p_value = "{:.6e}".format(
                    logistic_regression_model_fitted.pvalues[1]
                )  # noqa

                # Continuous Predictor by Continuous Response

                fig = px.scatter(x=column, y=y)
                fig.update_layout(
                    title=f"Variable: {feature_name}: (t-value={t_value}) (p-value={p_value})",  # noqa
                    xaxis_title=f"Variable: {feature_name}",
                    yaxis_title="y",
                )
                fig.show()

        elif round((np.unique(X.T[idx]).size / X.T[idx].size), 2) <= 0.05:
            v_type = "boolean"
            print(data.feature_names[idx], "is boolean")
            if response_type == "continuous":

                logistic_regression_model = statsmodels.api.GLM(y, predictor)
                logistic_regression_model_fitted = (
                    logistic_regression_model.fit()
                )  # noqa
                print(f"Variable: {feature_name}")
                print(logistic_regression_model_fitted.summary())
                t_value = round(logistic_regression_model_fitted.tvalues[1], 6)
                p_value = "{:.6e}".format(
                    logistic_regression_model_fitted.pvalues[1]
                )  # noqa

                # Categorical Predictor by Continuous Response

                fig = px.histogram(x=column, y=y, histfunc="count")
                fig.update_layout(
                    title=f"Variable: {feature_name}: (t-value={t_value}) (p-value={p_value})",  # noqa
                    xaxis_title=f"Variable: {feature_name}",
                    yaxis_title="Response",
                )
                fig.show()

            else:
                print(data.feature_names[idx], "is boolean")

                logistic_regression_model = statsmodels.api.GLM(y, predictor)
                logistic_regression_model_fitted = (
                    logistic_regression_model.fit()
                )  # noqa
                print(f"Variable: {feature_name}")
                print(logistic_regression_model_fitted.summary())
                t_value = round(logistic_regression_model_fitted.tvalues[1], 6)
                p_value = "{:.6e}".format(
                    logistic_regression_model_fitted.pvalues[1]
                )  # noqa

                # Categorical Predictor by Continuous Response

                fig = px.scatter(x=column, y=y)
                fig.update_layout(
                    title=f"Variable: {feature_name}: (t-value={t_value}) (p-value={p_value})",  # noqa
                    xaxis_title=f"Variable: {feature_name}",
                    yaxis_title="y",
                )
                fig.show()
        else:
            print(data.feature_names[idx], "is continuous")
            v_type = "continuous"

            if response_type == "continuous":

                linear_regression_model = statsmodels.api.OLS(y, predictor)
                linear_regression_model_fitted = linear_regression_model.fit()
                print(f"Variable: {feature_name}")
                print(linear_regression_model_fitted.summary())
                t_value = round(linear_regression_model_fitted.tvalues[1], 6)
                p_value = "{:.6e}".format(
                    linear_regression_model_fitted.pvalues[1]
                )  # noqa

                # Continuous Predictor by Continuous Response
                # Plot the figure
                fig = px.scatter(x=column, y=y)
                fig.update_layout(
                    title=f"Variable: {feature_name}: (t-value={t_value}) (p-value={p_value})",  # noqa
                    xaxis_title=f"Variable: {feature_name}",
                    yaxis_title="y",
                )
                fig.show()
            else:
                linear_regression_model = statsmodels.api.OLS(y, predictor)
                linear_regression_model_fitted = linear_regression_model.fit()
                print(f"Variable: {feature_name}")
                print(linear_regression_model_fitted.summary())
                t_value = round(linear_regression_model_fitted.tvalues[1], 6)
                p_value = "{:.6e}".format(
                    linear_regression_model_fitted.pvalues[1]
                )  # noqa
                # Continuous Predictor by Categorical Response
                # Plot the figure
                fig = px.histogram(x=column, y=y)
                fig.update_layout(
                    title=f"Variable: {feature_name}: (t-value={t_value}) (p-value={p_value})",  # noqa
                    xaxis_title=f"Variable: {feature_name}",
                    yaxis_title="y",
                )
                fig.show()
        # create a list of each variable type
        predictor_type.append(v_type)

    print("***Difference with mean table***")

    # Create difference with mean table

    # create a temp table df_bin to store raw data
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data["target"]
    count_row = df.shape[0]
    count_column = df.shape[1]

    predictor_list = []
    for sublist in X:
        for item in sublist:
            predictor_list.append(item)
    p_min = min(predictor_list)
    p_max = max(predictor_list)
    p_range = p_max - p_min
    n_of_bin = 10
    bin_width = p_range / n_of_bin

    bin_list = [p_min]
    s = 0
    while s < p_max:
        s += bin_width
        bin_list.append(s)

    df_bin = pd.DataFrame({"predictor": pd.Series(predictor_list)})
    df_bin["LowerBin_UpperBin"] = pd.cut(
        x=pd.Series(predictor_list), bins=bin_list, include_lowest=True
    )
    bincenter = []
    for bin_n in df_bin["LowerBin_UpperBin"]:
        bincenter.append(bin_n.mid)
    df_bincenters = pd.DataFrame({"BinCenters": pd.Series(bincenter)})
    df_bin["BinCenters"] = df_bincenters
    df_bin["response"] = pd.Series(np.repeat(y, count_column - 1))

    # Groupby df_bin table to create a Difference with mean table

    df_bin_groupby = df_bin.groupby(("LowerBin_UpperBin"), as_index=False).agg(
        bin_mean=pd.NamedAgg(column="predictor", aggfunc="mean"),
        bin_count=pd.NamedAgg(column="predictor", aggfunc="count"),
    )

    bin_center_list = []
    for bin_center in df_bin_groupby["LowerBin_UpperBin"]:
        bin_center_list.append(bin_center.mid)

    df_bin_groupby["BinCenter"] = pd.Series(bin_center_list)

    PopulationMean = (np.sum(X) + np.sum(y)) / (count_row * count_column)
    df_bin["PopulationMean"] = PopulationMean
    df_bin_groupby["PopulationMean"] = PopulationMean

    MeanSquaredDiff = (
        df_bin_groupby["bin_mean"] - df_bin_groupby["PopulationMean"]
    ) ** 2
    df_bin_groupby["MeanSquaredDiff"] = MeanSquaredDiff

    print(df_bin_groupby)

    # Square the difference, sum them up and divide by number of bins
    print(
        f"THE unWeighted NUMBER IS : {df_bin_groupby['MeanSquaredDiff'].sum() / n_of_bin}"  # noqa
    )

    fig = px.bar(df_bin_groupby, x="BinCenter", y="bin_count")
    fig.show()

    # Difference with mean table (weighted)
    print("***Difference with mean table (weighted)***")

    df_bin_groupby_weighted = df_bin_groupby.copy()

    population_proportion = []
    for count in df_bin_groupby["bin_count"]:
        population_proportion.append(count / len(predictor_list))

    df_bin_groupby_weighted["PopulationProportion"] = pd.Series(
        population_proportion
    )  # noqa
    df_bin_groupby_weighted["MeanSquaredDiffWeighted"] = (
        df_bin_groupby_weighted["MeanSquaredDiff"]
        * df_bin_groupby_weighted["PopulationProportion"]
    )

    # Square the difference, sum them up and divide by number of bins
    print(
        f"THE Weighted NUMBER IS :\
        {df_bin_groupby_weighted['MeanSquaredDiffWeighted'].sum() / n_of_bin}"
    )

    print(df_bin_groupby_weighted)

    fig = px.bar(df_bin_groupby_weighted, x="BinCenter", y="bin_count")
    fig.show()

    # Random Forest Variable importance ranking
    print("***Random Forest Variable importance ranking***")

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = RandomForestRegressor()
    model.fit(X, y)

    # get importance
    importance = model.feature_importances_

    feature = []
    score = []
    for i, v in enumerate(importance):
        feature.append(data.feature_names[i])
        score.append(round(v, 5))

    df_ranking_vartype = pd.DataFrame(
        {"Feature": pd.Series(feature), "Score": pd.Series(score)}
    )
    df_ranking_vartype["Variable_type"] = predictor_type
    df_ranking_vartype_sort = df_ranking_vartype.sort_values(by=["Score"])

    # path that will save the ranking excel file
    path = "D:\PycharmProjects\BDA696-MuTing\Feature_Importance_and_type.xlsx"  # noqa
    df_ranking_vartype_sort.to_excel(path, index=False)
    print(df_ranking_vartype_sort)


if __name__ == "__main__":
    sys.exit(main())

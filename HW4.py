import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api
from plotly import express as px
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestClassifier  # noqa
from sklearn.ensemble import RandomForestRegressor

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
    n_of_bin = 10

    for idx, column in enumerate(X.T):
        feature_name = data.feature_names[idx]
        predictor = column
        target = data["target"]
        df = pd.DataFrame({feature_name: pd.Series(predictor)})
        df["target"] = target
        count_row = df.shape[0]
        p_min = df[feature_name].min()
        p_max = df[feature_name].max()
        p_range = p_max - p_min
        bin_width = p_range / n_of_bin
        # to include min number
        bin_list = [p_min - 1]
        s = p_min
        # +1 to include max number
        while s < p_max + 1:
            s += bin_width
            bin_list.append(round(s, 0))

        df_bin = df
        df_bin["LowerBin_UpperBin"] = pd.cut(
            x=df[feature_name],
            bins=bin_list,
            include_lowest=True,
            duplicates="drop",  # noqa
        )

        bincenter = []
        for bin_n in df_bin["LowerBin_UpperBin"]:
            bincenter.append(bin_n.mid)

            df_bin["BinCenters"] = pd.DataFrame(
                {"BinCenters": pd.Series(bincenter)}
            )  # noqa
            df_bin["response"] = df["target"]

        # Groupby df_bin table to create a Difference with mean table

        df_bin_groupby = df_bin.groupby(
            ("LowerBin_UpperBin"), as_index=False
        ).agg(  # noqa
            bin_mean=pd.NamedAgg(column=feature_name, aggfunc="mean"),
            bin_count=pd.NamedAgg(column=feature_name, aggfunc="count"),
        )

        bin_center_list = []
        for bin_center in df_bin_groupby["LowerBin_UpperBin"]:
            bin_center_list.append(bin_center.mid)

        df_bin_groupby["BinCenter"] = pd.Series(bin_center_list)

        PopulationMean = (np.sum(column)) / (count_row)
        df_bin["PopulationMean"] = PopulationMean
        df_bin_groupby["PopulationMean"] = PopulationMean

        MeanSquaredDiff = (
            df_bin_groupby["bin_mean"] - df_bin_groupby["PopulationMean"]
        ) ** 2
        df_bin_groupby["MeanSquaredDiff"] = MeanSquaredDiff

        # Square the difference, sum them up and divide by number of bins
        print(
            f"THE unWeighted NUMBER of {feature_name} IS : {df_bin_groupby['MeanSquaredDiff'].sum() / n_of_bin}"  # noqa
        )
        print(feature_name, df_bin_groupby)

        trace1 = go.Bar(
            x=df_bin_groupby["BinCenter"],
            y=df_bin_groupby["bin_count"],
            name="population",
        )
        layout = go.Layout(title_text="Binned Response Mean vs Population Mean")  # noqa

        trace2 = go.Scatter(
            x=df_bin_groupby["BinCenter"],
            y=df_bin_groupby["PopulationMean"],
            name="population mean",
        )
        trace3 = go.Scatter(
            x=df_bin_groupby["BinCenter"],
            y=df_bin_groupby["bin_mean"],
            name="Bin Mean",  # noqa
        )
        combined = [trace1, trace2, trace3]
        fig = go.Figure(data=combined, layout=layout)

        fig.show()

        # Difference with mean table (weighted)

        print("***Difference with mean table (weighted)***")

        df_bin_groupby_weighted = df_bin_groupby.copy()

        population_proportion = []
        for count in df_bin_groupby["bin_count"]:
            population_proportion.append(count / count_row)

        df_bin_groupby_weighted["PopulationProportion"] = pd.Series(
            population_proportion
        )
        df_bin_groupby_weighted["MeanSquaredDiffWeighted"] = (
            df_bin_groupby_weighted["MeanSquaredDiff"]
            * df_bin_groupby_weighted["PopulationProportion"]
        )

        # Square the difference, sum them up and divide by number of bins
        print(
            f"THE Weighted NUMBER of {feature_name} IS : {df_bin_groupby_weighted['MeanSquaredDiffWeighted'].sum() / n_of_bin}"  # noqa
        )

        print(feature_name, df_bin_groupby_weighted)

    # Random Forest Variable importance ranking
    print("***Random Forest Variable importance ranking***")

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

import itertools
import os
import sys

import matplotlib.pyplot as plt
import mysql.connector as sql
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import statsmodels.api
from plotly import express as px
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from statsmodels.stats.outliers_influence import \
    variance_inflation_factor as vif
from xgboost import XGBClassifier


def hyperlink(val):
    return '<a href="{}">{}</a>'.format(val, val)


def cat_cont_correlation_ratio(categories, values):
    """
    Correlation Ratio: https://en.wikipedia.org/wiki/Correlation_ratio
    SOURCE:
    1.) https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9  # noqa
    :param categories: Numpy array of categories
    :param values: Numpy array of values
    :return: correlation
    """
    f_cat, _ = pd.factorize(categories)
    cat_num = np.max(f_cat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = values[np.argwhere(f_cat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)  # noqa
    numerator = np.sum(
        np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2))  # noqa
    )
    denominator = np.sum(np.power(np.subtract(values, y_total_avg), 2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator / denominator)
    return eta


def cat_correlation(x, y, bias_correction=True, tschuprow=False):
    """
    Calculates correlation statistic for categorical-categorical association. # noqa
    The two measures supported are:
    1. Cramer'V ( default )
    2. Tschuprow'T

    SOURCES:
    1.) CODE: https://github.com/MavericksDS/pycorr
    2.) Used logic from:
        https://stackoverflow.com/questions/20892799/using-pandas-calculate-cram%C3%A9rs-coefficient-matrix
        to ignore yates correction factor on 2x2
    3.) Haven't validated Tschuprow

    Bias correction and formula's taken from : https://www.researchgate.net/publication/270277061_A_bias-correction_for_Cramer's_V_and_Tschuprow's_T # noqa

    Wikipedia for Cramer's V: https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V # noqa
    Wikipedia for Tschuprow' T: https://en.wikipedia.org/wiki/Tschuprow%27s_T # noqa
    Parameters:
    -----------
    x : list / ndarray / Pandas Series
        A sequence of categorical measurements
    y : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    bias_correction : Boolean, default = True
    tschuprow : Boolean, default = False
               For choosing Tschuprow as measure
    Returns:
    --------
    float in the range of [0,1]
    """
    corr_coeff = np.nan
    try:
        # x, y = fill_na(x), fill_na(y)
        crosstab_matrix = pd.crosstab(x, y)
        n_observations = crosstab_matrix.sum().sum()

        yates_correct = True
        if bias_correction:
            if crosstab_matrix.shape == (2, 2):
                yates_correct = False

        chi2, _, _, _ = stats.chi2_contingency(
            crosstab_matrix, correction=yates_correct
        )
        phi2 = chi2 / n_observations

        # r and c are number of categories of x and y
        r, c = crosstab_matrix.shape
        if bias_correction:
            phi2_corrected = max(
                0, phi2 - ((r - 1) * (c - 1)) / (n_observations - 1)
            )  # noqa
            r_corrected = r - ((r - 1) ** 2) / (n_observations - 1)
            c_corrected = c - ((c - 1) ** 2) / (n_observations - 1)
            if tschuprow:
                corr_coeff = np.sqrt(
                    phi2_corrected
                    / np.sqrt((r_corrected - 1) * (c_corrected - 1))  # noqa
                )
                return corr_coeff
            corr_coeff = np.sqrt(
                phi2_corrected / min((r_corrected - 1), (c_corrected - 1))
            )
            return corr_coeff
        if tschuprow:
            corr_coeff = np.sqrt(phi2 / np.sqrt((r - 1) * (c - 1)))
            return corr_coeff
        corr_coeff = np.sqrt(phi2 / min((r - 1), (c - 1)))
        return corr_coeff
    except Exception as ex:
        print(ex)
        if tschuprow:
            warnings.warn("Error calculating Tschuprow's T", RuntimeWarning)  # noqa
        else:
            warnings.warn("Error calculating Cramer's V", RuntimeWarning)  # noqa
        return corr_coeff


def pair_subsets(ss):
    return itertools.chain(
        *map(lambda x: itertools.combinations(ss, x), range(2, 3))
    )  # noqa


def plot_feature_importance(importance, names, model_type):
    # souece https://www.analyseup.com/learn-python-for-data-science/python-random-forest-feature-importance-plot.html # noqa
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Create a DataFrame using a Dictionary
    data = {
        "feature_names": feature_names,
        "feature_importance": feature_importance,
    }  # noqa
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=["feature_importance"], ascending=False, inplace=True)

    # Define size of bar plot
    plt.figure(figsize=(10, 8))
    # Plot Searborn bar chart
    sns.barplot(x=fi_df["feature_importance"], y=fi_df["feature_names"])
    # Add chart labels
    plt.title(model_type + "FEATURE IMPORTANCE")
    plt.xlabel("FEATURE IMPORTANCE")
    plt.ylabel("FEATURE NAMES")
    plt.savefig(f"{model_type}-FEATURE IMPORTANCE.png")


def plot_vif(importance, names, model_type):
    # souece https://www.analyseup.com/learn-python-for-data-science/python-random-forest-feature-importance-plot.html # noqa
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Create a DataFrame using a Dictionary
    data = {
        "feature_names": feature_names,
        "feature_importance": feature_importance,
    }  # noqa
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=["feature_importance"], ascending=False, inplace=True)

    # Define size of bar plot
    plt.figure(figsize=(10, 8))
    # Plot Searborn bar chart
    sns.barplot(x=fi_df["feature_importance"], y=fi_df["feature_names"])
    # Add chart labels
    plt.title(model_type)
    plt.xlabel("variance inflation factor")
    plt.ylabel("FEATURE NAMES")


def main():

    cnx = sql.connect(host = 'db',
        user="root", password="root", database="baseball"
    )  # pragma: allowlist secret
    query = """select * from data_final"""
    df_b = pd.read_sql(query, con=cnx)

    for column in df_b:
        df_b[column].fillna(0, inplace=True)

    df_b.head(100)
    df_M = df_b.drop(
        columns=["local_date", "game_id", "home_team_id", "away_team_id"]
    )  # noqa

    print(df_M.head())
    df_b.to_csv(r'/results/baseball.csv')

    X = df_M.iloc[:, 1:].to_numpy()
    y = df_M.iloc[:, 0].to_numpy()
    f = list(df_M.columns)
    f.remove("home_team_wins")

    # Determine if response is continuous or boolean

    if y.dtype is str or bool is True:
        response_type = "boolean"
        print("---Response is boolean---")
    elif np.unique(y).size < 5:
        response_type = "boolean"
        print("---Response is boolean---")
    else:
        response_type = "continuous"
        print("---Response is continuous---")

    # Determine if the predictor continuous or boolean &
    # create plots for each variable type

    predictor_type = []
    t_score = []
    stat_plot = {}
    con_array = np.array([])
    cat_array = np.array([])
    for idx, column in enumerate(X.T):

        feature_name = f[idx]
        predictor = statsmodels.api.add_constant(column)

        # Get the stats & plot

        if np.unique(X.T[idx]).size < 5:
            v_type = "boolean"
            cat_array = np.append(cat_array, column)
            # print(f[idx], "is boolean" )
            if response_type == "continuous":

                logistic_regression_model = statsmodels.api.GLM(y, predictor)
                logistic_regression_model_fitted = (
                    logistic_regression_model.fit()
                )  # noqa
                # print(f"Variable: {feature_name}")
                # print(logistic_regression_model_fitted.summary(), '\n')
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
                fig.write_html(file="/results/plot_cat_con.html", include_plotlyjs="cdn")  # noqa

                # fig.show()
            else:

                logistic_regression_model = statsmodels.api.GLM(y, predictor)
                logistic_regression_model_fitted = (
                    logistic_regression_model.fit()
                )  # noqa
                # print(f"Variable: {feature_name}")
                # print(logistic_regression_model_fitted.summary(), '\n')
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
                fig.write_html(file="/results/plot_cat_cat.html", include_plotlyjs="cdn")  # noqa

                # fig.show()
        else:
            # print(f[idx], "is continuous")
            v_type = "continuous"
            con_array = np.append(cat_array, column)  # noqa
            if response_type == "continuous":

                linear_regression_model = statsmodels.api.OLS(y, predictor)
                linear_regression_model_fitted = linear_regression_model.fit()
                # print(f"Variable: {feature_name}")
                # print(linear_regression_model_fitted.summary(), '\n')
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
                fig.write_html(file="/results/plot_con_con.html", include_plotlyjs="cdn")  # noqa

                # fig.show()
            else:
                linear_regression_model = statsmodels.api.OLS(y, predictor)
                linear_regression_model_fitted = linear_regression_model.fit()
                # print(f"Variable: {feature_name}")
                # print(linear_regression_model_fitted.summary(), '\n')
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

                # fig.show()

                # plot link

                link = f"/results/stat-summary{feature_name}.html"
                fig.write_html(file=link, include_plotlyjs="cdn")
                stat_plot[feature_name] = link


        predictor_type.append(v_type)
        t_score.append(t_value)

    # Split dataset predictors between categoricals and continuous

    pd_PT = pd.DataFrame(predictor_type, columns=["type"])

    pd_PT["name"] = f

    df_PT_c = (
        pd_PT[pd_PT["type"] == "continuous"].drop(["type"], axis=1)
    ).copy()  # noqa
    df_PT_b = (pd_PT[pd_PT["type"] == "boolean"].drop(["type"], axis=1)).copy()
    pd_PT = pd_PT.drop(["type"], axis=1)
    con_list = df_PT_c["name"].to_list()  # noqa
    cat_list = df_PT_b["name"].to_list()  # noqa
    f_list = pd_PT["name"].to_list()  # noqa

    # Split dataset predictors between categoricals and continuous

    corr_plot = {}
    df_con_cat = (pd.DataFrame(X)).T
    df_con_cat["type"] = predictor_type
    df_con_cat["Feature"] = f

    df_con = df_con_cat[df_con_cat["type"] == "continuous"]
    df_cat = df_con_cat[df_con_cat["type"] == "boolean"]
    df_con_d = df_con.drop(["type"], axis=1).T
    df_cat_d = df_cat.drop(["type"], axis=1).T
    df_con_cat_d = df_con_cat.drop(["type"], axis=1).T

    df_con_cat_d.columns = df_con_cat_d.loc["Feature"]
    df_con_cat_d = df_con_cat_d.drop(["Feature"])
    df_con_d.columns = df_con_d.loc["Feature"]
    df_cat_d.columns = df_cat_d.loc["Feature"]
    df_con_d = df_con_d.drop(["Feature"])
    df_cat_d = df_cat_d.drop(["Feature"])

    # convert column data type
    df_con_d[
        df_con_d.select_dtypes(["object"]).columns
    ] = df_con_d.select_dtypes(  # noqa
        ["object"]
    ).apply(
        lambda x: x.astype("float")
    )

    df_cat_d[
        df_cat_d.select_dtypes(["object"]).columns
    ] = df_cat_d.select_dtypes(  # noqa
        ["object"]
    ).apply(
        lambda x: x.astype("float")
    )

    df_con_cat_d[
        df_con_cat_d.select_dtypes(["object"]).columns
    ] = df_con_cat_d.select_dtypes(["object"]).apply(
        lambda x: x.astype("float")
    )  # noqa

    # print(df_con_d)
    # print(df_cat_d)
    # print(df_con_cat_d)

    corr_df_con_d = df_con_d.corr(method="pearson")
    # print(f"correlation metrics : Continuous / Continuous pairs")
    # print(
    #     corr_df_con_d.sort_values(
    #         by=list(corr_df_con_d.columns), ascending=False
    #     )
    # )

    # corr_con plot

    trace_con = go.Heatmap(
        x=df_con_d.columns,
        y=df_con_d.columns,
        z=corr_df_con_d.to_numpy(),
        type="heatmap",
    )
    fig_con = go.Figure(data=[trace_con])
    plot_con = f"/results/plot_con.html"
    fig_con.write_html(file=plot_con, include_plotlyjs="cdn")
    corr_plot["con"] = plot_con
    # fig_con.show()

    plt.figure(figsize=(40, 20))
    sns.heatmap(
        corr_df_con_d, vmax=1, square=True, annot=True, cmap="cubehelix"
    )  # noqa
    # plt.show()

    # corr_df_con_d_html = corr_df_con_d.to_html()
    # corr_df_con_d_file = open("corr_df_con_d.html", "w")
    # corr_df_con_d_file.write(corr_df_con_d_html)
    # corr_df_con_html = "corr_df_con_html.html"
    # corr_df_con_d_file.close()
    # corr_plot["corr_df_con_d"] = corr_df_con_html
    # print(corr_plot)
    #

    # correlation metrics : Continuous / Categorical pair

    corr_list = []
    con_N = []
    cat_N = []

    for con_name, con_value in df_con_d.iteritems():
        # print(con_name,con_value.values)
        con_N.append(con_name)

        for cat_name, cat_value in df_cat_d.iteritems():
            corr_con_cat = (
                cat_cont_correlation_ratio(cat_value.values, con_value.values)
            ).tolist()
            # print(corr_con_cat,end=" ")
            corr_list.append(corr_con_cat)

    col = df_cat_d.shape[1]
    try:
        if col != 0:
            row = int(len(corr_list) / col)
            corr_list = np.reshape(corr_list, (row, col))
            df_corr_con_cat = pd.DataFrame(
                corr_list, columns=df_cat_d.columns, index=con_N
            )
            # print(
            #     df_corr_con_cat.sort_values(
            #         by=list(df_cat_d.columns), ascending=False
            #     )
            # )

            corr_df_con_cat_d_html = df_corr_con_cat.to_html()
            corr_df_con_cat_d_file = open("corr_df_con_cat_d.html", "w")
            corr_df_con_cat_d_file.write(corr_df_con_cat_d_html)
            corr_df_con_cat_d = "corr_df_con_cat_d.html"
            corr_df_con_cat_d_file.close()
            corr_plot["corr_df_con_cat_d"] = corr_df_con_cat_d

            # print(f"correlation metrics : Categorical / Categorical pairs")
            trace_con_cat = go.Heatmap(
                x=df_cat_d.columns, y=con_N, z=df_corr_con_cat, type="heatmap"
            )
            fig_con_cat = go.Figure(data=[trace_con_cat])

            plot_con_cat = f"/results/plot_con_cat.html"
            fig_con_cat.write_html(file=plot_con_cat, include_plotlyjs="cdn")  # noqa
            corr_plot["con_cat"] = plot_con_cat
            # fig_con_cat.show()
        else:
            pass
    except ZeroDivisionError:
        print("no cat predictor")

    # categorical-categorical association

    corr_list = []
    cat_N = []
    for cat_name_1, cat_value_1 in df_cat_d.iteritems():
        cat_N.append(cat_name_1)

        for cat_name_2, cat_value_2 in df_cat_d.iteritems():
            corr_cat_cat = cat_correlation(
                cat_value_1.values, cat_value_2.values
            )  # noqa
            corr_list.append(corr_cat_cat)

    col = df_cat_d.shape[1]
    try:
        if col != 0:
            row = int(len(corr_list) / col)
            corr_list = np.reshape(corr_list, (row, col))
            df_corr_cat_cat = pd.DataFrame(
                corr_list, columns=df_cat_d.columns, index=cat_N
            )
            # print(
            #     df_corr_cat_cat.sort_values(
            #         by=list(df_cat_d.columns), ascending=False
            #     )
            # )

            corr_df_cat_d_html = df_corr_cat_cat.to_html()
            corr_df_cat_d = open("corr_df_cat_d.html", "w")
            corr_df_cat_d.write(corr_df_cat_d_html)
            corr_df_cat_html = "corr_df_cat_d.html"
            corr_df_cat_d.close()
            corr_plot["corr_df_cat_d"] = corr_df_cat_html

            trace_cat = go.Heatmap(
                x=df_cat_d.columns, y=cat_N, z=df_corr_cat_cat, type="heatmap"
            )

            fig_cat = go.Figure(data=[trace_cat])
            plot_cat = f"/results/plot_cat.html"
            fig_cat.write_html(file=plot_cat, include_plotlyjs="cdn")
            corr_plot["cat"] = plot_cat
            # fig_cat.show()
        else:
            pass
    except ZeroDivisionError:
        print("no cat predictor")

    # Create difference with mean table

    # create a temp table df_bin to store raw data
    n_of_bin = 9
    diff_plot = {}
    BF_plot = {}
    appended_data = []
    MeanSquaredDiffWeighted = []
    for idx, column in enumerate(X.T):
        feature_name = f[idx]
        predictor = column
        target = df_M["home_team_wins"]
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

        df_bin["Name"] = pd.Series(np.repeat(feature_name, count_row))
        df_bin["Type"] = pd.Series(np.repeat(predictor_type, count_row))

        # Groupby df_bin table to create a Difference with mean table

        df_response = df_bin.groupby(("LowerBin_UpperBin"), as_index=False)[
            "response"
        ].sum()

        df_bin_groupby = df_bin.groupby(
            ("LowerBin_UpperBin"), as_index=False
        ).agg(  # noqa
            bin_mean=pd.NamedAgg(column=feature_name, aggfunc="mean"),
            bin_count=pd.NamedAgg(column=feature_name, aggfunc="count"),
        )
        df_bin_groupby["binned_response_mean"] = (
            df_response["response"] / df_bin_groupby["bin_count"]
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
        #  print(
        #      f"THE unWeighted NUMBER of {feature_name} IS : {df_bin_groupby['MeanSquaredDiff'].sum() / n_of_bin}"  # noqa
        #  )
        #  print(feature_name, df_bin_groupby)

        trace1 = go.Bar(
            x=df_bin_groupby["BinCenter"],
            y=df_bin_groupby["bin_count"],
            name="bin_count",
            yaxis="y2",
            opacity=0.5,
        )
        y2 = go.layout.YAxis(title="bin_count", overlaying="y", side="right")

        trace2 = go.Scatter(
            x=df_bin_groupby["BinCenter"],
            y=df_bin_groupby["PopulationMean"],
            name="population mean",
            mode="lines",
        )
        trace3 = go.Scatter(
            x=df_bin_groupby["BinCenter"],
            y=df_bin_groupby["binned_response_mean"],
            name="Bin Mean",
        )

        layout = go.Layout(
            title="Binned Response Mean vs Population Mean",
            xaxis_title=f"predictor: {feature_name}",
            yaxis_title=f"Binned Response Mean",  # noqa
            yaxis2=y2,
        )

        combined = [trace1, trace2, trace3]
        fig = go.Figure(data=combined, layout=layout)
        # fig.show()

        # plot link

        link = f"/results/diffmean{feature_name}.html"
        fig.write_html(file=link, include_plotlyjs="cdn")
        diff_plot[feature_name] = link

        feature_link = []
        for v in diff_plot.values():
            script_dir = os.path.dirname(__file__)
            rel_path = v
            abs_file_path = os.path.join(script_dir, rel_path)
            feature_link.append(abs_file_path)
            df_link = pd.DataFrame(feature_link)
            df_link.columns = ["feature_link"]

            df_link.style.format(hyperlink)

            # Difference with mean table (weighted)

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
        # print(
        #     f"THE Weighted NUMBER of {feature_name} IS : {df_bin_groupby_weighted['MeanSquaredDiffWeighted'].sum() / n_of_bin}"  # noqa
        # )
        # MeanSquaredDiffWeighted.append(df_bin_groupby_weighted['MeanSquaredDiffWeighted'].sum() / n_of_bin) # noqa

        # print(feature_name, df_bin_groupby_weighted)

        # Brute-Force" variable combination

        df_BF = df_bin.copy()
        df_BF.drop(["BinCenters", "target"], axis=1, inplace=True)
        df_BF_g = df_BF.groupby(("LowerBin_UpperBin"), as_index=False).agg(
            P_mean=pd.NamedAgg(column=feature_name, aggfunc="mean")
        )
        df_BF_g["Name"] = pd.Series(np.repeat(feature_name, n_of_bin + 1))

        df_BF_g["P_mean"]
        appended_data.append(df_BF_g["P_mean"].to_list())

    df_BF_C = pd.DataFrame(appended_data, index=f)
    df_BF_C["type"] = predictor_type
    df_BF_C_con_cat = df_BF_C.drop(["type"], axis=1).copy()
    df_BF_C_con = (
        df_BF_C[df_BF_C["type"] == "continuous"].drop(["type"], axis=1)
    ).copy()
    df_BF_C_cat = (
        df_BF_C[df_BF_C["type"] == "boolean"].drop(["type"], axis=1)
    ).copy()  # noqa
    df_BF_C_con = df_BF_C_con.T
    df_BF_C_con_cat = df_BF_C_con_cat.T
    df_BF_C_cat = df_BF_C_cat.T

    # CON vs CON
    tups_con = list(pair_subsets(df_BF_C_con))
    columns_con = df_BF_C_con.columns
    array_con = []
    for a_con, b_con in itertools.combinations(columns_con, 2):
        # print(a_con,b_con)
        meah_con = np.array(
            np.meshgrid(df_BF_C_con[a_con], df_BF_C_con[b_con])
        ).T.reshape(-1, 2)
        array_con.append(meah_con)
    # print(array_con)
    CM_con = []
    for ar_con in array_con:
        ar_con = np.array(ar_con)
        for i_con in ar_con:
            m_con = np.mean(i_con)
            CM_con.append(m_con)
    # print(CM_con)
    CM_con = np.array(CM_con)
    CM_con_s = np.array_split(CM_con, len(tups_con))
    # print(CM_con_s)
    for ind_con, sArr_con in enumerate(CM_con_s):
        sArr_con = sArr_con.reshape(df_BF_C_con.shape[0], df_BF_C_con.shape[0])
        # print(f'{tups_con[ind_con]}\n {sArr_con}\n')
        print(f"{tups_con[ind_con]}\n{pd.DataFrame(data=sArr_con)}")

        trace_BF_con = go.Heatmap(
            z=sArr_con,
            type="heatmap",
        )
        fig_BF_con = go.Figure(data=[trace_BF_con])
        plot_BF_con = f"/results/BF-{tups_con[ind_con]}.html"
        fig_BF_con.write_html(file=plot_BF_con, include_plotlyjs="cdn")
        BF_plot["con"] = plot_BF_con
        # fig_BF_con.show()

    # CON vs CAT
    # tups_con_cat = list(pair_subsets(df_BF_C_con_cat))
    # columns_con_cat = df_BF_C_con_cat.columns
    # array_con_cat = []
    # for a_con_cat, b_con_cat in itertools.combinations(columns_con_cat, 2):
    #     # print(a_con_cat,b_con_cat)
    #     meah_con_cat = np.array(np.meshgrid(df_BF_C_con_cat[a_con_cat], df_BF_C_con_cat[b_con_cat])).T.reshape(-1, 2) # noqa
    #     array_con_cat.append(meah_con_cat)
    # # print(array_con_cat)
    # CM_con_cat = []
    # for ar_con_cat in array_con_cat:
    #     ar_con_cat = np.array(ar_con_cat)
    #     for i_con_cat in ar_con_cat:
    #         m_con_cat = (np.mean(i_con_cat)-PopulationMean)
    #         CM_con_cat.append(m_con_cat)
    # # print(CM_con_cat)
    # CM_con_cat = np.array(CM_con_cat)
    # CM_con_cat_s = np.array_split(CM_con_cat, len(tups_con_cat))
    # # print(CM_con_cat_s)
    # for ind_con_cat, sArr_con_cat in enumerate(CM_con_cat_s):
    #     sArr_con_cat = sArr_con_cat.reshape(df_BF_C_con_cat.shape[0], df_BF_C_con_cat.shape[0]) # noqa
    #     # print(f'{tups_con_cat[ind_con_cat]}\n {sArr_con_cat}\n')
    #     # print(f'{tups_con_cat[ind_con_cat]}\n{pd.DataFrame(data=sArr_con_cat)}') # noqa
    #
    #     # Plot
    #     trace_BF_con_cat = go.Heatmap(
    #         z=sArr_con_cat,
    #         type="heatmap",
    #     )
    #     fig_BF_con_cat = go.Figure(data=[trace_BF_con_cat])
    #     plot_BF_con_cat = f"{tups_con_cat[ind_con_cat]}.html"
    #     fig_BF_con_cat.write_html(
    #         file=plot_BF_con_cat, include_plotlyjs="cdn"
    #     )  # noqa
    #     BF_plot["con_cat"] = plot_BF_con_cat
    #     # fig_BF_con_cat.show()
    #
    #
    #
    # # CAT vs CAT
    # tups_cat_cat = list(pair_subsets(df_BF_C_cat))
    # columns_cat_cat = df_BF_C_cat.columns
    # array_cat_cat = []
    # for a_cat_cat, b_cat_cat in itertools.combinations(columns_cat_cat, 2):
    #     # print(a_con_cat,b_cat_cat)
    #     meah_cat_cat = np.array(np.meshgrid(df_BF_C_cat[a_cat_cat], df_BF_C_cat[b_cat_cat])).T.reshape(-1, 2) # noqa
    #     array_cat_cat.append(meah_cat_cat)
    # # print(array_con_cat)
    # CM_cat_cat = []
    # for ar_cat_cat in array_cat_cat:
    #     ar_cat_cat = np.array(ar_cat_cat)
    #     for i_cat_cat in ar_cat_cat:
    #         m_cat_cat = (np.mean(i_cat_cat) - PopulationMean)
    #         CM_cat_cat.append(m_cat_cat)
    # # print(CM_cat_cat)
    # CM_cat_cat = np.array(CM_cat_cat)
    # CM_cat_cat_s = np.array_split(CM_cat_cat, len(tups_cat_cat))
    # # print(CM_cat_cat_s)
    # for ind_cat_cat, sArr_cat_cat in enumerate(CM_cat_cat_s):
    #     sArr_cat_cat = sArr_cat_cat.reshape(df_BF_C_cat.shape[0], df_BF_C_cat.shape[0]) # noqa
    #     # print(f'{tups_cat_cat[ind_cat_cat]}\n {sArr_cat_cat}\n')
    #     # print(f'{tups_cat_cat[ind_cat_cat]}\n{pd.DataFrame(data=sArr_cat_cat)}') # noqa
    #
    #     # Plot
    #     trace_BF_cat = go.Heatmap(
    #         z=sArr_cat_cat,
    #         type="heatmap",
    #     )
    #     fig_BF_cat = go.Figure(data=[trace_BF_cat])
    #     plot_BF_cat = f"BF-{tups_cat_cat[ind_cat_cat]}.html"
    #     fig_BF_cat.write_html(
    #         file=plot_BF_cat, include_plotlyjs="cdn"
    #     )  # noqa
    #     BF_plot["cat"] = plot_BF_cat
    #     # fig_BF_cat.show()

    # Machine Learning Models

    # Defining the categorical columns
    categoricalColumns = df_M.select_dtypes(include=[np.object]).columns

    onehot_categorical = OneHotEncoder(handle_unknown="ignore")

    categorical_transformer = Pipeline(steps=[("onehot", onehot_categorical)])

    # Defining the numerical columns
    numericalColumns = [
        col
        for col in df_M.select_dtypes(include=[np.float64]).columns
        if col not in ["home_team_wins"]
    ]

    scaler_numerical = StandardScaler()

    numerical_transformer = Pipeline(steps=[("scale", scaler_numerical)])

    preprocessorForCategoricalColumns = ColumnTransformer(  # noqa
        transformers=[("cat", categorical_transformer, categoricalColumns)],
        remainder="passthrough",
    )
    preprocessorForAllColumns = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categoricalColumns),
            ("num", numerical_transformer, numericalColumns),
        ],
        remainder="passthrough",
    )

    features = []
    features = df_M.drop(["home_team_wins"], axis=1)

    label = pd.DataFrame(df_M, columns=["price"])
    label_encoder = LabelEncoder()
    label = df_M["home_team_wins"]

    label = label_encoder.fit_transform(label)

    X_train, X_test, y_train, y_test = train_test_split(
        features, label, random_state=0
    )  # noqa

    # K-Nearest classification model

    model_name = "K-Nearest Neighbor Classifier"

    knnClassifier = KNeighborsClassifier()

    knn_model = Pipeline(
        steps=[
            ("preprocessorAll", preprocessorForAllColumns),
            ("classifier", knnClassifier),
        ]
    )

    knn_model.fit(X_train, y_train)

    y_pred_knn = knn_model.predict(X_test)

    y_test = label_encoder.transform(y_test)
    y_test = label_encoder.inverse_transform(y_test)
    y_pred_knn = label_encoder.inverse_transform(y_pred_knn)

    Accuracy_score = round(accuracy_score(y_test, y_pred_knn), 2)
    ax = plt.subplot()
    fig = sns.heatmap(
        confusion_matrix(y_test, y_pred_knn), annot=True, ax=ax, fmt=".1f"
    ).get_figure()
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.set_title(
        f"Confusion Matrix, {model_name}, Accuracy_score : {Accuracy_score}"
    )  # noqa
    ax.xaxis.set_ticklabels(["0", "1"])
    ax.yaxis.set_ticklabels(["0", "1"])
    fig.savefig(f"/results/Confusion Matrix-{model_name}.png")
    # plt.show()

    # Kernel SVM classification model

    model_name = "Kernel SVM Classifier"

    svmClassifier = SVC(kernel="rbf")

    svm_model = Pipeline(
        steps=[
            ("preprocessorAll", preprocessorForAllColumns),
            ("classifier", svmClassifier),
        ]
    )

    svm_model.fit(X_train, y_train)

    y_pred_svm = svm_model.predict(X_test)

    Accuracy_score = round(accuracy_score(y_test, y_pred_svm), 2)
    ax = plt.subplot()
    fig = sns.heatmap(
        confusion_matrix(y_test, y_pred_svm), annot=True, ax=ax, fmt=".1f"
    ).get_figure()
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.set_title(
        f"Confusion Matrix, {model_name}, Accuracy_score : {Accuracy_score}"
    )  # noqa
    ax.xaxis.set_ticklabels(["0", "1"])
    ax.yaxis.set_ticklabels(["0", "1"])
    fig.savefig(f"/results/Confusion Matrix-{model_name}.png")
    # plt.show()

    # Build decision tree regressor

    model_name = "Decision Tree Classifier"

    decisionTreeClassifier = DecisionTreeClassifier(
        random_state=0, max_features=25
    )  # noqa

    dtr_model = Pipeline(
        steps=[
            ("preprocessorAll", preprocessorForAllColumns),
            ("regressor", decisionTreeClassifier),
        ]
    )

    dtr_model.fit(X_train, y_train)

    y_pred_dtr = dtr_model.predict(X_test)

    y_test = label_encoder.inverse_transform(y_test)
    y_pred_dtr = label_encoder.inverse_transform(y_pred_svm)

    Accuracy_score = round(accuracy_score(y_test, y_pred_dtr), 2)
    ax = plt.subplot()
    fig = sns.heatmap(
        confusion_matrix(y_test, y_pred_dtr), annot=True, ax=ax, fmt=".1f"
    ).get_figure()
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.set_title(
        f"Confusion Matrix, {model_name}, Accuracy_score : {Accuracy_score}"
    )  # noqa
    ax.xaxis.set_ticklabels(["0", "1"])
    ax.yaxis.set_ticklabels(["0", "1"])
    fig.savefig(f"/results/Confusion Matrix-{model_name}.png")
    # plt.show()

    # Build Random Forest classification model

    model_name = "Random Forest Regressor"

    randomForestClassifier = RandomForestClassifier(
        n_estimators=100, max_depth=15, random_state=0
    )

    rfc_model = Pipeline(
        steps=[
            ("preprocessorAll", preprocessorForAllColumns),
            ("classifier", randomForestClassifier),
        ]
    )

    rfc_model.fit(X_train, y_train)

    y_pred_rfc = rfc_model.predict(X_test)

    y_test = label_encoder.inverse_transform(y_test)
    y_pred_svm = label_encoder.inverse_transform(y_pred_svm)
    y_test = label_encoder.inverse_transform(y_test)
    y_pred_rfc = label_encoder.inverse_transform(y_pred_rfc)

    Accuracy_score = round(accuracy_score(y_test, y_pred_rfc), 2)
    ax = plt.subplot()
    fig = sns.heatmap(
        confusion_matrix(y_test, y_pred_rfc), annot=True, ax=ax, fmt=".1f"
    ).get_figure()
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.set_title(
        f"Confusion Matrix, {model_name}, Accuracy_score : {Accuracy_score}"
    )  # noqa
    ax.xaxis.set_ticklabels(["0", "1"])
    ax.yaxis.set_ticklabels(["0", "1"])
    fig.savefig(f"/results/Confusion Matrix-{model_name}.png")
    # plt.show()

    # Build XGBoost model

    model_name = "XGBoost"

    xgboost = XGBClassifier(random_state=0)

    xg_model = Pipeline(
        steps=[
            ("preprocessorAll", preprocessorForAllColumns),
            ("regressor", xgboost),
        ]  # noqa
    )

    xg_model.fit(X_train, y_train)

    y_pred_xg = xg_model.predict(X_test)

    y_test = label_encoder.inverse_transform(y_test)
    y_pred_xg = label_encoder.inverse_transform(y_pred_xg)

    Accuracy_score = round(accuracy_score(y_test, y_pred_xg), 2)
    ax = plt.subplot()
    fig = sns.heatmap(
        confusion_matrix(y_test, y_pred_xg), annot=True, ax=ax, fmt=".1f"
    ).get_figure()
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.set_title(
        f"Confusion Matrix, {model_name}, Accuracy_score : {Accuracy_score}"
    )  # noqa
    ax.xaxis.set_ticklabels(["0", "1"])
    ax.yaxis.set_ticklabels(["0", "1"])
    fig.savefig(f"/results/Confusion Matrix-{model_name}.png")
    # plt.show()

    # Comparative of different classification algorithms

    uniqueValues, occurCount = np.unique(y_test, return_counts=True)
    frequency_actual = (occurCount[0], occurCount[1])

    uniqueValues, occurCount = np.unique(y_pred_knn, return_counts=True)
    frequency_predicted_knn = (occurCount[0], occurCount[1])

    uniqueValues, occurCount = np.unique(y_pred_rfc, return_counts=True)
    frequency_predicted_rfc = (occurCount[0], occurCount[1])

    uniqueValues, occurCount = np.unique(y_pred_dtr, return_counts=True)
    frequency_predicted_dtr = (occurCount[0], occurCount[1])

    uniqueValues, occurCount = np.unique(y_pred_svm, return_counts=True)
    frequency_predicted_svm = (occurCount[0], occurCount[1])

    uniqueValues, occurCount = np.unique(y_pred_xg, return_counts=True)
    frequency_predicted_xg = (occurCount[0], occurCount[1])

    n_groups = 2
    fig, ax = plt.subplots(figsize=(10, 5))
    index = np.arange(n_groups)
    bar_width = 0.1
    opacity = 0.8

    rects1 = plt.bar(  # noqa
        index,
        frequency_actual,
        bar_width,
        alpha=opacity,
        color="g",
        label="Actual",  # noqa
    )

    rects2 = plt.bar(  # noqa
        index + bar_width,
        frequency_predicted_knn,
        bar_width,
        alpha=opacity,
        color="pink",
        label="K-Nearest Neighbo - Predicted",
    )

    rects3 = plt.bar(  # noqa
        index + bar_width * 2,
        frequency_predicted_rfc,
        bar_width,
        alpha=opacity,
        color="y",
        label="Random Forest - Predicted",
    )

    rects4 = plt.bar(  # noqa
        index + bar_width * 3,
        frequency_predicted_dtr,
        bar_width,
        alpha=opacity,
        color="b",
        label="Decision Tree - Predicted",
    )

    rects5 = plt.bar(  # noqa
        index + bar_width * 4,
        frequency_predicted_svm,
        bar_width,
        alpha=opacity,
        color="red",
        label="Kernel SVM - Predicted",
    )

    rects6 = plt.bar(  # noqa
        index + bar_width * 5,
        frequency_predicted_xg,
        bar_width,
        alpha=opacity,
        color="purple",
        label="XGBoost - Predicted",
    )

    plt.xlabel("Result")
    plt.ylabel("Frequency")
    plt.title("Actual vs Predicted frequency.")
    plt.xticks(index + bar_width, ("0", "1"))
    plt.legend()

    plt.tight_layout()
    plt.savefig("/results/Comparative-of-different-classification-algorithms.png")
    # plt.show()

    # Feature importance

    # RandomForestRegressor
    model_RFR = RandomForestClassifier()
    importance_RFR = model_RFR.fit(X, y).feature_importances_

    feature_RFR = []
    score_RFR = []
    for i, v in enumerate(importance_RFR):
        feature_RFR.append(f[i])
        score_RFR.append(round(v, 5))

    plot_feature_importance(importance_RFR, f, "RANDOM FOREST")

    # DecisionTreeRegressor
    model_DTR = DecisionTreeClassifier()
    importance_DTR = model_DTR.fit(X, y).feature_importances_

    feature_DTR = []
    score_DTR = []
    for i, v in enumerate(importance_DTR):
        feature_DTR.append(f[i])
        score_DTR.append(round(v, 5))
    plot_feature_importance(importance_DTR, f, "DecisionTreeRegressor")

    # XGBClassifier
    model_XGB = XGBClassifier()
    importance_XBG = model_XGB.fit(X, y).feature_importances_

    feature_XGB = []
    score_XGB = []
    for i, v in enumerate(importance_XBG):
        feature_XGB.append(f[i])
        score_XGB.append(round(v, 5))
    plot_feature_importance(importance_XBG, f, " XBGoost")

    # Variance Inflation Factor (VIF)

    df_vif = df_M.drop(columns=["home_team_wins"])

    VIF = [vif(df_vif.values, i) for i in range(len(df_vif.columns))]

    feature = []
    score = []
    for i, v in enumerate(VIF):
        feature.append(f[i])
        score.append(round(v, 1))

    pd.DataFrame(score, columns=["VIF"], index=df_vif.columns).sort_values(
        by=["VIF"], ascending=False
    )

    plot_vif(VIF, df_vif.columns, "VIF")

    percentageM = []
    for m in MeanSquaredDiffWeighted:
        percentageM.append(round(m * 100, 3))
    percentageM
    df_ranking_vartype = pd.DataFrame({"Feature": pd.Series(feature)})
    df_ranking_vartype = df_ranking_vartype.assign(
        predictor_type=pd.Series(predictor_type),
        Score_DTR=pd.Series(score_DTR),
        score_RFR=pd.Series(score_RFR),
        Score_XGB=pd.Series(score_XGB),
        t_score=pd.Series(t_score),
        VIF=pd.Series(VIF),
        MSDw=pd.Series(percentageM),
    )

    df_ranking_vartype.sort_values(
        by=["MSDw", "t_score"], ascending=False
    ).to_csv(  
        r"/results/baseball.csv"
    )

    print(df_ranking_vartype)


if __name__ == "__main__":
    sys.exit(main())

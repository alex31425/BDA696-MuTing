import itertools
import os
import sys
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api
from plotly import express as px
from scipy import stats
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestClassifier  # noqa
from sklearn.ensemble import RandomForestRegressor

# uncomment for loading these datasets
# from sklearn.datasets import load_breast_cancer, load_diabetes


def main():
    # data = load_breast_cancer()
    data = load_boston()
    # data = load_diabetes()

    X = data.data
    y = data.target
    f = data["feature_names"]

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
    con_array = np.array([])
    cat_array = np.array([])
    for idx, column in enumerate(X.T):

        feature_name = data.feature_names[idx]
        predictor = statsmodels.api.add_constant(column)

        # Get the stats & plot
        if column.dtype is str or bool is True:
            v_type = "boolean"
            cat_array = np.append(cat_array, column)
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
                # fig.show()

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
                # fig.show()

        elif round((np.unique(X.T[idx]).size / X.T[idx].size), 2) <= 0.05:
            v_type = "boolean"
            cat_array = np.append(cat_array, column)
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
                # fig.show()

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
                # fig.show()
        else:
            print(data.feature_names[idx], "is continuous")
            v_type = "continuous"
            con_array = np.append(cat_array, column)  # noqa
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
                # fig.show()
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
                # fig.show()
        # create a list of each variable type
        predictor_type.append(v_type)
    pd_PT = pd.DataFrame(predictor_type, columns=["type"])
    pd_PT["name"] = f
    df_PT_c = (
        pd_PT[pd_PT["type"] == "continuous"].drop(["type"], axis=1)
    ).copy()  # noqa
    df_PT_b = (pd_PT[pd_PT["type"] == "boolean"].drop(["type"], axis=1)).copy()
    pd_PT = pd_PT.drop(["type"], axis=1)
    con_list = df_PT_c["name"].to_list()
    cat_list = df_PT_b["name"].to_list()
    f_list = pd_PT["name"].to_list()

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

    # df_con_cat_d['Response']=pd.Series(y)
    # print(df_con_cat_d)

    corr_df_con_d = df_con_d.corr(method="pearson")
    print(f"correlation metrics : Continuous / Continuous pairs")
    print(
        corr_df_con_d.sort_values(
            by=list(corr_df_con_d.columns), ascending=False
        )  # noqa
    )  # noqa

    # correlation metrics : Continuous / Categorical pair
    print(f"correlation metrics : Continuous / Categorical pairs")

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
        y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(
            n_array
        )  # noqa
        numerator = np.sum(
            np.multiply(
                n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2)
            )  # noqa
        )
        denominator = np.sum(np.power(np.subtract(values, y_total_avg), 2))
        if numerator == 0:
            eta = 0.0
        else:
            eta = np.sqrt(numerator / denominator)
        return eta

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
            print(
                df_corr_con_cat.sort_values(
                    by=list(df_cat_d.columns), ascending=False
                )  # noqa
            )
            print(f"correlation metrics : Categorical / Categorical pairs")
            trace_con_cat = go.Heatmap(
                x=df_cat_d.columns, y=con_N, z=df_corr_con_cat, type="heatmap"
            )
            fig_con_cat = go.Figure(data=[trace_con_cat])

            plot_con_cat = "plot_con_cat.html"
            fig_con_cat.write_html(
                file="plot_con_cat.html", include_plotlyjs="cdn"
            )  # noqa
            corr_plot["con_cat"] = plot_con_cat
            # fig_con_cat.show()
        else:
            pass
    except ZeroDivisionError:
        print("no cat predictor")

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
                )
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
                warnings.warn("Error calculating Cramer's V", RuntimeWarning)
            return corr_coeff

    corr_list = []
    cat_N = []
    for cat_name, cat_value in df_cat_d.iteritems():
        cat_N.append(cat_name)

        for cat_name, cat_value in df_cat_d.iteritems():
            corr_cat_cat = cat_correlation(cat_value.values, con_value.values)
            corr_list.append(corr_cat_cat)

    col = df_cat_d.shape[1]
    try:
        if col != 0:
            row = int(len(corr_list) / col)
            corr_list = np.reshape(corr_list, (row, col))
            df_corr_cat_cat = pd.DataFrame(
                corr_list, columns=df_cat_d.columns, index=cat_N
            )
            print(
                df_corr_cat_cat.sort_values(
                    by=list(df_cat_d.columns), ascending=False
                )  # noqa
            )
            trace_cat = go.Heatmap(
                x=df_cat_d.columns, y=cat_N, z=df_corr_cat_cat, type="heatmap"
            )

            fig_cat = go.Figure(data=[trace_cat])
            plot_cat = "plot_cat.html"
            fig_cat.write_html(file="plot_cat.html", include_plotlyjs="cdn")
            corr_plot["cat"] = plot_cat
            # fig_cat.show()
        else:
            pass
    except ZeroDivisionError:
        print("no cat predictor")

    # corrlation plot
    trace_con = go.Heatmap(
        x=df_con_d.columns,
        y=df_con_d.columns,
        z=corr_df_con_d.to_numpy(),
        type="heatmap",
    )
    fig_con = go.Figure(data=[trace_con])
    plot_con = "plot_con.html"
    fig_con.write_html(file="plot_con.html", include_plotlyjs="cdn")
    corr_plot["con"] = plot_con
    # fig_con.show()

    print("***Difference with mean table***")

    # Create difference with mean table

    # create a temp table df_bin to store raw data
    n_of_bin = 10
    stat_plot = {}
    BF_plot = {}
    appended_data = []
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
        print(
            f"THE unWeighted NUMBER of {feature_name} IS : {df_bin_groupby['MeanSquaredDiff'].sum() / n_of_bin}"  # noqa
        )
        print(feature_name, df_bin_groupby)

        # print(df_response)
        trace1 = go.Bar(
            x=df_bin_groupby["BinCenter"],
            y=df_bin_groupby["bin_count"],
            name="population",
            yaxis="y2",
            opacity=0.5,
        )
        y2 = go.layout.YAxis(title="Population", overlaying="y", side="right")

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
            yaxis_title=f"Binned Response Mean",
            yaxis2=y2,
        )

        combined = [trace1, trace2, trace3]
        fig = go.Figure(data=combined, layout=layout)
        # fig.show()

        # plot link

        link = f"diffmean{feature_name}.html"
        fig.write_html(file=link, include_plotlyjs="cdn")
        stat_plot[feature_name] = link

        feature_link = []
        for v in stat_plot.values():
            script_dir = os.path.dirname(__file__)
            rel_path = v
            abs_file_path = os.path.join(script_dir, rel_path)
            feature_link.append(abs_file_path)
            df_link = pd.DataFrame(feature_link)
            df_link.columns = ["feature_link"]

        def hyperlink(val):
            return '<a href="{}">{}</a>'.format(val, val)
            df_link.style.format(hyperlink)

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

        # Brute-Force" variable combination

        df_BF = df_bin.copy()
        df_BF.drop(["BinCenters", "target"], axis=1, inplace=True)
        df_BF_g = df_BF.groupby(("LowerBin_UpperBin"), as_index=False).agg(
            P_mean=pd.NamedAgg(column=feature_name, aggfunc="mean")
        )
        df_BF_g["Name"] = pd.Series(np.repeat(feature_name, n_of_bin + 1))

        df_BF_g["P_mean"]
        appended_data.append(df_BF_g["P_mean"].to_list())

    print('***Brute-Force" variable combinations***')

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

    def pair_subsets(ss):
        return itertools.chain(
            *map(lambda x: itertools.combinations(ss, x), range(1, 3))
        )

    # CAT vs CAT
    tups_cat = list(pair_subsets(df_BF_C_cat.columns))
    df_BF_C_cat_p = pd.concat(
        [df_BF_C_cat.loc[:, c].mean(axis=1) for c in tups_cat], axis=1
    )
    df_BF_C_cat_p.columns = [",".join(x) for x in tups_cat]
    df_BF_C_cat_p.drop(columns=df_BF_C_cat_p.columns & cat_list, inplace=True)
    df_BF_C_cat_p.reset_index()

    trace_BF_cat = go.Heatmap(
        x=df_BF_C_cat_p.columns,
        y=df_BF_C_cat_p.index,
        z=df_BF_C_cat_p.to_numpy(),
        type="heatmap",
    )
    fig_BF_cat = go.Figure(data=[trace_BF_cat])
    plot_BF_cat = "plot_BF_cat.html"
    fig_BF_cat.write_html(file="plot_BF_cat.html", include_plotlyjs="cdn")
    BF_plot["cat"] = plot_BF_cat
    # fig_BF_cat.show()

    df_BF_C_cat_p = df_BF_C_cat_p - PopulationMean
    df_BF_C_cat_p.loc["MSD"] = (df_BF_C_cat_p.mean() - PopulationMean) ** 2
    print(df_BF_C_cat_p)

    # CON vs CON
    tups_con = list(pair_subsets(df_BF_C_con.columns))
    df_BF_C_con_p = pd.concat(
        [df_BF_C_con.loc[:, c].mean(axis=1) for c in tups_con], axis=1
    )
    df_BF_C_con_p.columns = [",".join(x) for x in tups_con]
    df_BF_C_con_p.drop(columns=df_BF_C_con_p.columns & con_list, inplace=True)

    trace_BF_con = go.Heatmap(
        x=df_BF_C_con_p.columns,
        y=df_BF_C_con_p.index,
        z=df_BF_C_con_p.to_numpy(),
        type="heatmap",
    )
    fig_BF_con = go.Figure(data=[trace_BF_con])
    plot_BF_con = "plot_BF_con.html"
    fig_BF_con.write_html(file="plot_BF_con.html", include_plotlyjs="cdn")
    BF_plot["con"] = plot_BF_con
    # fig_BF_con.show()

    df_BF_C_con_p = df_BF_C_con_p - PopulationMean
    df_BF_C_con_p.loc["MSD"] = (df_BF_C_con_p.mean() - PopulationMean) ** 2
    print(df_BF_C_con_p)

    # CON vs CAT

    tups_con_cat = list(pair_subsets(df_BF_C_con_cat.columns))
    df_BF_C_con_cat_p = pd.concat(
        [df_BF_C_con_cat.loc[:, c].mean(axis=1) for c in tups_con_cat], axis=1
    )
    df_BF_C_con_cat_p.columns = [",".join(x) for x in tups_con_cat]
    df_BF_C_con_cat_p.drop(
        columns=df_BF_C_con_cat_p.columns & f_list, inplace=True
    )  # noqa

    trace_BF_con_cat = go.Heatmap(
        x=df_BF_C_con_cat_p.columns,
        y=df_BF_C_con_cat_p.index,
        z=df_BF_C_con_cat_p.to_numpy(),
        type="heatmap",
    )
    fig_BF_con_cat = go.Figure(data=[trace_BF_con_cat])
    plot_BF_con_cat = "plot_BF_con.html"
    fig_BF_con_cat.write_html(
        file="plot_BF_con_cat.html", include_plotlyjs="cdn"
    )  # noqa
    BF_plot["con_cat"] = plot_BF_con_cat
    # fig_BF_con_cat.show()

    df_BF_C_con_cat_p = df_BF_C_con_cat_p - PopulationMean
    df_BF_C_con_cat_p.loc["MSD"] = (
        df_BF_C_con_cat_p.mean() - PopulationMean
    ) ** 2  # noqa
    print(df_BF_C_con_cat_p)

    MSD_cat = df_BF_C_cat_p.iloc[[-1]]  # noqa
    MSD_con = df_BF_C_con_p.iloc[[-1]]  # noqa
    MSD_con_cat = df_BF_C_con_cat_p.iloc[[-1]]  # noqa
    # print(MSD_cat)
    # print(MSD_con)
    # print(MSD_con_cat)

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
    df_ranking_vartype_sort["link"] = df_link

    # path that will save the ranking excel file
    path = "D:\PycharmProjects\BDA696-MuTing\Feature_Importance_and_type.xlsx"  # noqa
    df_ranking_vartype_sort.to_excel(path, index=False)
    print(df_ranking_vartype_sort)

    # print(df_link)


if __name__ == "__main__":
    sys.exit(main())

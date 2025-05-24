# /// script
# [tool.marimo.runtime]
# auto_instantiate = false
# ///

import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium", app_title="RFGBoost")


@app.cell
def _():
    import marimo as mo

    mo.md("""
    # RFGBoost Dashboard 🌳

    Explore **Random Forest Gradient Boosting**, a hybrid ensemble method combining Random Forests with gradient boosting learning.
    """).callout(kind="neutral")
    return (mo,)


@app.cell
def _(X_test, X_train, cat_features, mo, np, num_features, y_train):
    mo.sidebar([
        mo.md("## Model Notebook"),

        mo.md("### 📊 Loan Approval Dataset"),
        mo.md(f"""
        - **Training samples**: {len(X_train):,}
        - **Test samples**: {len(X_test):,}  
        - **Features**: {len(num_features)} numeric, {len(cat_features)} categorical
        - **Classes**: {len(np.unique(y_train))} (binary)
        """),

        mo.md("---"),

        mo.md("### 🔗 Resources"),
        mo.nav_menu({
            "Links": {
                "https://github.com/deburky": f"{mo.icon('lucide:github')} Personal GitHub",
                "https://github.com/xRiskLab": f"{mo.icon('lucide:github')} xRiskLab",
            }
        }, orientation="vertical"),

        mo.md("---"),
        mo.md("Built with **marimo** 🌊🍃")
    ])
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import altair as alt
    from sklearn.metrics import (
        accuracy_score,
        roc_auc_score,
        log_loss,
        brier_score_loss,
        average_precision_score,
        balanced_accuracy_score
    )
    from rfgboost import RFGBoost
    _ = alt.data_transformers.enable("json")
    return (
        RFGBoost,
        accuracy_score,
        alt,
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        log_loss,
        np,
        pd,
        roc_auc_score,
    )


@app.cell
def _(mo):
    # Dataset controls
    sample_size = mo.ui.slider(500, 15000, value=1000, step=500, label="Sample size")
    random_seed = mo.ui.number(start=1, stop=999, value=42, label="Random seed")

    mo.vstack([
        mo.md("**Dataset Configuration**"),
        mo.hstack([sample_size, random_seed])
    ])
    return random_seed, sample_size


@app.cell
def _(np, pd, random_seed, sample_size):
    def preprocess_lti(series, special_codes):
        """Impute mean for non-special codes, keep special codes as-is"""
        # Calculate mean of non-special values
        mask = ~np.isin(series, special_codes)
        mean_value = series[mask].mean()

        return np.where(
            np.isin(series, special_codes),
            mean_value,
            series,
        )

    # Define URL
    url = "/Users/deburky/Documents/python/python-ml-projects/random-forest/BankCaseStudyData.csv"
    # url = "https://raw.githubusercontent.com/deburky/boosting-scorecards/refs/heads/main/rfgboost/BankCaseStudyData.csv"
    # url = "https://raw.githubusercontent.com/georgianastasov/credit-bureau-2021-experian/refs/heads/main/score-model/BankCaseStudyData.csv"

    # Load dataset
    dataset = pd.read_csv(url).sample(sample_size.value, random_state=random_seed.value)

    # Define features and labels
    label = "Final_Decision"
    dataset[label] = dataset[label].map({"Accept": 1, "Decline": 0})
    num_features = [
        "Application_Score", "Bureau_Score", "Loan_Amount", "Time_with_Bank",
        "Time_in_Employment", "Gross_Annual_Income", "Loan_to_income"
    ]
    cat_features = [
        "Loan_Payment_Frequency", "Residential_Status", "Cheque_Card_Flag",
        "Existing_Customer_Flag", "Home_Telephone_Number"
    ]
    special_codes_lti = [-9999997, -9999998]
    dataset["Loan_to_income"] = preprocess_lti(dataset["Loan_to_income"], special_codes_lti)

    # Create train-test split
    features = cat_features + num_features
    ix_train = dataset["split"] == "Development"
    ix_test = dataset["split"] == "Validation"
    X_train = dataset.loc[ix_train, features].copy()
    y_train = dataset.loc[ix_train, label].copy()
    X_test = dataset.loc[ix_test, features].copy()
    y_test = dataset.loc[ix_test, label].copy()
    X_train[cat_features] = X_train[cat_features].astype(str).fillna("NA")
    X_test[cat_features] = X_test[cat_features].astype(str).fillna("NA")
    X_train[num_features] = X_train[num_features].fillna(X_train[num_features].median())
    X_test[num_features] = X_test[num_features].fillna(X_train[num_features].median())
    return (
        X_test,
        X_train,
        cat_features,
        dataset,
        num_features,
        y_test,
        y_train,
    )


@app.cell
def _(dataset, mo):
    mo.vstack([
        mo.md("### Raw dataset"),
        dataset.head(300),
    ])
    return


@app.cell
def _(mo):
    # Dataset selection widget
    dataset_choice = mo.ui.dropdown(
        options=["Train", "Test"], 
        value="Train", 
        label="Select Dataset"
    )

    # Display the choice widget
    mo.md(f"### Quick EDA 📊: {dataset_choice}")
    return (dataset_choice,)


@app.cell
def _(X_test, X_train, alt, dataset_choice, mo, y_test, y_train):
    # Dynamic data selection based on choice
    if dataset_choice.value == "Train":
        X_selected = X_train
        y_selected = y_train
        dataset_name = "Train"
    else:
        X_selected = X_test
        y_selected = y_test
        dataset_name = "Test"

    # Calculate statistics dynamically
    label_dist = y_selected.value_counts(normalize=True).reset_index()
    label_dist.columns = ["Class", "Pct"]

    # Dynamic chart
    chart_labels = (
        alt.Chart(label_dist)
        .mark_bar()
        .encode(
            x=alt.X("Class:N", title="Class"),
            y=alt.Y("Pct:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Class:N")
        )
        .properties(title=f"Class Distribution ({dataset_name})", width=250)
    )

    # Dynamic stats
    stats_numeric = X_selected.describe().T
    stats_categorical = X_selected.describe(include=['O']).T

    # Display everything
    mo.vstack([
        mo.md(f"### Class distribution ({dataset_name})"),
        mo.ui.altair_chart(chart_labels),
        mo.md(f"### Numeric feature summary ({dataset_name})"),
        mo.plain(stats_numeric),
        mo.md(f"### Categorical feature summary ({dataset_name})"),
        mo.plain(stats_categorical)
    ])
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Interactive ML Model Dashboard 🧪

    We can tune different parameters of the underlying random forests to train a gradient boosting model. In this example we focus on binary classification tasks like credit scoring.

    **Algorithm Steps:**

    1. **Initialize:** f₀(x) = ȳ (regression) or log(ȳ/(1-ȳ)) (classification)

    2. **For each round m = 1, 2, ..., M:**  
       - Compute residuals: rᵢₘ = (y - rᵢₘ) / (rᵢₘ(1 - rᵢₘ))  
       - Fit Random Forest Regressor hₘ(x) on residuals with w = rᵢₘ(1 - rᵢₘ)
       - Update: fₘ(x) = fₘ₋₁(x) + η · hₘ(x)

    3. **Final prediction:** ŷ(x) = σ(fₘ(x)) for classification

    **Key Parameters:**

    - M: Number of boosting rounds  
    - η: Learning rate  
    - Random Forest depth and pruning parameters
    """
    )
    return


@app.cell
def _(mo):
    # UI for model hyperparameters
    n_estimators = mo.ui.slider(1, 20, value=5, step=1, label="Boosting rounds")
    random_state = mo.ui.slider(0, 123, value=42, step=1, label="Random seed")
    trees_in_rf = mo.ui.slider(1, 20, value=5, step=1, label="Trees per forest")
    learning_rate = mo.ui.slider(0.1, 2.0, value=0.5, step=0.1, label="Learning rate")
    max_depth = mo.ui.slider(2, 10, value=5, step=1, label="Max tree depth")
    ccp_alpha = mo.ui.slider(0.0, 0.4, value=0.0, step=0.05, label="Regularization")

    mo.vstack([
        mo.md("### RFGBoost hyperparameters 🎲"),
        mo.hstack([n_estimators, trees_in_rf, learning_rate]),
        mo.hstack([max_depth, ccp_alpha, random_state])
    ])
    return ccp_alpha, learning_rate, max_depth, n_estimators, trees_in_rf


@app.cell
def _(
    RFGBoost,
    X_train,
    cat_features,
    ccp_alpha,
    learning_rate,
    max_depth,
    n_estimators,
    trees_in_rf,
    y_train,
):
    model = RFGBoost(
        n_estimators=n_estimators.value,
        rf_params={
            "n_estimators": trees_in_rf.value,
            "max_depth": max_depth.value,
            "random_state": 42,
            "ccp_alpha": ccp_alpha.value,
        },
        learning_rate=learning_rate.value,
        cat_features=cat_features,
        task="classification",
    )
    _ = model.fit(X_train, y_train)
    return (model,)


@app.cell
def _(
    X_test,
    X_train,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    model,
    roc_auc_score,
    y_test,
    y_train,
):
    # Evaluate model
    y_train_prob = model.predict_proba(X_train)
    if y_train_prob.ndim > 1: 
        y_train_prob = y_train_prob[:, 1]
    train_acc = accuracy_score(y_train, y_train_prob > 0.5)
    train_auc = roc_auc_score(y_train, y_train_prob)
    train_logloss = log_loss(y_train, y_train_prob)
    train_brier = brier_score_loss(y_train, y_train_prob)
    train_bacc = balanced_accuracy_score(y_train, y_train_prob > 0.5)
    train_aucpr = average_precision_score(y_train, y_train_prob)

    y_test_prob = model.predict_proba(X_test)
    if y_test_prob.ndim > 1: 
        y_test_prob = y_test_prob[:, 1]
    test_acc = accuracy_score(y_test, y_test_prob > 0.5)
    test_auc = roc_auc_score(y_test, y_test_prob)
    test_logloss = log_loss(y_test, y_test_prob)
    test_brier = brier_score_loss(y_test, y_test_prob)
    test_bacc = balanced_accuracy_score(y_test, y_test_prob > 0.5)
    test_aucpr = average_precision_score(y_test, y_test_prob)
    gap = test_logloss - train_logloss
    return (
        gap,
        test_acc,
        test_auc,
        test_aucpr,
        test_bacc,
        test_brier,
        test_logloss,
        train_acc,
        train_auc,
        train_aucpr,
        train_bacc,
        train_brier,
        train_logloss,
    )


@app.cell
def _(
    alt,
    gap,
    mo,
    model,
    pd,
    test_acc,
    test_auc,
    test_aucpr,
    test_bacc,
    test_brier,
    test_logloss,
    train_acc,
    train_auc,
    train_aucpr,
    train_bacc,
    train_brier,
    train_logloss,
):
    perf_df = pd.DataFrame(
        [
            {
                "Set": "Train",
                "LogLoss": train_logloss,
                "BrierLoss": train_brier,
                "AUC": train_auc,
                "Accuracy": train_acc,
                "BalancedAccuracy": train_bacc,
                "PR-AUC": train_aucpr
            },
            {
                "Set": "Test",
                "LogLoss": test_logloss,
                "BrierLoss": test_brier,
                "AUC": test_auc,
                "Accuracy": test_acc,
                "BalancedAccuracy": test_bacc,
                "PR-AUC": test_aucpr
            },
        ]
    )

    # ━━━━━━━━━━━ Line plots ━━━━━━━━━━━
    line_chart_neg_better = (
        alt.Chart(perf_df)
        .transform_fold(["LogLoss", "BrierLoss"], as_=["Metric", "Value"])
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("Metric:N", axis=alt.Axis(title="Metrics"), sort=["LogLoss", "BrierLoss"]),
            y=alt.Y("Value:Q", axis=alt.Axis(title="Value")),
            color=alt.Color(
                "Set:N", scale=alt.Scale(range=["dodgerblue", "red"]),
                legend=alt.Legend(orient="none", legendX=10, legendY=250)
            ),
            tooltip=["Set:N", "Metric:N", "Value:Q"],
        )
        .properties(width=400, height=300, title="Proper Scoring Metrics (lower is better)")
    )

    line_chart_pos_better = (
        alt.Chart(perf_df)
        .transform_fold(["AUC", "PR-AUC", "Accuracy", "BalancedAccuracy"], as_=["Metric", "Value"])
        .mark_line(point=True, strokeWidth=0)
        .encode(
            x=alt.X("Metric:N", axis=alt.Axis(title="Metrics"), sort=["AUC", "PR-AUC", "Accuracy", "BalancedAccuracy"]),
            y=alt.Y("Value:Q", axis=alt.Axis(title="Value")),
            color=alt.Color(
                "Set:N", scale=alt.Scale(range=["dodgerblue", "red"]),
                legend=alt.Legend(orient="none", legendX=10, legendY=250)
            ),
            tooltip=["Set:N", "Metric:N", "Value:Q"],
        )
        .properties(width=400, height=300, title="Metrics (higher is better)")
    )

    # ━━━━━━━━━━━ Feature importance ━━━━━━━━━━━
    feat_imp = model.get_feature_importance()
    if not isinstance(feat_imp, pd.DataFrame):
        feat_imp = pd.DataFrame(list(feat_imp.items()), columns=["Feature", "Importance"])
    feat_imp = feat_imp.sort_values("Importance", ascending=False)
    chart_feat = alt.Chart(feat_imp).mark_bar().encode(
        x=alt.X("Importance:Q"),
        color=alt.Color("Importance:Q", scale=alt.Scale(scheme="tealblues")),
        y=alt.Y("Feature:N", sort="-x"),
        tooltip=["Feature", "Importance"]
    ).properties(width=400, height=300, title="Feature Importances")

    # ━━━━━━━━━━━ Bar charts ━━━━━━━━━━━
    logloss_chart = (
        alt.Chart(perf_df)
            .mark_bar(size=50)
            .encode(x="Set:N", y="LogLoss:Q")
            .properties(width=120, title="Log Loss")
    )
    auc_chart = (
        alt.Chart(perf_df).mark_bar(size=50).encode(x="Set:N", y=alt.Y("AUC:Q", scale=alt.Scale(domain=[0, 1]))).properties(width=120, title="AUC")
    )

    # ━━━━━━━━━━━ Overfitting gap ━━━━━━━━━━━
    gap_df = pd.DataFrame({"Type": ["Overfit Gap"], "Gap": [gap]})
    gap_chart = (
        alt.Chart(gap_df)
        .mark_bar(
            width=30,
            color=alt.expr("abs(datum.Gap) > 0.1 ? 'red' : abs(datum.Gap) > 0.05 ? 'orange' : 'palegreen'"),
        )
        .encode(
            x=alt.X("Type:N", axis=None),
            y=alt.Y("Gap:Q", axis=alt.Axis(format=".3f")),
            tooltip=["Gap:Q"],
        )
        .properties(width=80, height=150, title="Overfit Level")
    )

    # ━━━━━━━━━━━ Combine plots ━━━━━━━━━━━
    left_panel = alt.vconcat(
        line_chart_pos_better,  chart_feat
    )
    right_panel = alt.vconcat(
        line_chart_neg_better,
        alt.hconcat(gap_chart, logloss_chart, auc_chart)
    )
    layout_option3 = left_panel | right_panel
    mo.ui.altair_chart(layout_option3)
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 2D feature map model decision surface 🗾

    Below we plot a 2D scatterplot with model predicted probabilities to show confidence of the predictions.
    """
    )
    return


@app.cell
def _(mo, num_features):
    # Feature selection for next plots
    feature_x = mo.ui.dropdown(options=num_features, value=num_features[0], label="X-axis feature")
    feature_y = mo.ui.dropdown(options=[f for f in num_features if f != num_features[0]], value=num_features[1], label="Y-axis feature")
    mo.hstack([feature_x, feature_y])
    return feature_x, feature_y


@app.cell
def _(
    RFGBoost,
    X_train,
    cat_features,
    ccp_alpha,
    feature_x,
    feature_y,
    learning_rate,
    max_depth,
    n_estimators,
    y_train,
):
    # Train 2D model reactively when features change
    selected_features = [feature_x.value, feature_y.value]

    model_2d = RFGBoost(
        n_estimators=n_estimators.value,
        rf_params={"max_depth": max_depth.value, "random_state": 42, "ccp_alpha": ccp_alpha.value},
        learning_rate=learning_rate.value,
        cat_features=[cat for cat in cat_features if cat in selected_features],  # Only relevant cat features
        task="classification"
    )

    # Fit the model on just the selected 2D features
    _ = model_2d.fit(X_train[selected_features], y_train)

    print(f"2D model trained on features: {', '.join(selected_features)}")
    return (model_2d,)


@app.cell
def _(
    X_test,
    X_train,
    alt,
    balanced_accuracy_score,
    feature_x,
    feature_y,
    mo,
    model_2d,
    np,
    pd,
    resolution,
    roc_auc_score,
    show_conf,
    y_test,
    y_train,
):
    import scipy.stats as scipy_stats

    # Define consistent colors for classes
    class_colors = ["#d73027", "#4575b4"]  # Red for class 0, Blue for class 1

    plot_df = X_test.copy()
    plot_df["y_true"] = y_test

    # ━━━━━━━━━━━ Correlation metrics ━━━━━━━━━━━
    xi, _ = scipy_stats.chatterjeexi(plot_df[feature_x.value], plot_df[feature_y.value])
    rho, _ = scipy_stats.spearmanr(plot_df[feature_x.value], plot_df[feature_y.value])

    # Use the 2D model with 2D test data
    X_test_2d = X_test[[feature_x.value, feature_y.value]]
    y_raw_prob = model_2d.predict_proba(X_test_2d)[:, 1]

    auc_score = roc_auc_score(
        y_test, y_raw_prob
    )
    bacc_score = balanced_accuracy_score(
        y_test, (y_raw_prob > 0.5).astype(int)
    )

    y_prob = np.column_stack(
        (
            1 - model_2d.predict_ci(X_test_2d)[:, 0],  # Upper bound for P(class=0)
            model_2d.predict_ci(X_test_2d)[:, 1],      # Upper bound for P(class=1)
        )
    )
    # y_prob_scatter = y_prob
    y_prob_scatter = y_prob[np.arange(len(y_test)), y_test]
    plot_df["y_prob"] = y_prob_scatter

    # ━━━━━━━━━━━ Scatter 2D ━━━━━━━━━━━
    scatter_chart = (
        alt.Chart(plot_df)
        .mark_circle(size=60, opacity=0.8, stroke="white", strokeWidth=1)
        .encode(
            x=alt.X(feature_x.value + ":Q", title=feature_x.value).scale(zero=False),
            y=alt.Y(feature_y.value + ":Q", title=feature_y.value).scale(zero=False),
            color=alt.Color(
                "y_prob:Q",
                scale=alt.Scale(domain=["0", "1"], range=class_colors),
                legend=alt.Legend(title="Model Confidence"),
            ),
            shape=alt.Shape("y_true:N", scale=alt.Scale(range=["circle", "square"]), legend=alt.Legend(title="True Label")),
            tooltip=[feature_x.value, feature_y.value, "y_true", "y_prob"],
        )
        .properties(
            width=350,
            height=300,
            title=alt.Title("Scatterplot", subtitle=[f"Chatterjee xi = {xi:.3f}", f"Spearman ρ = {rho:.3f}"]),
        )
    )

    # ━━━━━━━━━━━ Decision surface ━━━━━━━━━━━
    x_feature = feature_x.value
    y_feature = feature_y.value
    x_min, x_max = X_train[x_feature].min() - 1, X_train[x_feature].max() + 1
    y_min, y_max = X_train[y_feature].min() - 1, X_train[y_feature].max() + 1
    res = resolution.value
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, res), np.linspace(y_min, y_max, res))
    grid = np.c_[xx.ravel(), yy.ravel()]

    grid_2d = pd.DataFrame(grid, columns=[x_feature, y_feature])
    Z = model_2d.predict_proba(grid_2d)
    if Z.ndim > 1:
        Z = Z[:, 1]

    contour_df = pd.DataFrame({"x": xx.ravel(), "y": yy.ravel(), "prob": Z})

    # ━━━━━━━━━━━ Confidence levels ━━━━━━━━━━━
    if show_conf.value:
        decision_surface = (
            alt.Chart(contour_df)
            .mark_square(size=(350 * 500) / (res * res), opacity=0.7)
            .encode(
                x=alt.X("x:Q", title=x_feature).scale(zero=False),
                y=alt.Y("y:Q", title=y_feature).scale(zero=False),
                color=alt.Color(
                    "prob:Q",
                    scale=alt.Scale(domain=[0, 1], range=class_colors),
                    legend=alt.Legend(title="P(y=1)"),
                ),
            )
        )
    else:
        decision_surface = (
            alt.Chart(contour_df[abs(contour_df.prob - 0.5) < 0.05])
            .mark_circle(size=8, color="black")
            .encode(
                x=alt.X("x:Q", title=x_feature).scale(zero=False),
                y=alt.Y("y:Q", title=y_feature).scale(zero=False)
            )
        )

    # Training points
    tdf = pd.DataFrame({x_feature: X_train[x_feature], y_feature: X_train[y_feature], "Class": y_train.astype(str)})

    training_points = (
        alt.Chart(tdf)
        .mark_circle(size=50, opacity=0.9, stroke="white", strokeWidth=1)
        .encode(
            x=f"{x_feature}:Q",
            y=f"{y_feature}:Q",
            color=alt.Color(
                "Class:N", scale=alt.Scale(domain=["0", "1"], range=class_colors), legend=alt.Legend(title="True Class")
            ),
        )
    )

    decision_chart = (decision_surface + training_points).properties(
        width=350, height=300, title=alt.Title("Decision surface", subtitle=[f"AUC = {auc_score:.3f}", f"Balanced Accuracy = {bacc_score:.3f}"])
    )


    # ━━━━━━━━━━━ Combined plot ━━━━━━━━━━━
    combined_chart = (scatter_chart | decision_chart).resolve_scale(color="independent")

    mo.ui.altair_chart(combined_chart)
    return


@app.cell
def _(mo):
    show_conf = mo.ui.checkbox(True, label="Show confidence")
    resolution = mo.ui.slider(20, 100, value=50, step=10, label="Resolution")
    mo.hstack([show_conf, resolution])
    return resolution, show_conf


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md("""
    # Categorical Encoders

    In this notebook we explore different ways to encode categorical data to use in classification tasks.

    Author: https://www.github.com/deburky

    Built with **marimo** 🌊🍃
    """)# .callout(kind="neutral")
    return (mo,)


@app.cell
def _():
    import uuid
    from itertools import zip_longest

    import altair as alt
    import numpy as np
    import pandas as pd
    from scipy.special import expit as sigmoid
    from scipy.special import logit
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, TargetEncoder
    from category_encoders import CatBoostEncoder
    return (
        CatBoostEncoder,
        LogisticRegression,
        OneHotEncoder,
        TargetEncoder,
        alt,
        brier_score_loss,
        log_loss,
        logit,
        np,
        pd,
        roc_auc_score,
        sigmoid,
        train_test_split,
        uuid,
        zip_longest,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Creating online lending data

    This is a simulation of real-world portfolio of online lending.

    We use three categorical variables: type of account, subscription type, and bureau score to predict default.
    """
    )
    return


@app.cell
def _(mo):
    # Dataset controls
    sample_size = mo.ui.slider(500, 50_000, value=25_000, step=500, label="Sample size")
    random_seed = mo.ui.number(start=1, stop=999, value=42, label="Random seed")

    mo.vstack([
        mo.md("**Dataset Configuration**"),
        mo.hstack([sample_size, random_seed])
    ])
    return random_seed, sample_size


@app.cell
def _(np, pd, random_seed, sample_size, uuid):
    np.random.seed(random_seed.value)


    def conditional_prob_cat_given_event(feature_dist, event_rate, overall_event_rate):
        """
        Calculate P(cat | is_default=1) and P(cat | is_default=0)
        feature_dist: dict, P(cat)
        event_rate: dict, P(is_default=1 | cat)
        overall_event_rate: float, P(is_default=1)
        Returns: dicts for class 1 and class 0
        """
        # P(cat | is_default=1) ∝ P(is_default=1 | cat) * P(cat)
        prob_given_1 = {}
        for cat in feature_dist:
            prob = event_rate[cat] * feature_dist[cat]
            prob_given_1[cat] = prob
        # Normalize
        total_1 = sum(prob_given_1.values())
        for cat in prob_given_1:
            prob_given_1[cat] /= total_1

        # P(cat | is_default=0) ∝ (1 - P(is_default=1 | cat)) * P(cat)
        prob_given_0 = {}
        for cat in feature_dist:
            prob = (1 - event_rate[cat]) * feature_dist[cat]
            prob_given_0[cat] = prob
        total_0 = sum(prob_given_0.values())
        for cat in prob_given_0:
            prob_given_0[cat] /= total_0

        return prob_given_1, prob_given_0

    def generate_stratified_synthetic_data(n_samples: int = 10_000):
        # Feature distributions (from your input)
        is_business_dist = {False: 0.894687, True: 0.105313}
        subscription_dist = {'BASIC': 0.895823, 'ENHANCED': 0.097411, 'CUSTOM': 0.006766}
        bureau_rating_raw = {
            'A': 0.084943, 'B': 0.145165, 'C': 0.160290, 'D': 0.196071,
            'E': (0.189412 + 0.142590 + 0.081423),
        }
        total = sum(bureau_rating_raw.values())
        bureau_rating_dist = {k: v / total for k, v in bureau_rating_raw.items()}
        event_rates = {
            'is_business': {False: 0.0384, True: 0.0621},
            'subscription': {'BASIC': 0.0363, 'ENHANCED': 0.0797, 'CUSTOM': 0.0957},
            'bureau_rating': {
                'A': 0.0134, 'B': 0.0224, 'C': 0.0332, 'D': 0.0417,
                'E': (0.0480 + 0.0601 + 0.0833),
            }
        }
        # 1. Calculate expected overall default rate
        overall_default_rate = (
            sum([is_business_dist[k] * event_rates['is_business'][k] for k in is_business_dist]) +
            sum([subscription_dist[k] * event_rates['subscription'][k] for k in subscription_dist]) +
            sum([bureau_rating_dist[k] * event_rates['bureau_rating'][k] for k in bureau_rating_dist])
        ) / 3

        n_pos = int(round(n_samples * overall_default_rate))
        n_neg = n_samples - n_pos

        # 2. Get conditional distributions P(cat | y)
        p_is_business_1, p_is_business_0 = conditional_prob_cat_given_event(is_business_dist, event_rates['is_business'], overall_default_rate)
        p_subscription_1, p_subscription_0 = conditional_prob_cat_given_event(subscription_dist, event_rates['subscription'], overall_default_rate)
        p_bureau_1, p_bureau_0 = conditional_prob_cat_given_event(bureau_rating_dist, event_rates['bureau_rating'], overall_default_rate)

        # 3. Simulate positives and negatives
        def draw_feature(dist, size):
            # Draw with replacement according to distribution
            return np.random.choice(list(dist.keys()), size=size, p=list(dist.values()))

        data_pos = {
            'user_id': [str(uuid.uuid4()) for _ in range(n_pos)],
            'is_default': np.ones(n_pos, dtype=int),
            'is_business': draw_feature(p_is_business_1, n_pos),
            'subscription': draw_feature(p_subscription_1, n_pos),
            'bureau_rating': draw_feature(p_bureau_1, n_pos)
        }
        data_neg = {
            'user_id': [str(uuid.uuid4()) for _ in range(n_neg)],
            'is_default': np.zeros(n_neg, dtype=int),
            'is_business': draw_feature(p_is_business_0, n_neg),
            'subscription': draw_feature(p_subscription_0, n_neg),
            'bureau_rating': draw_feature(p_bureau_0, n_neg)
        }
        df = pd.concat([pd.DataFrame(data_pos), pd.DataFrame(data_neg)], ignore_index=True)
        # Shuffle rows
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        return df

    df = generate_stratified_synthetic_data(sample_size.value)
    print(df.groupby('is_business')['is_default'].mean())
    print(df.groupby('subscription')['is_default'].mean())
    print(df.groupby('bureau_rating')['is_default'].mean())
    return (df,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Categorical encoders

    In this example, we use test different schemes of encoding categorical data.
    """
    )
    return


@app.cell
def _(TargetEncoder, df, random_seed, train_test_split):
    te = TargetEncoder(random_state=random_seed.value, cv=2, smooth=1e1).set_output(transform="pandas")

    features = ["is_business", "subscription", "bureau_rating"]
    label = "is_default"

    df_reindexed = df.sample(frac=1, random_state=random_seed.value).reset_index(drop=True)
    X, y = df_reindexed[features].copy(), df_reindexed[label].copy()

    ix_train, ix_test = train_test_split(X.index, stratify=y, random_state=random_seed.value)

    X_train, X_test = X.loc[ix_train], X.loc[ix_test]
    y_train, y_test = y.loc[ix_train], y.loc[ix_test]

    X_train_te = te.fit_transform(X_train, y_train)
    X_test_te = te.transform(X_test)
    return X_test, X_train, X_train_te, te, y_test, y_train


@app.cell
def _(X_train_te, pd, te):
    pd.DataFrame(X_train_te, columns=te.get_feature_names_out())
    return


@app.cell
def _(mo, te):
    feature_names = te.get_feature_names_out()

    feature_to_show = mo.ui.dropdown(
        options=feature_names,
        value=feature_names[0],
        label="Feature to display"
    )
    feature_to_show
    return feature_names, feature_to_show


@app.cell
def _(alt, feature_names, feature_to_show, pd, te, zip_longest):
    # ━━━━━━━━━━━ Target encodings ━━━━━━━━━━━
    categories, encodings = te.categories_, te.encodings_

    rows = []
    for fname, cats, encs in zip(feature_names, categories, encodings):
        for cat, enc in zip_longest(cats, encs, fillvalue=pd.NA):
            rows.append({'feature': fname, 'category': cat, 'encoding': enc})

    df_summary = pd.DataFrame(rows)
    df_summary['average'] = te.target_mean_

    # ━━━━━━━━━━━ Feature selection ━━━━━━━━━━━
    df_feat = df_summary[df_summary['feature'] == feature_to_show.value].copy()
    baseline = df_feat['average'].iloc[0]

    # The x-axis labels
    df_feat['label'] = df_feat['category'].astype(str)

    # Bar or line for cumulative probability per category
    line1 = alt.Chart(df_feat).mark_bar(color='teal').encode(
        x=alt.X('label:N', title='Category'),
        y=alt.Y('encoding:Q', title='Default Rate / Probability of Default', axis=alt.Axis(format='%'))
    )

    # Horizontal dashed average line
    avg_line = alt.Chart(pd.DataFrame({'rate': [baseline], 'desc': ['Average']})).mark_rule(
        color='orange', strokeDash=[10,10], size=3
    ).encode(
        y='rate:Q',
        tooltip=[alt.Tooltip('rate:Q', format='.2%'), alt.Tooltip('desc:N')]
    ).properties(title=feature_to_show.value)
    avg_text = alt.Chart(pd.DataFrame({'rate': [baseline]})).mark_text(
        align='left', baseline='bottom', dx=5, dy=-5, color='black'
    ).encode(
        x=alt.value(0),
        y='rate:Q',
        text=alt.value('Average')
    )

    chart = (line1 + avg_line + avg_text).properties(width=500, height=350)
    chart
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## From probabilities to WOE

    Now each categorical group has a value corresponding to the the average event rate in that category. To make this data transformation more interpretable, we can convert these values to Weight of Evidence (WOE) scores by using a trick. This formula is due to A. Turing / J. Good.

    If we center the encoded probabilities the average event rate of the sample, we can create a form of standardized scores that measure how far each group lies from the average.

    Since most models will produce an average score which is close to the true average (unless class weights or other algorithm-level adjustment is used), we can use either observed rate or expected (average of predictions).
    """
    )
    return


@app.cell
def _(TargetEncoder, np, pd):
    class WOEScorer:
        """Fit any encoder per feature and convert encodings to WOE directly."""

        def __init__(
            self, random_state=42, encoder_class=TargetEncoder, encoder_kwargs=None
        ):
            self.random_state = random_state
            self.encoder_class = encoder_class
            self.encoder_kwargs = encoder_kwargs or {}
            self.encoders = {}
            self.mapping_ = {}
            self.y_prior_ = None
            self.is_fitted_ = False

        def fit(self, X: pd.DataFrame, y: pd.Series):
            self.y_prior_ = y.mean()
            odds_prior = self.y_prior_ / (1 - self.y_prior_)

            for col in X.columns:
                enc_kwargs = self.encoder_kwargs.copy()
                enc_kwargs["random_state"] = self.random_state
                # Remove unused kwargs for CatBoostEncoder
                if self.encoder_class.__name__ == "CatBoostEncoder":
                    enc_kwargs.pop("cv", None)
                    enc_kwargs.pop("smooth", None)
                    enc = self.encoder_class(cols=[col], return_df=True, **enc_kwargs)
                    enc.fit(X[[col]], y)
                    self.encoders[col] = enc
                    # CatBoostEncoder does not expose mapping directly
                    cats = pd.Series(X[col].unique(), name=col)
                    event_rate = enc.transform(cats.to_frame()).values.flatten()
                else:
                    enc = self.encoder_class(**enc_kwargs)
                    enc.fit(X[[col]], y)
                    self.encoders[col] = enc
                    cats = enc.categories_[0]
                    event_rate = enc.encodings_[0]

                # Defensive clipping for WoE
                event_rate = np.clip(event_rate, 1e-15, 1 - 1e-15)
                odds_cat = event_rate / (1 - event_rate)
                woe = np.log(odds_cat / odds_prior)

                mapping = pd.DataFrame(
                    {"category": cats, "event_rate": event_rate, "woe": woe}
                ).set_index("category")
                self.mapping_[col] = mapping
            self.is_fitted_ = True
            return self

        def transform(self, X: pd.DataFrame) -> pd.DataFrame:
            odds_prior = self.y_prior_ / (1 - self.y_prior_)
            woe_df = pd.DataFrame(index=X.index)
            for col in X.columns:
                enc = self.encoders[col]
                event_rate = enc.transform(X[[col]])
                # CategoryEncoders always returns a DataFrame; sklearn can return np.ndarray
                if isinstance(event_rate, pd.DataFrame):
                    event_rate = event_rate.values.flatten()
                event_rate = np.clip(event_rate, 1e-15, 1 - 1e-15)
                odds_cat = event_rate / (1 - event_rate)
                woe = np.log(odds_cat / odds_prior)
                woe_df[f"{col}_woe"] = woe
            return woe_df

        def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
            return self.fit(X, y).transform(X)

        def get_mapping(self, col: str) -> pd.DataFrame:
            return self.mapping_[col].reset_index()
    return (WOEScorer,)


@app.cell
def _(WOEScorer, X_train, feature_to_show, mo, random_seed, y_train):
    scorer = WOEScorer(random_state=random_seed.value)
    X_woe_train = scorer.fit_transform(X_train, y_train)
    mo.plain(scorer.mapping_[feature_to_show.value])
    return (scorer,)


@app.cell
def _(alt, feature_to_show, pd, scorer):
    feature_name = feature_to_show.value

    df_feat_woe = scorer.mapping_[feature_name].reset_index().copy()
    df_feat_woe['label'] = df_feat_woe['category'].astype(str)

    woe_bar = alt.Chart(df_feat_woe).mark_bar(color='teal').encode(
        x=alt.X('label:N', title='Category'),
        y=alt.Y('woe:Q', title='Weight of Evidence (WOE)')
    )
    woe_zero_line = alt.Chart(pd.DataFrame({'woe': [0]})).mark_rule(
        color='orange', strokeDash=[10,10], size=3
    ).encode(
        y='woe:Q'
    )
    woe_zero_text = alt.Chart(pd.DataFrame({'woe': [0]})).mark_text(
        align='left', baseline='bottom', dx=5, dy=-5, color='orange', fontWeight='bold'
    ).encode(
        x=alt.value(0),
        y='woe:Q',
        text=alt.value('WOE=0 (avg risk)')
    )

    chart_woe = (woe_bar + woe_zero_line + woe_zero_text).properties(
        width=500, height=350, title=f"WOE for {feature_name}"
    )
    chart_woe
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Measure performance

    One way to turn the transformed input into a classifier is to sum up the WOE scores for each feature and add an intercept to it. It sounds a lot like what gradient boosted trees do, but it can also be seen as logistic regression with coefficients being equal to 1.
    """
    )
    return


@app.cell
def _(mo):
    # UI for model hyperparameters
    smooth = mo.ui.slider(0, 1.0, value=0.5, step=0.05, label="Smoothing")
    a = mo.ui.slider(0, 10, value=0, step=1, label="A")
    l1_ratio = mo.ui.slider(0.0, 1.0, value=1.0, step=0.1, label="L1 ratio")
    max_iter = mo.ui.slider(0, 200, value=5, step=1, label="Maximum iterations")

    mo.vstack([
        mo.md("### TargetEncoder Parameters"),
        mo.vstack([smooth]),
        mo.md("### CatBoostEncoder Parameters"),
        mo.vstack([a]),
        mo.md("### Logistic Regression Parameters"),
        mo.vstack([l1_ratio, max_iter]),
    ])
    return a, l1_ratio, max_iter, smooth


@app.cell
def _(
    CatBoostEncoder,
    LogisticRegression,
    OneHotEncoder,
    TargetEncoder,
    WOEScorer,
    X_test,
    X_train,
    a,
    brier_score_loss,
    l1_ratio,
    log_loss,
    logit,
    max_iter,
    mo,
    np,
    random_seed,
    roc_auc_score,
    sigmoid,
    smooth,
    y_test,
    y_train,
):
    # ━━━━━━━━━━━ Encoders and LR Parameters ━━━━━━━━━━━
    lr_params = dict(
        solver="saga",
        penalty="elasticnet",
        l1_ratio=l1_ratio.value,
        max_iter=max_iter.value,
        random_state=random_seed.value,
    )

    woe_te_params = dict(
        smooth=smooth.value,
        random_state=random_seed.value,
    )
    woe_cb_params = dict(
        a=a.value,
    )


    # ━━━━━━━━━━━ WOE Scorer - TargetEncoder ━━━━━━━━━━━
    woe_scorer_te = WOEScorer(
        random_state=random_seed.value,
        encoder_class=TargetEncoder,
        encoder_kwargs=woe_te_params
    )
    X_woe_te_train = woe_scorer_te.fit_transform(X_train, y_train)
    X_woe_te_test = woe_scorer_te.transform(X_test)

    # Scorecard-style: Sum WOE and add logit(prior)
    woe_to_p = sigmoid(X_woe_te_test.sum(axis=1) + logit(woe_scorer_te.y_prior_))
    gini_woe_te = roc_auc_score(y_test, woe_to_p) * 2 - 1
    log_loss_woe_te = log_loss(y_test, woe_to_p)
    brier_woe_te = brier_score_loss(y_test, woe_to_p)

    # ━━━━━━━━━━━ WOE Scorer - CatBoostEncoder ━━━━━━━━━━━
    woe_scorer_cb = WOEScorer(
        random_state=random_seed.value,
        encoder_class=CatBoostEncoder,
        encoder_kwargs=woe_cb_params,
    )
    X_woe_cb_train = woe_scorer_cb.fit_transform(X_train, y_train)
    X_woe_cb_test = woe_scorer_cb.transform(X_test)

    woe_cb_to_p = sigmoid(X_woe_cb_test.sum(axis=1) + logit(woe_scorer_cb.y_prior_))
    gini_woe_cb = roc_auc_score(y_test, woe_cb_to_p) * 2 - 1
    log_loss_woe_cb = log_loss(y_test, woe_cb_to_p)
    brier_woe_cb = brier_score_loss(y_test, woe_cb_to_p)

    # ━━━━━━━━━━━ WOE TargetEncoder + Logistic Regression ━━━━━━━━━━━
    lr_model_te = LogisticRegression(**lr_params)
    lr_model_te.fit(X_woe_te_train, y_train)
    lr_to_p_te = lr_model_te.predict_proba(X_woe_te_test)[:, 1]
    gini_lr_te = roc_auc_score(y_test, lr_to_p_te) * 2 - 1
    log_loss_lr_te = log_loss(y_test, lr_to_p_te)
    brier_lr_te = brier_score_loss(y_test, lr_to_p_te)

    # ━━━━━━━━━━━ WOE CatBoostEncoder + Logistic Regression ━━━━━━━━━━━

    lr_model_cb = LogisticRegression(
        **lr_params
    )
    lr_model_cb.fit(X_woe_cb_train, y_train)
    lr_to_p_cb = lr_model_cb.predict_proba(X_woe_cb_test)[:, 1]
    gini_lr_cb = roc_auc_score(y_test, lr_to_p_cb) * 2 - 1
    log_loss_lr_cb = log_loss(y_test, lr_to_p_cb)
    brier_lr_cb = brier_score_loss(y_test, lr_to_p_cb)

    # ━━━━━━━━━━━ OneHotEncoder + Logistic Regression ━━━━━━━━━━━
    one_hot_encoder = OneHotEncoder(drop="first", sparse_output=False).set_output(
        transform="pandas"
    )
    X_ohe_train = one_hot_encoder.fit_transform(X_train)
    X_ohe_test = one_hot_encoder.transform(X_test)
    lr_ohe_model = LogisticRegression(**lr_params)
    lr_ohe_model.fit(X_ohe_train, y_train)
    lr_ohe_to_p = lr_ohe_model.predict_proba(X_ohe_test)[:, 1]
    gini_lr_ohe = roc_auc_score(y_test, lr_ohe_to_p) * 2 - 1
    log_loss_lr_ohe = log_loss(y_test, lr_ohe_to_p)
    brier_lr_ohe = brier_score_loss(y_test, lr_ohe_to_p)

    def nsqrt(p):
        """Standard error for a Bernoulli variable with mean p."""
        return np.sqrt(p * (1 - p))

    mean_woe_te = woe_to_p.mean() / 100
    stderr_woe_te = nsqrt(mean_woe_te)
    mean_woe_cb = woe_cb_to_p.mean() / 100
    stderr_woe_cb = nsqrt(mean_woe_cb)

    # ━━━━━━━━━━━ Display Results ━━━━━━━━━━━
    mo.md(
        f"""
    | Model                                      | Gini    | Log Loss  | Brier  |
    |---------------------------------------------|---------|-----------|--------|
    | WOE TargetEncoder (sum, scorecard)         | {gini_woe_te:.4f} | {log_loss_woe_te:.4f} | {brier_woe_te:.4f} |
    | WOE CatBoostEncoder (sum, scorecard)       | {gini_woe_cb:.4f} | {log_loss_woe_cb:.4f} | {brier_woe_cb:.4f} |
    | WOE TargetEncoder + Logistic Regression    | {gini_lr_te:.4f}  | {log_loss_lr_te:.4f}  | {brier_lr_te:.4f}  |
    | WOE CatBoostEncoder + Logistic Regression  | {gini_lr_cb:.4f}  | {log_loss_lr_cb:.4f}  | {brier_lr_cb:.4f}  |
    | OHE + Logistic Regression                  | {gini_lr_ohe:.4f} | {log_loss_lr_ohe:.4f} | {brier_lr_ohe:.4f} |
    |---------------------------------------------|---------|-----------|--------|
    | Log Avg P | Log STD (WOE TE) {mean_woe_te:.10f}  {stderr_woe_te:.10f} |
    | Log Avg P | Log STD (WOE CB) {mean_woe_cb:.10f} {stderr_woe_cb:.10f} |
    """
    )
    return X_woe_cb_test, X_woe_te_test, woe_scorer_cb, woe_scorer_te


@app.cell
def _(X_test, mo):
    # Cell 1: Create interactive controls
    feature_options = list(X_test.columns)
    encoder_options = ['TargetEncoder', 'CatBoostEncoder']

    feature1_slider = mo.ui.dropdown(
        options=feature_options, 
        value=feature_options[0], 
        label="Feature 1 (X-axis)"
    )

    feature2_slider = mo.ui.dropdown(
        options=feature_options, 
        value=feature_options[1] if len(feature_options) > 1 else feature_options[0], 
        label="Feature 2 (Y-axis)"
    )

    encoder_slider = mo.ui.dropdown(
        options=encoder_options, 
        value='TargetEncoder', 
        label="Encoder"
    )

    mo.hstack([feature1_slider, feature2_slider, encoder_slider])
    return encoder_slider, feature1_slider, feature2_slider


@app.cell
def _(
    X_test,
    X_woe_cb_test,
    X_woe_te_test,
    alt,
    encoder_slider,
    feature1_slider,
    feature2_slider,
    logit,
    mo,
    np,
    pd,
    sigmoid,
    woe_scorer_cb,
    woe_scorer_te,
):
    # Get selected features and encoder
    feat1 = feature1_slider.value
    feat2 = feature2_slider.value
    encoder = encoder_slider.value

    # Choose the right WOE data and scorer based on encoder selection
    if encoder == "TargetEncoder":
        woe_data = X_woe_te_test.copy()
        scorer_ = woe_scorer_te
    else:
        woe_data = X_woe_cb_test.copy()
        scorer_ = woe_scorer_cb

    # Get WOE values for selected features
    feat1_woe = f"{feat1}_woe"
    feat2_woe = f"{feat2}_woe"

    # Calculate predicted probability for each point
    woe_sum = woe_data[feat1_woe] + woe_data[feat2_woe]
    predicted_prob = sigmoid(woe_sum + logit(scorer_.y_prior_))

    # Create plot data
    plot_data = pd.DataFrame({
        "feat1_woe": woe_data[feat1_woe],
        "feat2_woe": woe_data[feat2_woe],
        "feat1_cat": X_test[feat1].values,
        "feat2_cat": X_test[feat2].values,
        "predicted_prob": predicted_prob,
    })

    # Create a combined label for both categories
    plot_data["combined_label"] = plot_data["feat1_cat"].astype(str) + ", " + plot_data["feat2_cat"].astype(str)

    # For better positioning, get unique combinations only
    unique_plot_data = plot_data[
        ["feat1_woe", "feat2_woe", "feat1_cat", "feat2_cat", "predicted_prob", "combined_label"]
    ].drop_duplicates()

    # Create decision surface with contour
    x_range = np.linspace(plot_data["feat1_woe"].min(), plot_data["feat1_woe"].max(), 100)
    y_range = np.linspace(plot_data["feat2_woe"].min(), plot_data["feat2_woe"].max(), 100)
    xx, yy = np.meshgrid(x_range, y_range)

    # Calculate probability surface
    woe_surface = xx + yy
    prob_surface = woe_surface + logit(scorer_.y_prior_)

    # Create contour data
    contour_data = []
    for i in range(len(x_range)):
        for j in range(len(y_range)):
            contour_data.append({"feat1_woe": xx[j, i], "feat2_woe": yy[j, i], "prob": prob_surface[j, i]})
    contour_df = pd.DataFrame(contour_data)

    # Contour plot (decision surface)
    contour = (
        alt.Chart(contour_df)
        .mark_rect(opacity=0.3)
        .encode(
            x=alt.X("feat1_woe:Q", title=f"{feat1} WOE"),
            y=alt.Y("feat2_woe:Q", title=f"{feat2} WOE"),
            color=alt.Color("prob:Q", scale=alt.Scale(scheme="purples", domain=[-2, 2]), title="Predicted Log Odds"),
        )
    )

    # Scatter plot with actual data points
    scatter = (
        alt.Chart(unique_plot_data)
        .mark_circle(size=100, opacity=0.9, stroke=None, strokeWidth=0)
        .encode(
            x=alt.X("feat1_woe:Q", title=f"{feat1} WOE"),
            y=alt.Y("feat2_woe:Q", title=f"{feat2} WOE"),
            color=alt.Color("predicted_prob:Q", scale=alt.Scale(scheme="purples", domain=[-2, 2]), title="WOE"),
            tooltip=[
                alt.Tooltip("feat1_cat:N", title=f"{feat1} Category"),
                alt.Tooltip("feat2_cat:N", title=f"{feat2} Category"),
                alt.Tooltip("feat1_woe:Q", title=f"{feat1} WOE", format=".3f"),
                alt.Tooltip("feat2_woe:Q", title=f"{feat2} WOE", format=".3f"),
                alt.Tooltip("predicted_prob:Q", title="Predicted Prob", format=".3f"),
                alt.Tooltip("target:N", title="Actual Target"),
            ],
        )
    )

    # Text labels for categories
    text_labels = (
        alt.Chart(unique_plot_data)
        .mark_text(
            align="center",
            baseline="middle",
            dx=0,
            dy=-20,
            fontSize=10,
            fontWeight="bold",
        )
        .encode(
            x=alt.X("feat1_woe:Q"),
            y=alt.Y("feat2_woe:Q"),
            text=alt.Text("combined_label:N"),
            color=alt.value("black"),  # text color
        )
    )

    # Combine all plots
    combined_chart = (
        (contour + scatter + text_labels)
        .resolve_scale(color="shared")
        .properties(width=500, height=400, title=f"{encoder}: Decision Surface - {feat1} vs {feat2}")
    )

    mo.ui.altair_chart(combined_chart)
    return encoder, feat1, feat2, plot_data, scorer_


@app.cell
def _(encoder, feat1, feat2, mo, plot_data, scorer_):
    # Cell 3: Show WOE mappings and unique category combinations
    mapping1 = scorer_.get_mapping(feat1)
    mapping2 = scorer_.get_mapping(feat2)

    # Show unique combinations and their predicted probabilities
    unique_combinations = plot_data[['feat1_cat', 'feat2_cat', 'feat1_woe', 'feat2_woe', 'predicted_prob']].drop_duplicates()
    unique_combinations = unique_combinations.sort_values('predicted_prob', ascending=False)

    mo.md(f"**Unique Category Combinations - Ranked by Predicted Probability**")
    mo.ui.table(unique_combinations.round(4))

    mo.hstack([
        mo.vstack([
            mo.md(f"**{feat1} WOE Mapping ({encoder})**"),
            mo.ui.table(mapping1.round(4))
        ]),
        mo.vstack([
            mo.md(f"**{feat2} WOE Mapping ({encoder})**"), 
            mo.ui.table(mapping2.round(4))
        ])
    ])
    return


if __name__ == "__main__":
    app.run()

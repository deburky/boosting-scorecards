import numpy as np
import pandas as pd
import xgboost as xgb
from optbinning import BinningProcess, OptimalBinning, Scorecard
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from typing import List, Dict, Any


"""
Function to calculate an Expected Calibration Error (ECE) on observation level.
"""

def calibration_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate calibration score.
    Calibration_score is derived as 1 - ECE (Expected Calibration Error)
    
    Expected Calibration Error is calculated without binning in the same
    logic as Brier's score decomposition into calibration and refinement.
    Calibration in Brier's score is an average of squared differences (L2).
    This makes the Brier calibration sensitive, the goal of using ECE
    (L1) is to ensure robust treatment of deviations similar to ECE score
    based on binning (e.g., 10 equal bins).
    

    Author: A. Calabourdin (hybe.io)
    Source: https://github.com/ColdTeapot273K/scikit-learn/blob/brier-score-binless-decomposition/
    
    Edits by D. Burakov (N26)

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted probabilities.

    Returns:
        float: ECE score.
    """
    
    # Sort predicted probabilities and ground truth labels based on predictions
    a = np.stack((y_pred, y_true), axis=-1)
    a = a[a[:, 0].argsort()]

    # Calculate the observed and predicted probabilities
    p_i = np.array([
        x.mean()
        for x in np.split(a[:, 1], np.unique(a[:, 0], return_index=True)[1][1:])
    ])
    p_i_hat, n_i = np.unique(a[:, 0], return_counts=True)

    # Calculate calibration loss using and L1 (robust) approach
    clb_score = 1 - (np.sum(n_i * abs(p_i - p_i_hat)) / a[:, 1].shape[0])

    return clb_score

"""
Function to create a WOE LR pipeline.
"""

def create_woe_lr_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    variable_names: List[str], special_codes: Dict[str, Any], 
    binning_fit_params: Dict[str, Any], 
    lr_params: Dict[str, Any]
    ) -> Pipeline:
    
    # Instantiate a binning process class
    binning_process = BinningProcess(
        variable_names=variable_names,
        special_codes=special_codes,
        binning_fit_params=binning_fit_params
    )

    # Building a pipeline
    woe_lr_model = Pipeline(
        steps=[
            ("woe_binner", binning_process),
            ("logistic_regression", LogisticRegression(**lr_params)),
        ]
    )
    
    woe_lr_model.fit(X, y)

    return woe_lr_model

"""
Function to create an XGBoost object.
"""

def create_xgb_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    best_params: Dict[str, Any],
    eval_set
    ) -> Pipeline:

    xgb_model = XGBClassifier(**best_params)
    xgb_model.fit(X, y, eval_set=eval_set, verbose=False)
    
    return xgb_model


"""
Module with plots for visualizing model performance.
"""

# Gabriel S. Gonçalves <gabrielgoncalvesbr@gmail.com>
# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

# Edits by Denis Burakov (https://github.com/deburky)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.utils import check_array
from sklearn.utils import check_consistent_length

def _check_arrays(y, y_pred):
    y = check_array(y, ensure_2d=False, force_all_finite=True)
    y_pred = check_array(y_pred, ensure_2d=False, force_all_finite=True)

    check_consistent_length(y, y_pred)

    return y, y_pred


def _check_parameters(title, xlabel, ylabel, savefig, fname):
    if title is not None and not isinstance(title, str):
        raise TypeError("title must be a string or None; got {}."
                        .format(title))

    if xlabel is not None and not isinstance(xlabel, str):
        raise TypeError("xlabel must be a string or None; got {}."
                        .format(xlabel))

    if ylabel is not None and not isinstance(ylabel, str):
        raise TypeError("ylabel must be a string or None; got {}."
                        .format(ylabel))

    if not isinstance(savefig, bool):
        raise TypeError("savefig must be a boolean; got {}.".format(savefig))

    if fname is not None and not isinstance(fname, str):
        raise TypeError("fname must be a string or None; got {}."
                        .format(fname))

    if savefig is True and fname is None:
        raise ValueError("fname must be provided if savefig is True.")


# changes in model_name and line_color to plot multiple curves
def plot_cap(y, y_pred, model_name='Model', title=None, xlabel=None, ylabel=None,
             legend_title=None,
             savefig=False, line_color='g', fname=None, **kwargs):
    """Plot Cumulative Accuracy Profile (CAP).
    Parameters
    ----------
    y : array-like, shape = (n_samples,)
        Array with the target labels.
    y_pred : array-like, shape = (n_samples,)
        Array with predicted probabilities.
    title : str or None, optional (default=None)
        Title for the plot.
    xlabel : str or None, optional (default=None)
        Label for the x-axis.
    ylabel : str or None, optional (default=None)
        Label for the y-axis.
    savefig : bool (default=False)
        Whether to save the figure.
    fname : str or None, optional (default=None)
        Name for the figure file.
    **kwargs : keyword arguments
        Keyword arguments for matplotlib.pyplot.savefig().
    """
    y, y_pred = _check_arrays(y, y_pred)

    _check_parameters(title, xlabel, ylabel, savefig, fname)

    n_samples = y.shape[0]
    n_event = np.sum(y)

    idx = y_pred.argsort()[::-1][:n_samples]
    yy = y[idx]

    p_event = np.append([0], np.cumsum(yy)) / n_event
    p_population = np.arange(0, n_samples + 1) / n_samples

    # Define the plot settings
    if title is None:
        title = "Cumulative Accuracy Profile (CAP) Curve"
    if xlabel is None:
        xlabel = "Cumulative share of observations"
    if ylabel is None:
        ylabel = "Cumulative share of defaults"

    plt.plot([0, 1], [0, 1], color='red', linewidth=2)
    plt.plot([0, n_event / n_samples, 1], [0, 1, 1], linewidth=2, color='dodgerblue') # changed colors and line width
    plt.plot(p_population, p_event, linewidth=2, color=line_color,
             label=(f"{legend_title}")) # changed to Python 3
             
    plt.title(title, fontdict={'fontsize': 14})
    plt.xlabel(xlabel, fontdict={'fontsize': 12})
    plt.ylabel(ylabel, fontdict={'fontsize': 12})
    plt.legend(loc='lower right')

    # Save figure if requested. Pass kwargs.
    if savefig:
        plt.savefig(fname=fname, **kwargs)
        plt.close()
        

"""
Class for XGBoost tree parsing.
Authors: P. Edwards, S. Denton (Scotiabank)
"""

import pandas as pd
import numpy as np

class XGBoostTreeParser:
    def __init__(self, bstr):
        self.mdf_leafs, self.mdf_parents = self.get_booster_leafs(bstr)
    
    def get_booster_leafs(self, bstr):
        mdf = bstr.trees_to_dataframe()
        mdf_parents = mdf[mdf.Feature != 'Leaf'].drop(columns=['Tree', 'Gain', 'Cover'])
        mdf_leafs = mdf[mdf.Feature == 'Leaf'].drop(columns=['Feature', 'Split', 'Yes', 'No', 'Missing', 'Cover', 'Category'])
        mdf_leafs.rename(columns={'ID': 'ID0', 'Node': 'Node0'}, inplace=True)
        return mdf_leafs, mdf_parents

    def get_tree_traceback(self): 
        tree_traceback = pd.DataFrame()
        itr = 0 
        itrs = str(itr)
        while self.mdf_leafs.shape[0] > 0:
            NoSprout = pd.merge(self.mdf_leafs, self.mdf_parents, how='inner', left_on='ID'+itrs, right_on='No')
            YesSprout = pd.merge(self.mdf_leafs, self.mdf_parents, how='inner', left_on='ID'+itrs, right_on='Yes')
            MissingSprout = pd.merge(self.mdf_leafs, self.mdf_parents, how='inner', left_on='ID'+itrs, right_on='Missing')
            MissingSprout.Split = np.nan

            itr += 1
            itrs = str(itr)    
            NoSprout.insert(NoSprout.shape[1]-4, 'Sign'+itrs, '>=')
            YesSprout.insert(YesSprout.shape[1]-4, 'Sign'+itrs, '<')
            MissingSprout.insert(MissingSprout.shape[1]-4, 'Sign'+itrs, '.')
            self.mdf_leafs = pd.concat([NoSprout, YesSprout, MissingSprout]) 
            self.mdf_leafs.rename(columns={'ID':'ID'+itrs, 
                                    'Split':'Split'+itrs, 
                                    'Feature':'Feature'+itrs, 
                                    'Node':'Node'+itrs, 
                                    'Yes':'Yes'+itrs, 
                                    'No':'No'+itrs, 
                                    'Missing':'Missing'+itrs,
                                    }, inplace=True)
            
            tree_traceback = pd.concat([tree_traceback, self.mdf_leafs.loc[self.mdf_leafs['Node' + itrs] == 0, :]], sort=False)
            self.mdf_leafs = self.mdf_leafs[self.mdf_leafs['Node'+itrs]!=0]
            
        ttb_missing = tree_traceback.copy()
        ttb_non_missing = tree_traceback.copy()
        
        for i in range(1,itr+1): 
            ttb_missing = ttb_missing[(ttb_missing['Sign'+str(i)] == '.') | ttb_missing['Sign'+str(i)].isna()]
            ttb_non_missing = ttb_non_missing[ttb_non_missing['Sign'+str(i)] != '.']

        ttb = ttb_non_missing.copy()
        ttb.sort_values(['Tree', 'Split1', 'Sign1'], inplace=True, na_position='first')
        ttb.reset_index(drop=True, inplace=True)
        
        return ttb, ttb_missing, tree_traceback, self.mdf_leafs, itr, itrs

    def setup_scorecard(self, ttb, ttb_missing, itr):
        sc_df = ttb.iloc[:,:4].rename(columns={'ID0':'ID', 'Node0':'Node', 'Gain':'XAddEvidence'}).copy()
        sc_df['Feature'] = ttb.Feature1.values
        sc_df['Sign'] = ttb.Sign1.values
        sc_df['Split'] = ttb.Split1.values

        for i in range(1,itr): 
            replace_in_sc = ( ( sc_df['Sign']=='>=').values 
                                & (ttb['Split'+str(i)] < ttb['Split'+str(i+1)]).values 
                                & (ttb['Feature'+str(i)] == ttb['Feature'+str(i+1)]).values ) 

            sc_df.loc[replace_in_sc,'Sign'] = ttb['Sign'+str(i+1)][replace_in_sc].values
            sc_df.loc[replace_in_sc,'Split'] = ttb['Split'+str(i+1)][replace_in_sc].values

        sc_df['Inc_Missing'] = sc_df.ID.isin(ttb_missing.ID0).astype(int) 
        
        # Add XAddEvidence col
        cols = sc_df.columns.to_list()
        cols.pop(cols.index('XAddEvidence')) 
        sc_df = sc_df[cols+['XAddEvidence']]
        
        return sc_df

    def format_scorecard(self, sc_df):

        # sc_df.sort_values(['Feature', 'Tree',  'Sign', 'Split'], inplace=True, na_position='last')
        sc_df.sort_values(['Tree','Feature', 'Split', 'Sign'], inplace=True, na_position='last')
        sc_df.set_index(['Tree'], inplace=True)

        scorecard = sc_df.drop(columns=['Node', 'ID'])
        #scorecard.sort_values(by='Feature', inplace=True).sort_index()

        pd.set_option('display.max_rows', scorecard.shape[0]+1)
        return scorecard

    def setup_pointscard(self, scorecard, PDO=50, standardSc_pts=500, standardSc_odds=19, pts_dec_prec=0, base_rate=None):
        """Append 'Points' field to a scorecard dataframe

        Parameters
        ----------
        scdf : DataFrame
            A scorecard dataframe
        PDO : int, float, optional (default=20)
            Points to double odds - number of points needed for the outcome odds to double
        standardSc_pts : int, float, optional (default=500)
            Standard score points - a fixed point on the points scale with fixed odds 
        standardSc_odds : int, float, optional (default=19)
            Standard Odds - odds of good at standard score fixed point
        pts_dec_prec : int, optional (default=1)
            Decimal places to show in scorecard points 
        trndict : dictionary, optional (default=None)
            The output of woesc.describe_data_g_targ(trndat, targ_var)

        Returns
        -------
        pointscard : DataFrame
            The input scorecard with points column appended
        """

        scdf = scorecard.copy()
        scdf['Score'] = scdf.XAddEvidence.values
        scdf.base_score = base_rate
        
        factor = PDO / np.log(2)
        offset = standardSc_pts - factor * np.log(standardSc_odds)

        # Scale the XAddEvidence scores (xgb gain)
        sclSc = factor * scdf.Score  

        # var_offsets = sclSc.groupby(level='var_name').max()
        var_offsets = sclSc.groupby(level='Tree').max()  # Each tree only has 1 variable or Feature

        # Do Score Offset
        tmp_sclSc = pd.DataFrame(sclSc)
        tmp_sclSc.columns=['score']

        tmp_offsets = pd.DataFrame(var_offsets)
        tmp_offsets.columns=['offsets']

        tmp = pd.merge(tmp_sclSc, tmp_offsets,
                left_index=True, right_index=True)

        ## The negative sign here flips the evidence scale (for <--> against)
        shftSc = -tmp['score'] + tmp['offsets']

        pointscard = scdf.copy()
        pointscard['Points'] = shftSc.round(pts_dec_prec).values
        shft_base_pts = (offset - var_offsets.sum()).round(pts_dec_prec)

        if (pts_dec_prec <= 0):
            pointscard['Points'] = pointscard.Points.astype(int)
            shft_base_pts = shft_base_pts.astype(int)
        
        return pointscard

    def generate_scorecard(self, bstr, PDO=50, standardSc_pts=500, standardSc_odds=19, pts_dec_prec=0, base_rate=None):
        
        self.mdf_leafs, self.mdf_parents = self.get_booster_leafs(bstr)
        
        ttb, ttb_missing, tree_traceback, mdf_leafs, itr, itrs = self.get_tree_traceback()
        sc_df = self.setup_scorecard(ttb, ttb_missing, itr)
        
        scorecard = self.format_scorecard(sc_df)

        pointscard = self.setup_pointscard(scorecard, PDO=PDO, standardSc_pts=standardSc_pts, 
                                    standardSc_odds=standardSc_odds, pts_dec_prec=pts_dec_prec, 
                                    base_rate=base_rate)
        return pointscard


"""
Function to create credit scores for tree-based models.
Author: D. Burakov (N26).
"""

# scores for features and total score
def get_credit_scores(df: pd.DataFrame = None, features: List[str] = None,
                      scorecard: pd.DataFrame = None) -> pd.DataFrame:
    
    if df is None or features is None or scorecard is None:
        raise ValueError("df, features, and scorecard parameters are required.")
    
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a DataFrame.")
    
    if not isinstance(features, list):
        raise TypeError("features must be a list.")
    
    if not isinstance(scorecard, pd.DataFrame):
        raise TypeError("scorecard must be a DataFrame.")
    
    dataframe = df[features].copy()
    val_scores = np.zeros(len(dataframe))
    
    for feature in features:
        feature_scores = np.zeros(len(dataframe))
        
        for tree_id in scorecard.index.unique():
            criteria = scorecard.loc[tree_id]
            
            for index, row in criteria.iterrows():
                sc_feature = row['Feature']
                sc_sign = row['Sign']
                sc_split = row['Split']
                sc_score = row['Points']
                
                if sc_feature == feature:
                    if sc_sign == '<':
                        feature_scores += (dataframe[sc_feature] < sc_split) * sc_score
                    elif sc_sign == '>=':
                        feature_scores += (dataframe[sc_feature] >= sc_split) * sc_score
        
        dataframe[feature] = feature_scores
    
    dataframe['Score'] = dataframe[features].sum(axis=1)
    
    return dataframe[features + ['Score']]


"""
Function to create credit scores for WOE LR models.
Author: D. Burakov (N26).
Adapted from: https://github.com/guillermo-navas-palencia/optbinning/scorecard/scorecard.py
"""

def woe_scorer(X: pd.DataFrame, scorecard: Scorecard, variable_name: str = None) -> np.ndarray:
    """
    Calculate the scores for a given dataset using a Scorecard model.

    Args:
        X (pd.DataFrame): The input dataset with features to score.
        scorecard (Scorecard): The trained Scorecard model.
        variable_name (str, optional): The name of the specific variable to score.
            If provided, only this variable will be scored. If not provided, all selected variables will be scored.

    Returns:
        np.ndarray: An array of scores corresponding to the input dataset.

    Raises:
        ValueError: If the specified variable_name is not in the list of supported variables.
    """
    
    # Check if a specific variable name is provided
    if variable_name is not None:
        # Check if the variable name is in the list of supported variables
        if variable_name not in scorecard.binning_process_.get_support(names=True):
            raise ValueError(f"Variable '{variable_name}' is not in the list of supported variables.")
    else:
        # Use all selected variables if none is specified
        variable_name = scorecard.binning_process_.get_support(names=True)

    # If a single variable is specified, convert it to a list
    if isinstance(variable_name, str):
        variable_name = [variable_name]

    X_t = scorecard.binning_process.transform(
        X=X, metric="indices", metric_special="empirical",
        metric_missing="empirical")

    score = np.zeros(X_t.shape[0])

    df_scorecard = scorecard.table(style="summary")

    for variable in variable_name:
        mask = df_scorecard.Variable == variable
        points = df_scorecard[mask].Points.values
        score += points[X_t[variable]]

    return score
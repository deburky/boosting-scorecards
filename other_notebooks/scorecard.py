import pandas as pd
import numpy as np

def get_booster_leafs(bstr):
    mdf = bstr.trees_to_dataframe()
    mdf_parents = mdf[mdf.Feature!='Leaf'].drop(columns=['Tree','Gain','Cover'])
    mdf_leafs = mdf[mdf.Feature=='Leaf'].drop(columns=['Feature','Split','Yes','No','Missing','Cover'])
    mdf_leafs.rename(columns={'ID': 'ID0', 'Node': 'Node0'}, inplace=True)
    return mdf_leafs, mdf_parents
    
def get_tree_traceback(mdf_leafs, mdf_parents): 
    tree_traceback = pd.DataFrame()
    itr = 0
    itrs = str(itr)
    while mdf_leafs.shape[0] > 0:
        NoSprout = pd.merge(mdf_leafs, mdf_parents, how='inner', left_on='ID'+itrs, right_on='No')
        YesSprout = pd.merge(mdf_leafs, mdf_parents, how='inner', left_on='ID'+itrs, right_on='Yes')
        MissingSprout = pd.merge(mdf_leafs, mdf_parents, how='inner', left_on='ID'+itrs, right_on='Missing')
        MissingSprout.Split = np.nan

        itr += 1
        itrs = str(itr)
        NoSprout.insert(NoSprout.shape[1]-4, f'Sign{itrs}', '>=')
        YesSprout.insert(YesSprout.shape[1]-4, f'Sign{itrs}', '<')
        MissingSprout.insert(MissingSprout.shape[1]-4, f'Sign{itrs}', '.')
        mdf_leafs = pd.concat([NoSprout, YesSprout, MissingSprout])
        mdf_leafs.rename(
            columns={
                'ID': f'ID{itrs}',
                'Split': f'Split{itrs}',
                'Feature': f'Feature{itrs}',
                'Node': f'Node{itrs}',
                'Yes': f'Yes{itrs}',
                'No': f'No{itrs}',
                'Missing': f'Missing{itrs}',
            },
            inplace=True,
        )

        tree_traceback = tree_traceback.append(
            mdf_leafs.loc[mdf_leafs[f'Node{itrs}'] == 0, :], sort=False
        )
        mdf_leafs = mdf_leafs[mdf_leafs[f'Node{itrs}'] != 0]

    ttb_missing = tree_traceback.copy()
    ttb_non_missing = tree_traceback.copy()
    for i in range(1,itr+1): 
        ttb_missing = ttb_missing[
            (ttb_missing[f'Sign{str(i)}'] == '.')
            | ttb_missing[f'Sign{str(i)}'].isna()
        ]
        ttb_non_missing = ttb_non_missing[ttb_non_missing[f'Sign{str(i)}'] != '.']

    ttb = ttb_non_missing.copy()
    ttb.sort_values(['Tree', 'Split1', 'Sign1'], inplace=True, na_position='first')
    ttb.reset_index(drop=True, inplace=True)

    return ttb, ttb_missing, tree_traceback, mdf_leafs, itr, itrs

def setup_scorecard(ttb, ttb_missing, itr):
    sc_df = ttb.iloc[:,:4].rename(columns={'ID0':'ID', 'Node0':'Node', 'Gain':'XAddEvidence'}).copy()
    sc_df['Feature'] = ttb.Feature1.values
    sc_df['Sign'] = ttb.Sign1.values
    sc_df['Split'] = ttb.Split1.values

    for i in range(1,itr): 
        replace_in_sc = (
            (sc_df['Sign'] == '>=').values
            & (ttb[f'Split{str(i)}'] < ttb[f'Split{str(i + 1)}']).values
        ) & (ttb[f'Feature{str(i)}'] == ttb[f'Feature{str(i + 1)}']).values 

        sc_df.loc[replace_in_sc, 'Sign'] = ttb[f'Sign{str(i + 1)}'][
            replace_in_sc
        ].values
        sc_df.loc[replace_in_sc, 'Split'] = ttb[f'Split{str(i + 1)}'][
            replace_in_sc
        ].values

    sc_df['Inc_Missing'] = sc_df.ID.isin(ttb_missing.ID0).astype(int) 

    # Add XAddEvidence col
    cols = sc_df.columns.to_list()
    cols.pop(cols.index('XAddEvidence'))
    sc_df = sc_df[cols+['XAddEvidence']]
    return sc_df

def format_scorecard(sc_df):
    OTHER_CAT_IND = "OTHER"  ## The label for the all other items in categorical variable
    feature_decomp = sc_df.Feature.str.split('__', n=1, expand=True)
    # cat_rows = ~feature_decomp[0].isna()
    cat_rows = ~feature_decomp[1].isna()
    other_cat_rows = (cat_rows & (sc_df['Sign'] == '<')).values
    feat_categories = feature_decomp.iloc[:,1].copy()
    # feat_categories = feature_decomp.iloc[:,0].copy()
    feat_categories.loc[other_cat_rows] = OTHER_CAT_IND

    sc_df.loc[cat_rows, 'Split'] = feat_categories[cat_rows].values
    sc_df.loc[cat_rows, 'Feature'] = feature_decomp[0][cat_rows].values
    sc_df.loc[cat_rows, 'Sign'] = "="
    sc_df.loc[cat_rows, 'Inc_Missing'] = 0

    # sc_df.sort_values(['Feature', 'Tree',  'Sign', 'Split'], inplace=True, na_position='last')
    sc_df.sort_values(['Tree','Feature', 'Split', 'Sign'], inplace=True, na_position='last')
    sc_df.set_index(['Tree'], inplace=True)

    scorecard = sc_df.drop(columns=['Node', 'ID'])
    #scorecard.sort_values(by='Feature', inplace=True).sort_index()

    pd.set_option('display.max_rows', scorecard.shape[0]+1)
    return scorecard

def setup_pointscard(scorecard, PDO=50, standardSc_pts=500, standardSc_odds=19, pts_dec_prec=0, base_rate=None):
    """Append 'Points' field to a scorecard dataframe

    Parameters
    ----------
    scdf : DataFrame
        A scorecard dataframe
    PDO : int, float, optional (default=20)
        Points to double odds - number of points needed for the outcome odds to double
    standardSc_pts : int, float, optional (default=600)
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
    # print("Offset: {:.7g}, Factor: {:.6g}".format(offset, factor))

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

def generate_scorecard(bstr, PDO=50, standardSc_pts=500, standardSc_odds=19, pts_dec_prec=0, base_rate=None):
    mdf_leafs, mdf_parents = get_booster_leafs(bstr)
    ttb, ttb_missing, tree_traceback, mdf_leafs, itr, itrs = get_tree_traceback(mdf_leafs, mdf_parents)
    sc_df = setup_scorecard(ttb, ttb_missing, itr)
    scorecard = format_scorecard(sc_df)
    return setup_pointscard(
        scorecard,
        PDO=PDO,
        standardSc_pts=standardSc_pts,
        standardSc_odds=standardSc_odds,
        pts_dec_prec=pts_dec_prec,
        base_rate=base_rate,
    )
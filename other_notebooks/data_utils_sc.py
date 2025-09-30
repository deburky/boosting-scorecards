import pandas as pd
import numpy as np


def describe_data_g_targ(dat_df, target_var, logbase=np.e):
    """Describe the data given a target variable
    
    Parameters
    ----------
    dat_df : DataFrame
        A dataframe that contains the target variable 
    target_var : string
        The model target variable, corresponding to a binary variable in dat_df
    logbase : int, float, optional (default=np.e) 
        The base for logarithm functions used to compute log-odds - default is natural log (ln = log_e)
    
    Returns
    -------
    Dict
        A dictionary containing calculated values
    """
    num = dat_df.shape[0]
    n_targ = (dat_df[target_var]==1).sum()
    n_ctrl = (dat_df[target_var]==0).sum()
    assert n_ctrl + n_targ == num
    base_rate = n_targ/num
    base_odds = n_targ/n_ctrl
    lbm = 1/np.log(logbase)
    base_log_odds = np.log(base_odds) * lbm
    NLL_null = -(dat_df[target_var] * np.log(base_rate)*lbm 
                + (1-dat_df[target_var]) * np.log(1-base_rate)*lbm).sum()
    LogLoss_null = NLL_null/num
    
    print("Number of records (num):", num)
    print("Target count (n_targ):", n_targ)
    print("Target rate (base_rate):", base_rate)
    print("Target odds (base_odds):", base_odds)
    print("Target log odds (base_log_odds):", base_log_odds)
    print("Dummy model negative log-likelihood (NLL_null):", NLL_null)
    print("Dummy model LogLoss (LogLoss_null):", LogLoss_null)
    print("")
    return {'num':num, 'n_targ':n_targ , 'base_rate':base_rate, 'base_odds':base_odds
           , 'base_log_odds':base_log_odds, 'NLL_null':NLL_null, 'LogLoss_null':LogLoss_null }

def one_hot_encode_data(dataset, id_vars, targ_var):
    non_p_vars = id_vars + [targ_var]
    p_vars = dataset.columns.drop(non_p_vars).tolist()
#     num_p_vars = sorted(dataset[p_vars]._get_numeric_data().columns)
    num_p_vars = ['AgeInMonths', 'DaysSinceDisbursement', 'LTV', 'PERFORM_CNS_SCORE']
    cat_p_vars = sorted(set(dataset[p_vars].columns) - set(num_p_vars))

    ## One-hot encode categorical variables
    x_dat = dataset.loc[:, non_p_vars + num_p_vars]
    x_dat_c = dataset.loc[:, cat_p_vars]
    for c in cat_p_vars:
        x_dat_c[c] = x_dat_c[c].astype("category")
    dxx = pd.get_dummies(x_dat_c, prefix_sep='__')
    x_dat.shape, x_dat_c.shape, dxx.shape

    ## Regenerate dataset from components 
    dataset = pd.concat([x_dat, dxx], axis=1, sort=False)

    ## Regenerate p_vars to include one-hot encoded variables.
    p_vars = list(dataset.columns.drop(non_p_vars))
    
    return dataset, p_vars


def create_feature_interaction_constraints(p_vars, data_dir):
    x_constraints = []
    interaction_constraints = []

    dat_cols = pd.Index(p_vars)
    ind_vars = sorted(set([item[0] for item in dat_cols.str.split('__')]))

    for iv in ind_vars:
        if np.any(dat_cols.isin([iv])):
            f_var = [dat_cols.get_loc(iv)]
            f_var_name = [dat_cols[f_var[0]]]
        else:
            f_var = []
            f_var_name = []

        x_constraints.append(f_var_name + list(dat_cols[np.where(dat_cols.str.startswith(iv+'__'))[0]]) )
        interaction_constraints.append(f_var + list(np.where(dat_cols.str.startswith(iv+'__'))[0]))
    
    # Save the constraints to file
    constraints_df = pd.DataFrame({'x_contraints':x_constraints, 
                                'interaction_constraints':interaction_constraints})
    constraints_path = data_dir/'interaction_constraints.csv'
    constraints_df.to_csv(constraints_path, index=False)
    
    return x_constraints, interaction_constraints, constraints_path

def get_monotonic_constraints(monotonic_vars, p_vars, data_dir):
    monotonic_constraints = pd.Series(0, index=p_vars)
    for mono_var in monotonic_vars:
        monotonic_constraints[mono_var] = 1

    monotonic_constraints_str = "("+ ",".join([str(i) for i in monotonic_constraints.values]) +")"
    
    return monotonic_constraints_str

def load_training_data(run, data_dir, artifact_name:str, train_file='train.csv', val_file='val.csv'):
    data_art = run.use_artifact(artifact_name)
    dataset_dir = data_art.download(data_dir)
    trndat = pd.read_csv(dataset_dir/train_file)
    valdat = pd.read_csv(dataset_dir/val_file)
    return trndat, valdat


def calculate_credit_scores(n_samples=1000,  p_vars=None, pointscard=None, valdat=None):
    val_scores = [0] * n_samples
    OTHER_CAT_IND = 'OTHER'

    MINI_DATA = valdat[p_vars].iloc[:n_samples,:].copy()
    tree_dict = {}
    # For each datapoint in the dataset
    for i, val_row in enumerate(MINI_DATA.iterrows()):

        # For each Tree in the scorecard
        for tree_id in pointscard.index.unique():
            tree_dict[tree_id] = {'has_state_score':0,
                                 'has_manufacturer_score':0,
                                 'has_employment_score':0}
            
            # For each row in that tree
            for criteria in pointscard.loc[[int(tree_id)]].iterrows():
                sc_feature = criteria[1]['Feature']
                sc_sign = criteria[1]['Sign']
                sc_split = criteria[1]['Split']
                sc_score = criteria[1]['Points']
                
                # If categorical variable
                if sc_sign == '=':
                    sc_col = sc_feature + '__' + sc_split
                else: 
                    sc_col = sc_feature

                # Add Credit Scores
                if sc_sign == '=':
                    
                    # test if `sc_col` column exists
                    try:
                        if val_row[1][sc_col] == 1:
                            val_scores[i] += sc_score 

                            if 'state' in sc_feature.lower(): 
                                tree_dict[tree_id]['has_state_score'] = 1
                            elif 'manufacturer_id' in sc_feature.lower(): 
                                tree_dict[tree_id]['has_manufacturer_score'] = 1
                            elif 'employment' in sc_feature.lower(): 
                                tree_dict[tree_id]['has_employment_score'] = 1

                    except:
                        # try format for "OTHER columns, ending in __"
                        try:
                            sc_col = sc_feature + '__'
                            if val_row[1][sc_col] == 1:
                                val_scores[i] += sc_score 
                                
                                if 'state' in sc_feature.lower(): 
                                    tree_dict[tree_id]['has_state_score'] = 1
                                elif 'manufacturer_id' in sc_feature.lower(): 
                                    tree_dict[tree_id]['has_manufacturer_score'] = 1
                                elif 'employment' in sc_feature.lower(): 
                                    tree_dict[tree_id]['has_employment_score'] = 1
                        except: pass
                    
                    # Assign scores for categorical values if they fall into the "OTHER" class
                    if 'state' in sc_feature.lower() and \
                        tree_dict[tree_id]['has_state_score'] == 0 \
                        and sc_split.lower() == 'other':
                        val_scores[i] += sc_score 
                        tree_dict[tree_id]['has_state_score'] = 1
                    
                    elif 'manufacturer_id' in sc_feature.lower() and \
                        tree_dict[tree_id]['has_manufacturer_score'] == 0 and \
                        sc_split.lower() == 'other':
                        val_scores[i] += sc_score 
                        tree_dict[tree_id]['has_manufacturer_score'] = 1
                    
                    elif 'employment' in sc_feature.lower() and \
                        tree_dict[tree_id]['has_employment_score'] == 0 and \
                        sc_split.lower() == 'other':
                        val_scores[i] += sc_score 
                        tree_dict[tree_id]['has_employment_score'] = 1

                elif sc_sign == '<': 
                    if val_row[1][sc_col] < sc_split:
                        val_scores[i] += sc_score 

                elif sc_sign == '>=': 
                    if val_row[1][sc_col] >= sc_split:
                        val_scores[i] += sc_score
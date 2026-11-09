import numpy as np

def group_mean_change(pre_outcomes, post_outcomes):
    np_pre = np.asarray(pre_outcomes)
    np_post = np.asarray(post_outcomes)
    pre_average = np.mean(np_pre)
    post_average = np.mean(np_post)
    return post_average - pre_average 

def difference_in_differences_simple(treated_pre, treated_post, control_pre, control_post):
    treated_change = group_mean_change(treated_pre, treated_post)
    control_change = group_mean_change(control_pre, control_post)
    return treated_change - control_change

def build_did_design_matrix(treatment_indicator, post_indicator):
    if len(treatment_indicator) != len(post_indicator):
        raise ValueError("Treatment indicator and post indicator vectors are not of the same length.")
    
    n = len(treatment_indicator)
    treat_ind_np = np.asarray(treatment_indicator)
    post_ind_np = np.asarray(post_indicator)

    intercept_np = np.ones((n, 1))
    interaction_np = treat_ind_np * post_ind_np

    design_matrix = np.column_stack((intercept_np, treat_ind_np, post_ind_np, interaction_np))
    return design_matrix

def ols_normal_equations(design_matrix, outcomes):
    design_matrix_np = np.asarray(design_matrix, dtype = float)
    outcomes_np = np.asarray(outcomes, dtype = float)

    beta_hat, *_ = np.linalg.lstsq(design_matrix_np, outcomes_np, rcond = None)

    return beta_hat

def did_effect_from_regression(treatment_indicator, post_indicator, outcomes):
    design_matrix = build_did_design_matrix(treatment_indicator, post_indicator)
    beta_hat = ols_normal_equations(design_matrix, outcomes)

    effect = beta_hat[3]
    if np.isclose(effect, 0.0, atol=1e-12):
        effect = 0.0
    
    return effect
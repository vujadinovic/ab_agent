import numpy as np
from ab_agent.stats.two_prop_hyp_test import (
    confidence_interval_from_se, 
    two_sided_p_value,
)

def group_mean_change(pre_outcomes, post_outcomes):
    np_pre = np.asarray(pre_outcomes)
    np_post = np.asarray(post_outcomes)
    if np_pre.size == 0 or np_post.size == 0:
        raise ValueError("Pre and post outcomes must not be empty.")
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

    for t in (0,1):
        for p in (0,1):
            if not np.any((treat_ind_np == t) & (post_ind_np == p)): #proverava da li je bar jedno True. Ako nije celija je prazna.
                raise ValueError(
                    f"No observations with treatment={t} and post={p}."
                    "All four cells are needed to estimate the DiD effect."
                    )

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


def did_standard_error(treatment_indicator, post_indicator, outcomes):
    treat_ind_np = np.asarray(treatment_indicator)
    post_ind_np = np.asarray(post_indicator)
    outcomes_np = np.asarray(outcomes, dtype = float)

    variance = 0.0
    for t in (0,1):
        for p in (0,1):
            cell = outcomes_np[(treat_ind_np == t) & (post_ind_np == p)]
            if cell.size < 2:
                raise ValueError(
                    f"Cell treatment={t}, post={p} has fewer than 2 observations;"
                    "its variance cannot be estimated."
                )
            variance += np.var(cell, ddof=1) / cell.size

    return float(np.sqrt(variance))


def did_test(treatment_indicator, post_indicator, outcomes, confidence_level=0.95):
    effect = float(did_effect_from_regression(treatment_indicator, post_indicator, outcomes))
    se = did_standard_error(treatment_indicator, post_indicator, outcomes)

    if se == 0:
        z = 0.0
        p_value = 1.0
    else:
        z = effect / se
        p_value = two_sided_p_value(z)

    ci_lower, ci_upper = confidence_interval_from_se(effect, se, confidence_level)
    return {
        'effect': effect,
        'se': se,
        'z': float(z),
        'p_value': float(p_value),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
    }
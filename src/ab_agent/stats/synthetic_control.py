import numpy as np

def fit_synthetic_control_weights(treated_pre, donor_pre, num_iterations=5000, learning_rate=0.01):
    if  len(treated_pre) == 0 or len(donor_pre) == 0:
        raise ValueError("Empty array passed to the fit_synthetic_control_weights function.")
    if num_iterations < 1:
        raise ValueError("Number of iterations cannot be less than 1.")
    if learning_rate <= 0:
        raise ValueError("Learning rate cannot be less than or equal to 0.")

    

    n = len(treated_pre)
    k = len(donor_pre[0])
    X = np.asarray(donor_pre)
    y = np.asarray(treated_pre)

    theta = np.zeros(k)
    w = np.exp(theta) / np.sum(np.exp(theta))

    for i in range(num_iterations):
        theta = theta - theta.max()
        w = np.exp(theta) / np.sum(np.exp(theta))
        r = X @ w - y
        g_w = X.T @ r
        g_theta = w * (g_w - w.T @ g_w)
        theta = theta - learning_rate * g_theta

    return np.exp(theta) / np.sum(np.exp(theta))

    
def synthetic_control_effect(treated_post, donor_post, weights):
    results = {}

    results["synthetic"] = np.asarray(donor_post @ weights)
    results["gap"] = treated_post - results["synthetic"] 
    results["average_effect"] = np.sum(results["gap"]) / len(results["synthetic"] )

    return results

    
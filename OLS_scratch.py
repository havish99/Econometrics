import numpy as np

def adjust_df_regress(df, Cols):
    for col in Cols:
        df[col + "_Market Adjusted"] = df[col] - df["RF"]
    df["intercept"] = 1
    cols_req = [(lambda x: x + "_Market Adjusted")(x) for x in Cols]
    cols_req = ["intercept"] + cols_req
    return df[cols_req]

def t_test(beta,b_0,SE,dof,alpha):
    t_val = (beta - b_0)/(SE)
    lower_bound = t.ppf(0.5*alpha, dof)
    upper_bound = t.ppf(1-0.5*alpha, dof)
    if t_val > lower_bound and t_val < upper_bound:
        print(f"Test accepts b_0 = {b_0} with estimate = {beta}, t_val = {t_val} and alpha = {alpha}")
    else:
        print(f"Test rejects b_0 = {b_0} with estimate = {beta}, t_val = {t_val} and alpha = {alpha}")
    CIs = [beta + SE*lower_bound, beta + SE*upper_bound]
    print(f'Confidence Interval is:{CIs} for {100*(1-alpha)}%')
    return CIs

def OLS_regress(X,y,intercept = True):
    if intercept:
        X = np.column_stack([np.ones(X.shape[0]), X])
    else:
        X = X.reshape(-1,1)
    y = y.reshape(-1,1)
    Inv_matrix = np.linalg.inv(X.T.dot(X)) ### (X'X)^-1
    P = X.dot(Inv_matrix).dot(X.T)
    OLS_estimate = P.dot(y)
    beta = Inv_matrix.dot(X.T).dot(y)
    return OLS_estimate,beta
    
def OLS_params(X,y):
    Inv_matrix = np.linalg.inv(X.T.dot(X)) ### (X'X)^-1
    P = X.dot(Inv_matrix).dot(X.T)
    M = np.identity(X.shape[0]) - P
    OLS_estimate = P.dot(y)
    residuals = M.dot(y)
    beta = Inv_matrix.dot(X.T).dot(y)
    residual_variance = np.std(residuals)**2
    y_variance   = np.std(y)**2
    R_squared = 1 - residual_variance/y_variance
    OLS_Covar = Inv_matrix*residual_variance
    white_diagonals = np.diag(residuals**2)
    mid_term = (X.T.dot(white_diagonals)).dot(X)
    white_Covar = (Inv_matrix.dot(mid_term)).dot(Inv_matrix)
    log_likelihood = -0.5*X.shape[0]*(np.log(2*np.pi) + 1 + np.log(residual_variance))
    condition_number = np.linalg.cond(X)
    return [OLS_estimate, residuals, beta, R_squared, OLS_Covar, white_Covar, log_likelihood, condition_number]
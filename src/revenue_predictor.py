import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as scs
import statsmodels.formula.api as smf
from pandas.plotting import scatter_matrix as scatter
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.utils import XyScaler
from sklearn.base import clone


# Workflow:
# 1) Clean Everything
# 2) Split data
#       - Hold out
#       - Train
#       - Test
#       - Standardize x's
# 3) Fit Linear regression
#       - On standardized x_train, y_train
# 4) Calculate RMSE (untransformed X_train....?)
# 5) Try LASSO
# 6) Fit


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def load_pickle(file_path):
    df = pd.read_pickle(file_path)
    return df

def clean_names(df):
    temp_liz = [name.lower().replace(" ","_") for name in df.columns]
    temp_liz = [name.replace(".","_") for name in temp_liz]
    df.columns = temp_liz
    return df

def convert_dates(df, date_column, to_drop):
    ''' Input -- Pandas series of strings: formatted as yyyy-mm-dd
        Output -- Pandas series: integer number of days since the start of the data.
    '''
    old_column = date_column
    date_column = pd.to_datetime(date_column)
    df['date_delta'] = (date_column - date_column.min())  / np.timedelta64(1,'D')
    df.drop(to_drop, axis=1, inplace=True)
    return df

def convert_to_dummies(df, list_of_categorical_columns):
    ''' Input -- Pandas DataFrame
        Input -- List of Pandas Series -- Categorical column to be dummied
        Output -- DataFrame with the categorical column replaced with dummy columns
    '''
    for categorical_column in list_of_categorical_columns:
        dummied_column = pd.get_dummies(df[categorical_column], prefix='cat', drop_first=True)
        df = pd.concat([df, dummied_column], axis=1)
        df.drop(categorical_column, axis=1, inplace=True)
    return df

df = load_data('/Users/alanteran/galvanize/restaurant-revenue-prediction/data/train.csv')
# df = load_pickle('/Users/alanteran/galvanize/restaurant-revenue-prediction/src/dropped_influencers_and_vifs.pkl')
df = clean_names(df)
df = convert_dates(df, df['open_date'], 'open_date')
df = convert_to_dummies(df, ['type', 'city_group', 'p9', 'p13', 'p33'])
df = clean_names(df)

## BEGIN TO SPLIT DATA ###
idees = df.pop('id')
cities = df.pop('city')
y = df.pop('revenue') # ENTIRE SET OF DATA
X = df  # ENTIRE SET OF DATA


def get_vifs(df):
    '''
    Input -- Pandas DataFrame, calculates the vifs for each variable in the DataFrame.
    Returns list of tuples: (column name, vif value)
    '''
    vifs = []
    cols = df.columns
    for index in range(df.shape[1]):
        vifs.append(round(variance_inflation_factor(df.values, index),2))
    return [(column, vif) for column, vif in zip(cols, vifs)]

# get_vifs(X)

def plot_vifs(df):
    sorted_vifs = sorted(get_vifs(df), key=lambda x:x[1])
    N = len(sorted_vifs)
    features_sorted= []
    vifs_sorted= []
    for elem in sorted_vifs:
        features_sorted.append(elem[0])
        vifs_sorted.append(elem[1])
    features_sorted.reverse()
    vifs_sorted.reverse()

    ind = np.arange(N)
    width = 0.35
    fig, ax = plt.subplots(figsize=(10,4))
    rects1 = ax.bar(ind, vifs_sorted, width, color='r')

    ax.set_ylabel('Variance Inflation Factor')
    ax.set_title('Distribution of VIFs')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(features_sorted, rotation='90')
    ax.axhline(10, color='blue')

    return plt.show()

# plot_vifs(X)


# FINISH SPLITTING DATA

X_working, X_hold_out, y_working, y_hold_out = train_test_split(X, y, test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(X_working, y_working, test_size=0.2)
scaler = StandardScaler()
scaler2 = StandardScaler()
# scaler.fit(X_train, X_test) ## WTF? did I fuck this up? I think so, but it still works, since I'm not using y transformed.
scaler.fit(X_train, y_train.values.reshape(-1,1))
X_train_std, y_train_std = scaler.transform(X_train, y_train.values.reshape(-1,1))
X_train_std = pd.DataFrame(X_train, columns=X_train.columns)
X_test_std, y_test_std = scaler.transform(X_test, y_test.values.reshape(-1,1))
X_test_std = pd.DataFrame(X_test, columns=X_test.columns)
scaler2.fit(X_hold_out, y_hold_out.values.reshape(-1,1))
X_hold_out_std, y_hold_out_std = scaler2.transform(X_hold_out, y_hold_out.values.reshape(-1,1))
X_hold_out_std = pd.DataFrame(X_hold_out, columns=X_hold_out.columns)
# X_hold_out_std = pd.DataFrame(scaler.transform(X_hold_out), columns=X_hold_out.columns)


standardized = pd.concat([pd.DataFrame(X_train_std, columns=X_train.columns), y_train], axis=1)
# Next, fit a linear model on the training data
vanilla_lm = smf.ols(formula = "revenue ~ p1 + p2 + p3 + p4+ p5 + p6 + p7 + p8 + p10 + p11 + p12 + p14 + p15 + p16 + p17 + p18 + p19 + p20 + p21 + p22 + p23 + p24 + p25 + p26 + p27 + p28 + p29 + p30 + p31 + p32 + p34 + p35 + p36 + p37 + date_delta + cat_fc + cat_il + cat_other + cat_5 + cat_8 + cat_10 + cat_4_0 + cat_5_0 + cat_6_0 + cat_7_5 + cat_2 + cat_3 + cat_4 + cat_5 + cat_6", data = standardized).fit()
vanilla_rmse = np.sqrt(mse(y_test, vanilla_lm.predict(X_test_std)))
# 35054027.17   Not too good.

vanilla_sklm = LinearRegression().fit(X_train_std, y_train)
vanilla_sklm.score(X_test_std, y_test)
# -1.71
# This means that picking any random line will be better than the prediction.
# My model was super overfit because the


## NOW LET's Try to improve on that RMSE!
fig,ax = plt.subplots(figsize=(10,10))
ax.plot([0,10000000], [0,10000000], "k-")
ax.scatter(LinearRegression().fit(X_train_std, y_train).predict(X_test_std), y_test)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.savefig('predicted_xaxis__vs_actual_yaxis_vanilla_lm.png');

def cv(X, y, base_estimator, n_folds, random_seed=154):
    '''
    Credit for this code goes to dsi-solns-71 ! Nice job dsi-solns-71!
    '''
    kf = KFold(n_splits=n_folds, random_state=random_seed)
    test_cv_errors, train_cv_errors = np.empty(n_folds), np.empty(n_folds)
    for idx, (train, test) in enumerate(kf.split(X_train_std)):
        X_cv_train, y_cv_train = X[train], y[train]
        X_cv_test, y_cv_test = X[test], y[test]
        standardizer = XyScaler()
        standardizer.fit(X_cv_train, y_cv_train)
        X_cv_train_std, y_cv_train_std = standardizer.transform(X_cv_train, y_cv_train)
        X_cv_test_std, y_cv_test_std = standardizer.transform(X_cv_test, y_cv_test)
        estimator = clone(base_estimator)
        estimator.fit(X_cv_train_std, y_cv_train_std)
        y_hat_train = estimator.predict(X_cv_train_std)
        y_hat_test = estimator.predict(X_cv_test_std)
        train_cv_errors[idx] = rss(y_cv_train_std, y_hat_train)
        test_cv_errors[idx] = rss(y_cv_test_std, y_hat_test)
    return train_cv_errors, test_cv_errors

def rss(y, y_hat):
    '''
    Credit for this code goes to dsi-solns-71 ! Nice job dsi-solns-71!
    '''
    return np.mean((y  - y_hat)**2)

def train_at_various_alphas(X, y, model, alphas, n_folds=10, **kwargs):
    '''
    Credit for this code goes to dsi-solns-71 ! Nice job dsi-solns-71!
    '''
    cv_errors_train = pd.DataFrame(np.empty(shape=(n_folds, len(alphas))),
                                     columns=alphas)
    cv_errors_test = pd.DataFrame(np.empty(shape=(n_folds, len(alphas))),
                                        columns=alphas)
    for alpha in alphas:
        train_fold_errors, test_fold_errors = cv(X, y, model(alpha=alpha, **kwargs), n_folds=n_folds)
        cv_errors_train.loc[:, alpha] = train_fold_errors
        cv_errors_test.loc[:, alpha] = test_fold_errors
    return cv_errors_train, cv_errors_test

def get_optimal_alpha(mean_cv_errors_test):
    '''
    Credit for this code goes to dsi-solns-71 ! Nice job dsi-solns-71!
    '''
    alphas = mean_cv_errors_test.index
    optimal_idx = np.argmin(mean_cv_errors_test.values)
    optimal_alpha = alphas[optimal_idx]
    return optimal_alpha

## LASSO

lasso_alphas = np.logspace(-3, 1, num=250)
lasso_cv_errors_train, lasso_cv_errors_test = train_at_various_alphas(X_train.values, y_train.values, Lasso, lasso_alphas, max_iter=100000)

lasso_mean_cv_errors_train = lasso_cv_errors_train.mean(axis=0)
lasso_mean_cv_errors_test = lasso_cv_errors_test.mean(axis=0)

lasso_optimal_alpha = get_optimal_alpha(lasso_mean_cv_errors_test)


fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(np.log10(lasso_alphas), lasso_mean_cv_errors_train)
ax.plot(np.log10(lasso_alphas), lasso_mean_cv_errors_test)
ax.axvline(np.log10(lasso_optimal_alpha), color='grey')
ax.set_title("LASSO Regression Train and Test MSE")
ax.set_xlabel(r"$\log(\alpha)$")
ax.set_ylabel("MSE")
plt.savefig('lasso_optimal_alpha')



lasso_models = []

for alpha in lasso_alphas:
    scaler = XyScaler()
    scaler.fit(X_train.values, y_train.values)
    X_train_std, y_train_std = scaler.transform(X_train.values, y_train.values)
    lasso = Lasso(alpha=alpha, tol=.01)
    lasso.fit(X_train_std, y_train_std)
    lasso_models.append(lasso)

paths = pd.DataFrame(np.empty(shape=(len(lasso_alphas), len(X_train.columns))),
                     index=lasso_alphas, columns=X_train.columns)

for idx, model in enumerate(lasso_models):
    paths.iloc[idx] = model.coef_

fig, ax = plt.subplots(figsize=(14, 8))
for column in X_train.columns:
    path = paths.loc[:, column]
    ax.plot(np.log10(lasso_alphas), path, label=column)
ax.axvline(np.log10(lasso_optimal_alpha), color='grey')
ax.legend(loc='lower right')
ax.set_title("LASSO Regression, Standardized Coefficient Paths")
ax.set_xlabel(r"$\log(\alpha)$")
ax.set_ylabel("Standardized Coefficient")
plt.savefig('LASSO_stdized_coeff_paths_initial')

lass_mod_initial = Lasso(alpha=.01, tol=.3)
lass_mod_initial.fit(X_train_std, y_train)
LASSO_rmse_initial = np.sqrt(mse(y_test, lass_mod_initial.predict(X_test_std)))
# 1725592.21

lass_mod_initial.score(X_train_std, y_test)
# 0.22

fig,ax = plt.subplots(figsize=(10,10))
ax.plot([0,10000000], [0,10000000], "k-")
ax.scatter(lass_mod_initial.predict(X_test_std), y_test)
ax.set_xlabel('Predicted From LASSO')
ax.set_ylabel('Actual')
plt.savefig('LASSO_model_prediction')


lass_mod_initial.score(X_test_std, y_test)
lass_mod_initial.score(X_hold_out, y_hold_out)
# Absolute bullshit code to get initial plots and check for correlation...
# ex = X
# y = y
# set1 = ex.iloc[:, 0:5]
# set2 = ex.iloc[:, 6:10]
# set3 = ex.iloc[:, 11:15]
# set4 = ex.iloc[:, 16:20]
# set5 = ex.iloc[:, 21:25]
# set6 = ex.iloc[:, 26:30]
#
# set1['revenue'] = y
# set2['revenue'] = y
# set3['revenue'] = y
# set4['revenue'] = y
# set5['revenue'] = y
# set6['revenue'] = y
#
# scatter(set1)
# plt.show()
# scatter(set2)
# plt.show()
# scatter(set3)
# plt.show()
# scatter(set4)
# plt.show()
# scatter(set5)
# plt.show()
# scatter(set6)
# plt.show()


# Nothing stands out as particularly correlated with revenue. A few of the features seem correlated though:
# (p1 & p2), (p24 & p26), (p25 & p26), (p24 & p25)

# Check for heteroscedasticity:
#
#
# X = PCA(whiten=True, n_components=15).fit_transform(X)

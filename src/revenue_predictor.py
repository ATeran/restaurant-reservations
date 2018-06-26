# import statsmodels.api as sm
# import statsmodels.formula.api as smf
# from statsmodels.graphics import regressionplots as smg
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import plotly.plotly as py
# import plotly.graph_objs as go
# from pandas.plotting import scatter_matrix as scatter
# from sklearn.preprocessing import scale
from statsmodels.stats.outliers_influence import variance_inflation_factor


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

df = load_data('/Users/alanteran/galvanize/restaurant-revenue-prediction/data/train.csv')

def clean_names(df):
    temp_liz = [name.lower().replace(" ","_") for name in df.columns]
    df.columns = temp_liz
    return df

df = clean_names(df)

def convert_dates(df, date_column, to_drop):
    ''' Input -- Pandas series of strings: formatted as yyyy-mm-dd
        Output -- Pandas series: integer number of days since the start of the data.
    '''
    old_column = date_column
    date_column = pd.to_datetime(date_column)
    df['date_delta'] = (date_column - date_column.min())  / np.timedelta64(1,'D')
    df.drop(to_drop, axis=1, inplace=True)
    return df

df = convert_dates(df, df['open_date'], 'open_date')

def convert_to_dummies(df, list_of_categorical_columns):
    ''' Input -- Pandas DataFrame
        Input -- List of Pandas Series -- Categorical column to be dummied
        Output -- DataFrame with the categorical column replaced with dummy columns
    '''
    for categorical_column in list_of_categorical_columns:
        dummied_column = pd.get_dummies(df[categorical_column])
        df = pd.concat([df, dummied_column], axis=1)
        df.drop(categorical_column, axis=1, inplace=True)
    return df

df = convert_to_dummies(df, ['type', 'city_group'])
df = clean_names(df)

cities = df.pop('city')
y = df.pop('revenue')
x = df


def check_vifs(x):
	'''
	Input -- Pandas DataFrame, calculates the vifs for each variable in the DataFrame.
	Returns dictionary where key is column name and value is the vif for that column.
	Requires scipy.stats be imported as scs
	'''
	vifs = {}
	for index in range(x.shape[1]):
		vifs.append(round(variance_inflation_factor(x.values, index),2))
	return vifs

vifs(x)



# Try smaller sets of scatter matrices
def plot_scatters(key, value):
    from pandas.plotting import scatter_matrix as scatter
    scatter(value)
    plt.savefig('{}.png'.format(key))


def smaller_matrices(df, number_in_smaller):
    d = {}
    start = 0
    end = number_in_smaller + 3
    for num in range(1, number_in_smaller+1):
        d["set{}".format(num)] = df.iloc[:, start:end]
        start += number_in_smaller
        end += number_in_smaller
    # for key, value in d.items():
    #     pd.concat([value, df['revenue']], ignore_index=True)
    for key, value in d.items():
        value.add(df['revenue'])
        plot_scatters(key, value)

def

set1 = df.iloc[:, 3:8]
set2 = df.iloc[:, 9:14]
set3 = df.iloc[:, 15:20]
set4 = df.iloc[:, 21:26]
set5 = df.iloc[:, 27:32]
set6 = df.iloc[:, 33:38]
set7 = df.iloc[:, 39:43]
set1['revenue'] = df['revenue']
set2['revenue'] = df['revenue']
set3['revenue'] = df['revenue']
set4['revenue'] = df['revenue']
set5['revenue'] = df['revenue']
set6['revenue'] = df['revenue']
set7['revenue'] = df['revenue']
scatter(set1)
plt.show()
scatter(set2)
plt.show()
scatter(set3)
plt.show()
scatter(set4)
plt.show()
scatter(set5)
plt.show()
scatter(set6)
plt.show()
scatter(set7)
plt.show()

# Nothing stands out as particularly correlated with revenue. A few of the features seem correlated though:
# (p1 & p2), (p24 & p26), (p25 & p26), (p24 & p25)

# Check for heteroscedasticity:

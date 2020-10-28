# SCALER FUNCTIONS

# https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py
# https://towardsdatascience.com/scale-standardize-or-normalize-with-scikit-learn-6ccc7d176a02


# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def load_titanic():
    data = pd.read_csv(r"C:\Users\yakup\PycharmProjects\dsmlbc\datasets\train.csv")
    return data


# Define standard_scaler function
def standard_scaler_with_lambda(variable):

    df[[variable + '_std_scaled']] = df[[variable]].apply(lambda x: round((x - x.mean()) / x.std(), 3))


# Define robust_scaler function
def robust_scaler_with_lambda(variable):

    df[[variable + '_robust_scaled']] = df[[variable]].apply(lambda x: round((x - x.median()) / (x.quantile(0.75) - x.quantile(0.25)), 3))


# Define standard_scaler function
def standard_scaler(variable):
    variable_mean = variable.mean()
    variable_std = variable.std()
    z_value = (variable - variable_mean) / variable_std
    return round(z_value, 3)


# Define robust_scaler function
def robust_scaler(variable):
    variable_median = variable.median()
    quartile1 = variable.quantile(0.25)
    quartile3 = variable.quantile(0.75)
    interquartile_range = quartile3 - quartile1
    z_value = (variable - variable_median) / interquartile_range
    return round(z_value, 3)


# Let's run standard_scaler

df = load_titanic()
df.head()

df['Age'].describe().T

age_std_scaled = standard_scaler(df['Age'])
age_std_scaled.describe().T

df['Fare'].describe().T

fare_std_scaled = standard_scaler(df['Fare'])
fare_std_scaled.describe().T

df['Age'].hist()
plt.show()

age_std_scaled.hist()
plt.show()

df['Fare'].hist()
plt.show()

fare_std_scaled.hist()
plt.show()


# Let's run robust_scaler

df = load_titanic()
df.head()

df['Age'].describe().T

age_robust_scaled = robust_scaler(df['Age'])
age_robust_scaled.describe().T

df['Fare'].describe().T

fare_robust_scaled = robust_scaler(df['Fare'])
fare_robust_scaled.describe().T

df['Age'].hist()
plt.show()

age_robust_scaled.hist()
plt.show()

df['Fare'].hist()
plt.show()

fare_robust_scaled.hist()
plt.show()


# Compare scalers by visualizing them
age_std_scaled = standard_scaler(df['Age'])
age_robust_scaled = robust_scaler(df['Age'])
df.loc[:, 'Age_standard_scaled' ] = age_std_scaled
df.loc[:, 'Age_robust_scaled' ] = age_robust_scaled
df[['Age', 'Age_standard_scaled', 'Age_robust_scaled']].head()


fig = plt.figure()
ax = fig.add_subplot(111)
df['Age'].plot(kind='kde', ax=ax)
df['Age_standard_scaled'].plot(kind='kde', ax=ax, color='red')
df['Age_robust_scaled'].plot(kind='kde', ax=ax, color='green')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')
plt.show()


# Scaling by using apply/lambda

df = load_titanic()
df.head()

# Standard scaler
df['Age_std'] = df[['Age']].apply(lambda x: round((x - x.mean()) / x.std(), 3))
df['Age_std']

df['Fare_std'] = df[['Fare']].apply(lambda x: round((x - x.mean()) / x.std(), 3))
df['Fare_std']


# Robust scaler
df['Age_robust'] = df[['Age']].apply(lambda x: round((x - x.median()) / (x.quantile(0.75) - x.quantile(0.25)), 3))
df['Age_robust']

df['Fare_robust'] = df[['Fare']].apply(lambda x: round((x - x.median()) / (x.quantile(0.75) - x.quantile(0.25)), 3))
df['Fare_robust']


# Define standard_scaler function
def standard_scaler_with_lambda(variable):

    df[[variable + '_std_scaled']] = df[[variable]].apply(lambda x: round((x - x.mean()) / x.std(), 3))


# Define robust_scaler function
def robust_scaler_with_lambda(variable):

    df[[variable + '_robust_scaled']] = df[[variable]].apply(lambda x: round((x - x.median()) / (x.quantile(0.75) - x.quantile(0.25)), 3))

# Load the titanic dataset again
df = load_titanic()
df.head()

# Catch the numerical variablers to scale
num_cols = [col for col in df.columns if df[col].dtypes != "O"
            and col not in ["PassengerId", "Survived"]
            and len(df[col].value_counts()) > 10]
num_cols

# Describe numerical variables before scaling
df[num_cols].describe()

# Apply standard_scaler_with_lambda to numerical variables
for col in num_cols:
    standard_scaler_with_lambda(col)

# Describe numerical variables after scaling
df.describe()

# Apply robust_scaler_with_lambda to numerical variables
for col in num_cols:
    robust_scaler_with_lambda(col)

# Describe numerical variables after scaling and column names
df.describe()
df.columns

# Do not forget to drop the original columns after using these scaker functions with lambda

# SPECIAL FUNCTIONS FOR DATA PREPROCESSING AND FEATURE ENGINEERING

# Import libraries
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
import os
import sklearn
from sklearn.metrics import accuracy_score

# Make sum adjustments
pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# let's define a function to avoid coming here over and over again, when you need to read the data set
def load_titanic():
    data = pd.read_csv("C:/Users/yakup/OneDrive/Masaüstü/DSMLBC/datasets/train.csv")
    return data


df = load_titanic()


# Define a function to find the outlier thresholds
# According to data structure, business area different quartiles like 0.05, 0.95 can be selected, too.
# For linear models (LinearRegression, LogisticRegression, KNN), ANN, CNN, etc. we need to analyse outliers and missing values.
# However, for tree models we do not need to detect outliers, because they are robust to outlier and missing values.

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.25)
    quartile3 = dataframe[variable].quantile(0.75)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


outlier_thresholds(df, "Age")

low, up = outlier_thresholds(df, "Fare")


# Define a function to check if the variable has outlier or not
def has_outliers(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    if dataframe[(dataframe[variable] < low_limit) | (dataframe[variable] > up_limit)].any(axis=None):
        print(variable, "yes")


has_outliers(df, "Age")

# Let's find the numerical variables in the dataset
num_names = [col for col in df.columns if len(df[col].unique()) > 10
             and df[col].dtypes != 'O'
             and col not in "PassengerId"]

num_names

# Check the all numerical variables for outliers
for col in num_names:
    has_outliers(df, col)


# 1. Report variables with outliers
# 2. Create boxplot for the variables with outliers
# 2. User configurable boxplot feature
# 3. Return the names of the variables with outliers with a list

def has_outliers(dataframe, num_col_names, plot=False):
    variable_names = []

    for col in num_col_names:
        low_limit, up_limit = outlier_thresholds(dataframe, col)

        if dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].any(axis=None):
            number_of_outliers = dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].shape[0]

            print(col, ":", number_of_outliers)
            variable_names.append(col)

            if plot:
                sns.boxplot(x=dataframe[col])
                plt.show()

    return variable_names


has_outliers(df, num_names)

has_outliers(df, num_names, plot=True)


# Define a function to exclude the rows with outliers by using ~ --> get the rows without outliers
def remove_outliers(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    df_without_outliers = dataframe[~((dataframe[variable] < low_limit) | (dataframe[variable] > up_limit))]
    return df_without_outliers


df = remove_outliers(df, "Age")

df.shape

# Apply the function remove_outliers() for each column in the dataset
for col in num_names:
    new_df = remove_outliers(df, col)

new_df.shape

df.shape[0] - new_df.shape[0]


# Define a function to reassign up/low limits to the ones above/below up/low limits
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# Define a function to reassign up/low limits to the ones above/below up/low limits with apply/lambda
def replace_with_thresholds_with_lambda(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe[variable] = dataframe[variable].apply(
        lambda x: up_limit if x > up_limit else (low_limit if x < low_limit else x))


# Determine the thresholds
low, up = outlier_thresholds(df, "Age")

replace_with_thresholds(df, "Age")

df = load_titanic()

# Check if we have any outliers after assignment --> manually
# df[((df["Age"] < low) | (df["Age"] > up))]["Age"]

# Check if we have any outliers after assignment
has_outliers(df, num_names)

# Return the variables with outliers
var_names = has_outliers(df, num_names)

var_names

# Reassign up/low limits to the ones above/below up/low limits for each variables with outliers
for col in var_names:
    replace_with_thresholds(df, col)

has_outliers(df, num_names)

# Catching the observations with missing value
cols_with_na = [var for var in df.columns if df[var].isnull().sum() > 0]
cols_with_na


# Create a function to catch missing variables, count them and find ratio (in descending order)
def missing_values_table(dataframe):
    variables_with_na = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[variables_with_na].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[variables_with_na].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df)
    return variables_with_na


missing_values_table(df)

cols_with_na = missing_values_table(df)

cols_with_na


# Define a function to see the relationship of missing values with the dependent variable. If missing value, then --> 1
def missing_vs_target(dataframe, target, variable_with_na):
    temp_df = dataframe.copy()

    for variable in variable_with_na:
        temp_df[variable + '_NA_FLAG'] = np.where(temp_df[variable].isnull(), 1, 0)

    flags_na = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for variable in flags_na:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(variable)[target].mean()}), end="\n\n\n")


cols_with_na
missing_vs_target(df, "Survived", cols_with_na)
#df["Embarked_NA_FLAG"].value_counts()


# RECAP

df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)  # filling numerical variables
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)  # filling categorical variables
df["maas"].fillna(df.groupby("departman")["maas"].transform("mean"))  # filling numerical variables with respect to categorical breakdown
msno.heatmap(df)  # nullity correlation
missing_vs_target(df, "Survived", cols_with_na)  # the relationship of missing values with the dependent variable.
# Or we can leave missing values as they are thanks to advanced algorithms

from sklearn import preprocessing


# Define a function to label the variables with only 2 classes. First catch the variables with 2 classes and then label them.
def label_encoder(dataframe):
    labelencoder = preprocessing.LabelEncoder()

    label_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"
                  and len(dataframe[col].value_counts()) == 2]

    for col in label_cols:
        dataframe[col] = labelencoder.fit_transform(dataframe[col])
    return dataframe


df = label_encoder(df)

df["Sex"].head()


# Define a function to catch categorical variables and apply one hot encoding. You may also create dummy_na to trace missing values
def one_hot_encoder(dataframe, category_freq=10, nan_as_category=False):
    categorical_cols = [col for col in dataframe.columns if len(dataframe[col].value_counts()) < category_freq
                        and dataframe[col].dtypes == 'O']

    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, dummy_na=nan_as_category, drop_first=True)

    return dataframe


df = one_hot_encoder(df)
df.head()


# It makes no sense, if the category is too rare. It is an effort in vain. Unneccessary deployment effort!
# It makes no sense to create that as a feature.
# We need to create a new feature only if it is not rare!

# Define a function to catch categorical variables with frequency more than 0.10 and if there are more than 2 lables in the class
# We need to check the meaning of the difference, too. Nonparametric!
def super_cat_catch(dataframe, target):
    categorical_cols = [col for col in dataframe.columns if dataframe[col].dtypes == 'O']

    for col in categorical_cols:
        tmp_df = dataframe[col].value_counts() / len(dataframe)
        freq_labels = tmp_df[tmp_df > 0.10].index
        all_label = dataframe[col].value_counts().index
        selected_labels = [label for label in all_label if label in freq_labels]

        if len(selected_labels) > 2:
            min_label = dataframe.groupby(col)[target].mean().sort_values(ascending=False).first_valid_index()
            max_label = dataframe.groupby(col)[target].mean().sort_values(ascending=False).last_valid_index()
            dataframe[col + "_SUPER_MIN" + min_label] = np.where(dataframe[col] == min_label, 1, 0)
            dataframe[col + "_SUPER_MAX" + max_label] = np.where(dataframe[col] == max_label, 1, 0)

    return dataframe


df_new = super_cat_catch(df, "TARGET")

df_new.head()

# Catch categorical variables
cat_cols = [col for col in df.columns if df[col].dtypes == 'O']
cat_cols


# Define a function to count classes of categorical variables and calculate ratio of this class in the dataset
# Now we can see the frequencies of the classes and rare classes !
def cat_summary(data, cat_names):
    for var in cat_names:
        print(var, ":", len(data[var].value_counts()))
        print(pd.DataFrame({"COUNT": data[var].value_counts(),
                            "RATIO": data[var].value_counts() / len(data)}), end="\n\n\n")


cat_summary(df, cat_cols)


# Define a function to catch rare variables (if there is any rare class, then bring it),
#   print number of classes, count of classes, ratio of classes in the dataset and target mean by using;
# 1. Class Frequency
# 2. Class Ratio
# 3. Evaluate classes with respect to target variable
# 4. Determine rare ratio by yourself
def rare_analyser(dataframe, target, rare_perc):
    rare_columns = [col for col in df.columns if df[col].dtypes == 'O'
                    and (df[col].value_counts() / len(df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        print(var, ":", len(dataframe[var].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[var].value_counts(),
                            "RATIO": dataframe[var].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(var)[target].mean()}), end="\n\n\n")


rare_analyser(df, "TARGET", 0.001)


# Define a function to catch variables with rare classes. Rare ratio will be defined by the user.
def rare_encoder(dataframe, rare_perc):
    tempr_df = dataframe.copy()

    rare_columns = [col for col in tempr_df.columns if tempr_df[col].dtypes == 'O'
                    and (tempr_df[col].value_counts() / len(tempr_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = tempr_df[var].value_counts() / len(tempr_df)
        rare_labels = tmp[tmp < rare_perc].index
        tempr_df[var] = np.where(tempr_df[var].isin(rare_labels), 'Rare', tempr_df[var])

    return tempr_df


new_df = rare_encoder(df, 0.001)

rare_analyser(new_df, "TARGET", 0.001)


# TITANIC EDA & DP & FE

# In this project, Titanic dataset is scrutinized.

# Dataset: titanic dataset
# - https://www.kaggle.com/hesh97/titanicdataset-traincsv

# Steps and sub-steps to follow;

# 1. EXPLORATORY DATA ANALYSIS

# - GENERAL / GENERAL OVERVIEW / GENERAL PICTURE
#     - shape, info(), columns, index, describe(), isnull().any, isnull().sum()
#
# - CATEGORICAL VARIABLE ANALYSIS
#     - Catching categorical variables
#     - Number of categorical variables
#     - Classes
#     - Frequencies
#     - Basic visualization --> barplot, countplot
#
# - NUMERICAL VARIABLE ANALYSIS
#     - Catching numerical variables
#     - describe with quantiles to see whether there are extraordinary values or not
#     - Basic visualization --> histogram, boxplot and density
#
# - TARGET ANALYSIS
#     - Target analysis according to categorical variables --> target_summary_with_cats()
#     - target analysis according to numerical variables --> target_summary_with_nums()
#
# - ANALYSIS OF NUMERICAL VARIABLES IN COMPARISON WITH EACH OTHER
#     - scatterplot
#     - lmplot
#     - correlation

# 2. DATA PREPARATION & FEATURE ENGINEERING

#   - OUTLIER ANALYSIS (boxplot, catching, deleting, calculating threshold, reassignment with thresholds)
#   - MISSING VALUES ANALYSIS (catching, deleting, filling with mean/median/mode)
#   - RARE ENCODING (catching rare classes on the dataset remove them or gather them in a separate class and name this as rare class
#                   and analyse the effect of rare classes)
#   - LABEL ENCODING (0-1 labeling i.e. Gender --> Female, Male)
#   - ONE-HOT ENCODING (pd.get_dummies(drop_first=True)
#   - SUPER-CATEGORY CATCHING (catching super classes that has great effect on target, do rare encoding first!)
#   - FEATURE ENGINEERING (Flag/Bool, Letter/Word count, Numerical to categorical, Interactions)
#   - STANDARDIZATION (StandardScaler, RobustScaler, MinMaxScaler, log)


# Import dependencies
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno

pd.set_option('display.max_columns', None)


## 1. EXPLORATORY DATA ANALYSIS

# GENERAL OVERVIEW

# Define a function to avoid coming here over and over again, when you need to read the data set
def load_titanic():
    data = pd.read_csv(r"C:\Users\yakup\PycharmProjects\dsmlbc\datasets\train.csv")
    return data


df = load_titanic()
df.head()
df.tail()
df.shape
df.info()
df.columns
df.index
df.describe().T
df.isnull().values.any()
df.isnull().sum()

# CATEGORICAL VARIABLE ANALYSIS

# Number of categorical variables
# Classes
# Frequencies
# Basic visualization

df.head()

# Categorical Classes --> Names?
df["Survived"].unique()

# Number of categorical classes
len(df["Survived"].unique())
df["Survived"].nunique()

# Classes and frequencies?
df["Survived"].value_counts()

# Let's catch categorical variables
cat_cols = [col for col in df.columns if df[col].dtype == "O"]
cat_cols

print('Number of Categorical Variables : ', len(cat_cols))

# We want to catch secret categorical variables (They seem to be numerical, but in fact they are categorical
# because they do not have many unique values, but a few and repeated) and
# exclude the ones with many classes (These are in fact not categorical).
# Let's bring the ones with less than 10 unique values. You may choose another number, too. For this dataset we chose '10'

more_cat_cols = [col for col in df.columns if len(df[col].unique()) < 10]
more_cat_cols

print('Number of Actual Categorical Variables : ', len(more_cat_cols))

# numbers of unique classes for each cat_cols
df[more_cat_cols].nunique()

# Basic visualization rules
# categorical variables --> barplot, countplot
# numerical variables --> histogram, boxplot and density

# countplot
sns.countplot(x = "Sex", data = df)
plt.show()

# See the unique values in 'Sex'
df["Sex"].value_counts()

# Ratio
100 * df["Sex"].value_counts() / len(df)


# Define a function that catches categorical variables, prints the distribution and ratio of unique values and finally creates a countplot
def cats_summary1(data):
    cat_names = [col for col in data.columns if len(data[col].unique()) < 10]
    for col in cat_names:
        print(pd.DataFrame({col: data[col].value_counts(),
                            "Ratio": 100 * data[col].value_counts() / len(data)}), end="\n\n\n")
        sns.countplot(x=col, data=data)
        plt.show()


cats_summary1(df)


# Define a function takes dataframe, categorical columns and if required number of classes (default value = 10).
# Then, it prints the distribution and ratio of unique values for the variables, which have less than 10 (number of classes) unique values.
# Afterwards, it reports categorical variables it described, how many numerical variables, but seem categorical we have and finally report these variables.
def cats_summary2(data, categorical_cols, number_of_classes=10):
    var_count = 0  # reporting how many categorical variables are there?
    vars_more_classes = []  # save the variables that have classes more than a number that we determined

    for var in data:
        if var in categorical_cols:
            if len(list(data[var].unique())) <= number_of_classes:  # select according to number of classes
                print(pd.DataFrame({var: data[var].value_counts(),
                                    "Ratio": 100 * data[var].value_counts() / len(data)}), end="\n\n\n")
                var_count += 1
            else:
                vars_more_classes.append(data[var].name)
    print('%d categorical variables have been described' % var_count, end="\n\n")
    print('There are', len(vars_more_classes), "variables have more than", number_of_classes, "classes", end="\n\n")
    print('Variable names have more than %d classes:' % number_of_classes, end="\n\n")
    print(vars_more_classes)


cats_summary2(df, cat_cols)


# NUMERICAL VARIABLES ANALYSIS

# Remember the dataset
df = load_titanic()
df.describe().T

# See the distribution of variables according to quantiles
df.describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99]).T

# boxplot for Age
sns.boxplot(x=df["Age"])
plt.show()

# boxplot for Fare
sns.boxplot(x=df["Fare"])
plt.show()

# Number of numerical variables
num_cols = [col for col in df.columns if df[col].dtypes != "O"]
num_cols

print('NUMBER OF NUMERICAL VARIABLES: ', len(num_cols))

# Drop 'PassengerId' column, because it is not important for us
df.drop("PassengerId", axis=1).columns

# We know, that 'Survived' is our target variable and 'PassengerId' is not a numerical variable. Let's exclude them.
num_cols = [col for col in df.columns if df[col].dtypes != "O"
            and col not in ["PassengerId", "Survived"]]
num_cols

# Basic visualization rules
# categorical variables --> barplot, countplot
# numerical variables --> histogram, boxplot and density

# Let's see the frequencies of values and distribution of the dataset by using histogram and boxplot

# see the frequencies of values
df["Age"].hist()
plt.show()

df["Fare"].hist()
plt.show()

# distribution of the variable
sns.boxplot(x=df["Age"])
plt.show()

sns.boxplot(x=df["Fare"])
plt.show()


# Define a function that plots histogram of numerical variables and prints the number of numerical variables if plotted.
def hist_for_nums(data, numeric_cols):
    col_counter = 0

    for col in numeric_cols:
        data[col].hist()
        plt.xlabel(col)
        plt.title(col)
        plt.show()

        col_counter += 1

    print(col_counter, "variables have been plotted")

num_cols
hist_for_nums(df, num_cols)


# TARGET ANALYSIS

df = load_titanic()
df.head()

df["Survived"].value_counts()

# Target analysis according to categorical variables
df.groupby("Sex")["Survived"].mean()
df.groupby("Pclass")["Survived"].mean()
df.groupby("Embarked")["Survived"].mean()

more_cat_cols = [col for col in df.columns if len(df[col].unique()) < 10]
print('Number of Categorical Variables: ', len(cat_cols))
print(more_cat_cols)


# Define a function that catches categorical variables with number of classes less than 10 and
# exclude target variable and analyse target variable with respect to categories
def target_summary_with_cats(data, target):
    cats_names = [col for col in data.columns if len(data[col].unique()) < 10 and col not in target]

    for var in cats_names:
        print(pd.DataFrame({"TARGET_MEAN": data.groupby(var)[target].mean()}), end="\n\n\n")


target_summary_with_cats(df, "Survived")

# Target analysis according to numerical variables

# See the average ages for survived and not survived
df.groupby("Survived").agg({"Age": np.mean})
df.groupby("Survived").agg({"Fare": np.mean})


# Define a function that catches numerical variables exclude target variable ang group survived by numerical variables
def target_summary_with_nums(data, target):
    num_names = [col for col in data.columns if len(data[col].unique()) > 5
                 and df[col].dtypes != 'O'
                 and col not in target
                 and col not in "PassengerId"]

    for var in num_names:
        print(df.groupby(target).agg({var: np.mean}), end="\n\n\n")


target_summary_with_nums(df, "Survived")


# ANALYSIS OF NUMERICAL VARIABLES IN COMPARISON WITH EACH OTHER

df = load_titanic()
df.head()

# scatterplot
sns.scatterplot(x="Survived", y="Fare", data=df)
plt.show()

# lmplot
sns.lmplot(x="Pclass", y="Fare", data=df)
plt.show()

# Correlation table by using heatmap
plt.subplots(figsize=(15,12))
correlation_matrix = df.corr().round(2)
sns.heatmap(data=correlation_matrix, cmap='coolwarm', annot=True, square=True)
plt.show()


df = load_titanic()

# Special function to summarize this notebook

def analyse(data, target):
    cat_cols = [col for col in df.columns if df[col].dtype == "O"]
    num_cols = [col for col in df.columns if df[col].dtypes != 'O' and col not in target]

    print(
        '********************************Let\'s see categorical variables and take a quick look at them!********************************\n')
    cats_summary1(data)
    print(
        '********************************Let\'s see numerical variables and take a quick look at them with histograms!********************************\n')
    hist_for_nums(data, num_cols)
    print(
        '********************************Let\'s see target summary with categorical variables!********************************\n')
    target_summary_with_cats(data, target)
    print(
        '********************************Let\'s see target summary with numerical variables!********************************\n')
    target_summary_with_nums(data, target)


analyse(df, "Survived")


# 2. DATA PREPARATION & FEATURE ENGINEERING

# OUTLIER ANALYSIS
#     - Catching outliers
#     - Removing outliers
#     - Reassignment with thresholds

# Catching outliers

df = load_titanic()
df.head()

# Quantile method to catch outliers
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


# Catch numerical variables to send them in has_outliers() function
num_names = [col for col in df.columns if len(df[col].unique()) > 10 and df[col].dtypes != 'O' and col not in "PassengerId"]
num_names


# Define a function to report variables with outliers and number of outliers for this variable,
#           create boxplot for the variables with outliers and
#           return the names of the variables with outliers with a list
def has_outliers(dataframe, num_col_names, plot=False):
    variable_names = []

    for col in num_col_names:
        low_limit, up_limit = outlier_thresholds(dataframe, col)

        if dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].any(axis=None):
            number_of_outliers = dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].shape[0]

            print(col, ":", number_of_outliers)
            variable_names.append(col)

            if plot:
                sns.boxplot(x = dataframe[col])
                plt.show()

    return variable_names


has_outliers(df, num_names)
# has_outliers(df, num_names, plot=True)

# # Removing Outliers
#
# # Define a function to exclude the rows with outliers by using ~ --> get the rows without outliers
# def remove_outliers(dataframe, variable):
#     low_limit, up_limit = outlier_thresholds(dataframe, variable)
#     df_without_outliers = dataframe[~((dataframe[variable] < low_limit) | (dataframe[variable] > up_limit))]
#     return df_without_outliers
#
#
# # Apply the function remove_outliers() for each column in the dataset
# for col in num_names:
#     new_df = remove_outliers(df, col)
#
# new_df.shape
#
# # How many rows did we removed?
# df.shape[0] - new_df.shape[0]

# Reassignment with thresholds --> Instead of removing outliers, threshold values can be assigned to outliers


# # Define a function to reassign up/low limits to the ones above/below up/low limits
# def replace_with_thresholds(dataframe, variable):
#     low_limit, up_limit = outlier_thresholds(dataframe, variable)
#     dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
#     dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# Define a function to reassign up/low limits to the ones above/below up/low limits with apply/lambda
def replace_with_thresholds_with_lambda(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe[variable] = dataframe[variable].apply(lambda x: up_limit if x > up_limit else (low_limit if x < low_limit else x))


# Return the variables with outliers
var_names = has_outliers(df, num_names)
var_names

# Reassign up/low limits to the ones above/below up/low limits for each variables with outliers
for col in var_names:
    replace_with_thresholds_with_lambda(df, col)

# Check again, if we have any outliers after assignment
has_outliers(df, num_names)


# MISSING VALUES ANALYSIS
#     - Catch
#     - Look for randomness (for example average credit card spendings, presence of credit card )
#     - Solve the problem (Delete, Basic assign, Assignment by breakdown, Assignment by predictions)

# Catching Missing Values

df.head()

# Advanced Analysis for Missing Variables

# Visualize missing variables
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()

# Missing values overall view
msno.bar(df)
plt.show()

# Now, we can see the relationship between missing values
msno.matrix(df)
plt.show()

# Nullity correlation visualization
msno.heatmap(df)
plt.show()

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


# # Define a function to see the relationship of missing values with the dependent variable. If missing value, then --> 1
# def missing_vs_target(dataframe, target, variable_with_na):
#     temp_df = dataframe.copy()
#
#     for variable in variable_with_na:
#         temp_df[variable + '_NA_FLAG'] = np.where(temp_df[variable].isnull(), 1, 0)
#
#     flags_na = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
#
#     for variable in flags_na:
#         print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(variable)[target].mean()}), end="\n\n\n")
#
#
# cols_with_na
# missing_vs_target(df, "Survived", cols_with_na)

# There are many approaches to solve missing variables problem
#     - 1: do not touch! if you use tree models, if there are a lot of aggregation, singularization
#     - 2: delete / drop na values
#     - 3: basic assignment methods (mean, median, mode)
#     - 4: assignment values by categorical breakdown


# For this dataset I choose the third option , namely basic assignment by using median for numerical variables and
# mode for categorical variables. Because I prefer not to distort the mean and teh general structure of the dataset.

# Fill numerical missing values with median
df = df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)
# Fill real categorical(categorical classes<10) missing values with  mode
df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)

# Check again for missing values
missing_values_table(df) # We see, that there are still missing values by 'Cabin' --> Ratio: 77.1%

df.info()
df.loc[:, 'Cabin'].value_counts()
df.loc[:, 'Cabin'].nunique()
# As we see there are 687 missing variables and 147 unique classes, so I decided to drop this variable.

# Drop 'Cabin' variable
df.drop('Cabin', axis = 1, inplace=True)

# Check again for missing values
missing_values_table(df)

df.shape

# Investigation of the relationship of missing values with the dependent variable

df.head()


# RARE ENCODING

# # Load the dataset
# df = load_titanic()
# df.head()

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


rare_analyser(df, "Survived", 0.05)


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


new_df = rare_encoder(df, 0.05)
new_df

rare_analyser(new_df, "Survived", 0.05)


# FEATURE ENGINEERING

df.head()

# Create NEW_IS_ALONE that shows if the person is alone or not.
df.loc[((df['SibSp'] + df['Parch']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SibSp'] + df['Parch']) == 0), "NEW_IS_ALONE"] = "YES"
df.head()

# LETTER COUNT
df["NEW_NAME_COUNT"] = df["Name"].str.len()
df.head()

# WORD COUNT
df["NEW_NAME_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" ")))
df.head()

# Catch the names with Dr.
df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
df["NEW_NAME_DR"].head()
df["NEW_NAME_DR"].mean()

# Group by NEW_NAME_DR and see the mean
df.groupby("NEW_NAME_DR").agg({"Survived": "mean"})
# Number of names with Dr. Watch out for the frequencies!!!
df["NEW_NAME_DR"].value_counts()

# Create a new feature 'NEW_TITLE'. After blank get ones before dot. --> ' ([A-Za-z]+)\.'
df['NEW_TITLE'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
df.head()

# Group by NEW_TITLE and see the mean of Age
df.groupby("NEW_TITLE").agg({"Age": "mean"})

# Group NEW_TITLE, Survived and Age by NEW_TITLE see the mean and count
df[["NEW_TITLE", "Survived", "Age"]].groupby(["NEW_TITLE"]).agg({"Survived": "mean", "Age": ["count", "mean"]})

# NUMERIC TO CATEGORICAL
df.loc[(df['Age'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['Age'] >= 18) & (df['Age'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['Age'] >= 56), 'NEW_AGE_CAT'] = 'senior'
df.head()

# INTERACTIONS
df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1
df.head()

df.loc[(df['Sex'] == 1) & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['Sex'] == 1) & ((df['Age'] > 21) & (df['Age']) < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['Sex'] == 1) & (df['Age'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['Sex'] == 0) & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['Sex'] == 0) & ((df['Age'] > 21) & (df['Age']) < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['Sex'] == 0) & (df['Age'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'
df.head()


# df = one_hot_encoder(df)
# df.head()


# LABEL ENCODING

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
df.head()


# ONE HOT ENCODING

# Define a function to catch categorical variables and apply one hot encoding. You may also create dummy_na to trace missing values
def one_hot_encoder(dataframe, category_freq=10, nan_as_category=False):
    categorical_cols = [col for col in dataframe.columns if len(dataframe[col].value_counts()) < category_freq
                        and dataframe[col].dtypes == 'O']

    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, dummy_na=nan_as_category, drop_first=True)

    return dataframe


df = one_hot_encoder(df)
df.head()


# STANDARDIZATION

# RobustScaler: Subtract median and divide by IQR (Inter Quartile Range). X_std = (X - median) / IQR
# Suggested for numerical variables. It is not affected from outliers. Robust to outliers!
from sklearn.preprocessing import RobustScaler
transformer = RobustScaler()
df["Age"] = transformer.fit_transform(df[["Age"]])
df["Age"].describe().T

df["Fare"] = transformer.fit_transform(df[["Age"]])
df["Fare"].describe().T


# During data preprocessing and feature engineering we implemented the following step:
#     - outlier analysis
#     - missing values analysis
#     - rare encoding
#     - label encoding
#     - one hot encoding
#     - feature engineering
#     - standardization

# See teh new shape of the dataframe
df.shape

# Show the first rows again for the last time
df.head()

# Now, we are satisfied with the dataframe. We can save it.
df.to_csv('titanic_after_preprocessing.csv')

# We can also use excel.
df.to_excel('titanic_after_preprocessing.xlsx')

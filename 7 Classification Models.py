# CLASSIFICATION MODELS


# DIABETES PREDICTION MODEL

# Import dependencies
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.simplefilter(action="ignore")

df = pd.read_csv(r"C:\Users\yakup\PycharmProjects\dsmlbc\datasets\diabetes.csv")

# EDA

df.head()
df.shape
df["Outcome"].value_counts() * 100 / len(df)


# SORU: class ratio 1: 0.05, 0: 0.95
# Böyle bir durumda ne yaparsınız?
# 1. Oranlar böyle tamam ama frekanslar ne?
# 2. Hepsine 1 desem zaten 95 başarılıyım. Neden model kuralım?
# Dengesiz veri problemini araştırınız.

df.describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
sns.countplot(x='Outcome', data=df)
plt.show()

df["Age"].hist(edgecolor="black")
plt.show()

df.groupby("Outcome").agg({"Pregnancies": "mean"})
df.corr()

# -1,1
# 0.7, 1
# -0.7,-1

# Data Preprocessing

df.isnull().sum()


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.10)
    quartile3 = dataframe[variable].quantile(0.90)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def has_outliers(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    if dataframe[(dataframe[variable] < low_limit) | (dataframe[variable] > up_limit)].any(axis=None):
        print(variable, "yes")


outlier_thresholds(df, "BloodPressure")


has_outliers(df, "Age")

for col in df.columns:
    has_outliers(df, col)


# FEATURE ENGINEERING

# Modeling

# Logistic Regression

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

X.head()
y.head()

log_model = LogisticRegression().fit(X, y)
log_model.intercept_
log_model.coef_

log_model.predict(X)[0:10]
y[0:10]

log_model.predict_proba(X)[0:10]
y_pred = log_model.predict(X)
accuracy_score(y, y_pred)

cross_val_score(log_model, X, y, cv=10).mean()

print(classification_report(y, y_pred))



logit_roc_auc = roc_auc_score(y, log_model.predict(X))
fpr, tpr, thresholds = roc_curve(y, log_model.predict_proba(X)[:, 1])
plt.figure()
plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
#plt.savefig('Log_ROC')
plt.show()

# RF

rf_model = RandomForestClassifier(random_state=12345).fit(X, y)

cross_val_score(rf_model, X, y, cv=10).mean()

rf_params = {"n_estimators": [200, 500],
             "max_features": [5, 7],
             "min_samples_split": [5, 10],
             "max_depth": [5, None]}

rf_model = RandomForestClassifier(random_state=12345)

rf_cv_model = GridSearchCV(rf_model, rf_params, cv=10, n_jobs=-1, verbose=2).fit(X, y)
rf_cv_model.best_params_

rf_tuned = RandomForestClassifier(**rf_cv_model.best_params_)
cross_val_score(rf_tuned, X, y, cv=10).mean()


# LightGBM


lgbm = LGBMClassifier(random_state=12345)
cross_val_score(lgbm, X, y, cv=10).mean()

# model tuning
lgbm_params = {"learning_rate": [0.01, 0.1, 0.5],
               "n_estimators": [500, 1000, 1500],
               "max_depth": [3, 5, 8]}

lgbm_cv_model = GridSearchCV(lgbm, lgbm_params, cv=5, n_jobs=-1, verbose=2).fit(X, y)
lgbm_tuned = LGBMClassifier(**lgbm_cv_model.best_params_).fit(X, y)

cross_val_score(lgbm_tuned, X, y, cv=10).mean()

feature_imp = pd.Series(lgbm_tuned.feature_importances_, index=X.columns).sort_values(ascending=False)
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Değişken Önem Skorları')
plt.ylabel('Değişkenler')
plt.title("Değişken Önem Düzeyleri")
plt.show()

# eda
# data prep
# feature eng.
# model
# tahmin
# model tuning
# final model
# feature importance

# CV for all models

models = [('LR', LogisticRegression()),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier()),
          ('RF', RandomForestClassifier()),
          ('SVM', SVC(gamma='auto')),
          ('XGB', GradientBoostingClassifier()),
          ("LightGBM", LGBMClassifier())]

# evaluate each model in turn
results = []
names = []

for name, model in models:
    kfold = KFold(n_splits=10, random_state=123456)
    cv_results = cross_val_score(model, X, y, cv=10, scoring="accuracy")
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# boxplot algorithm comparison
fig = plt.figure(figsize=(15, 10))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# 2. OPTION: HOLDOUT + CV

# Validation scires for all models

models = [('LR', LogisticRegression()),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier()),
          ('RF', RandomForestClassifier()),
          ('SVM', SVC(gamma='auto')),
          ('XGB', GradientBoostingClassifier()),
          ("LightGBM", LGBMClassifier())]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=46)

# evaluate each model in turn
results = []
names = []

for name, model in models:
    kfold = KFold(n_splits=10, random_state=123456)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring="accuracy")
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# boxplot algorithm comparison
fig = plt.figure(figsize=(15, 10))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    msg = "%s: (%f)" % (name, acc)
    print(msg)


df.head()

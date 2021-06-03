# data analysis and wrangling
import pandas as pd
import numpy as np

# visualization
import seaborn as sns
import matplotlib.pyplot as plt


# machine learning
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report, plot_roc_curve
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



def divide_to_groups(data, var, bins, group_names):
    num_groups = bins
    group = group_names
    data[var] = pd.cut(data[var], num_groups, labels=group)


# the following def- divide the countries to mainlands:
def native(country):
    if country in ['United-States', 'Canada']:
        return 'North_America'
    if country in ['Cuba', 'Jamaica', 'Mexico', 'Guatemala', 'El-Salvador',
                   'Dominican-Republic', 'Haiti', 'Honduras', 'Nicaragua',
                   'Outlying-US(Guam-USVI-etc)', 'Puerto-Rico', 'Trinadad&Tobago']:
        return 'Center_America'
    elif country in ['Columbia', 'Ecuador', 'Peru']:
        return 'South_America'
    elif country in ['Cambodia', 'China', 'Hong', 'India', 'Iran', 'Japan', 'Laos',
                     'Philippines', 'Taiwan', 'Thailand', 'Vietnam']:
        return 'Asia'
    elif country in ['England', 'France', 'Germany', 'Greece', 'Holand-Netherlands',
                     'Hungary', 'Ireland', 'Italy', 'Poland', 'Portugal', 'Scotland', 'Yugoslavia']:
        return 'Europe'
    elif country in ['South', '?']:
        return 'Other'
    else:
        return country


def basic_education(x):
    if x in ['1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th']:
        return 'basic_education'
    else:
        return x


def encode(data, var, bins):
    for i in range(len(bins)-1):
        data.loc[(data[var] >= bins[i]) & (data[var] < bins[i+1]), var] = i
    data.loc[data[var] > bins[len(bins)-1], var] = len(bins)-1

# read the data:
Data = pd.read_csv('C:/Users/karin/Documents/Projects/interview_projects/Tapas/Data Scientist Home Assignment/'
                   'income_pred.csv')

# ************************* EDA *************************
Data.head()
Data.tail()
Nsampelse, Nfeatures = Data.shape  # check the size of the data set
data_columns = Data.columns
# check the information at each category - 1. check if there is a missing values
# (with the count) and the type:
Data.info()
Data.isnull().any()  # secound check for missing values (nan's)
categorical_data_info = Data.describe(include='object').transpose()
numerical_data_info = Data.describe().transpose()

# check the categorical unique values to see if there is any missing values with others format (as "--", "?" "-")
print('workclass' + str(Data['workclass'].unique()))
print('occupation:' + str(Data['occupation:'].unique()))
print('education:' + str(Data['education:'].unique()))
print('marital-status:' + str(Data['marital-status:'].unique()))
print('relationship:' + str(Data['relationship:'].unique()))
print('native-country:' + str(Data['native-country:'].unique()))
print('race:' + str(Data['race:'].unique()))

# Conclusions :
# 1. there is numerical and categorical features
# 2. few features with large options (large unique variables )
# 3. there are missing values, represented by "?"


# ************************* Data preparation *************************

# Replace the categorical class (the feature that we need to predict) to numerical class
# (for binary binary classification on 'class'): "<=50K" = 0, ">50K" =1
cleanup_nums = {"class":     {"<=50K": 0, ">50K": 1}}
Data = Data.replace(cleanup_nums)
Data.head()

# the following Data preparation manipulations created with the aim of reducing the high number of unique categorize
# in each categorical feature (reducing the "noise"):

# divide the countries to mainlands:
Data['native-country:'] = Data['native-country:'].apply(native)
Data['native-country:'].value_counts()

# replace the "?" in the dataset the represent the missing data with nan's:
Data.replace("?", np.nan, inplace=True)
# count the nan's, check whether the missing data is categorical or numerical,
# If the missing data is numerical and the %of missing value is large- can be fill with values (as feature column average value)
# for large categorical missing data - can be consider to use as another category/develop model to predict missing values
count_missing_data = Data.isna().sum()
# check the % of the missing data:
missing_data = (count_missing_data/Nsampelse)*100
# In our case all the missing data are categorical and with ~5.5% from the data, so can be ignored

# divide the feature  'age' feature to 3 sub-age ranges:
age_bands = pd.cut(Data['age'], 3)
# convert the age feature to groups based on the bands:
encode(Data, 'age', [17, 35, 55, 90])
# divide the feature education-num into three groups:
pd.cut(Data['education-num:'], 3)
encode(Data, 'education-num:', [0, 6, 11, 16])
# divide the feature 'marital-status:' into two groups: 1. Married, 2. Not_married:
Data['marital-status:'] = Data['marital-status:'].apply\
    (lambda x: 'married' if (x.startswith('Married', 0)) else 'Not_married').astype(object)
# divide the 'hours-per-week' information into four groups:
encode(Data, 'hours-per-week:', [0, 25, 50, 70, 100])
# check the education feature using plot:
sns.factorplot(x="education:", y="class", data=Data, kind="bar", size=5)
plt.xticks(rotation=50)
# we can see that all the (1th-12th) can be grouped into one iption- we call it 'basic education' (12 years of study):
Data['education:'] = Data['education:'].apply(basic_education)
Data['education:'].value_counts()

# Drop the columns that will not contribute to the model:
Data.drop(['ID'], axis='columns', inplace=True)
# save the labels ('class' feature) as label vector and delete it from feature matrix (Data):
labels = pd.DataFrame(Data['class'])
Data.drop(['class'], axis='columns', inplace=True)

# handle with the categorical_features:
categorical_features = Data.select_dtypes(include=['object']).axes
for col in categorical_features[1]:
    print(col, Data[col].nunique())

# handle with the missing values (nan's)
for col in ['workclass', 'occupation:']:
    Data[col].fillna(Data[col].mode()[0], inplace=True)

# double checke that there is no more any missing data:
Data.isnull().sum()

# replace the categorical feature with numbers:
for feature in categorical_features[1]:
        le = preprocessing.LabelEncoder()
        Data[feature] = le.fit_transform(Data[feature])

# ************************* Modeling *************************

# divide to train and test:
x_train, x_test, y_train, y_test = train_test_split(Data, labels, test_size=0.2)

# check the ration between class in the label vector (y_train):
class_amount = y_train.value_counts()
Ratio = class_amount[0]/class_amount[1]  # the clsss 0 (income <=50K) is 3 times larger compering to class 1 (income <50K)

# standarize features:
scaler = StandardScaler().fit(x_train)
Xtrain_Scaled = scaler.transform(x_train)
Xtest_Scaled = scaler.transform(x_test)

# check if the scaled data have mean = 0 and variance = 1:
check_mean = Xtrain_Scaled.mean(axis=0)
checkSD = Xtrain_Scaled.std(axis=0)
# specify model
model = LogisticRegression(max_iter=200)
model.fit(Xtrain_Scaled, y_train.values.ravel())

# LogisticRegression model - using PCA information:

pca = PCA()
pca.fit_transform(Xtrain_Scaled)
pca.explained_variance_ratio_
# check the PC1 (First Principal Component) which is the first row :
The_principal_comp = (abs(pca.components_)) # con:the less theimportant features are [feature 2: fnlwgt and, feature 13:native-country]
# delete the less importance features:
Xtrain_Scaled_new = np.delete(Xtrain_Scaled, [2, 13], 1)
Xtest_Scaled_new = np.delete(Xtest_Scaled, [2, 13], 1)

# fit the model & predict:
model.fit(Xtrain_Scaled_new, y_train)
Ypredict_train = model.predict(Xtrain_Scaled_new)
Ypredict_test = model.predict(Xtest_Scaled_new)

# ************************* Evaluating performance *************************

# a. statistics:
print(classification_report(Ypredict_train, y_train))
# b. check the confusion_matrix:
cm = confusion_matrix(Ypredict_test, y_test)
plot_confusion_matrix(model, Xtest_Scaled_new, y_test, cmap=plt.cm.Blues)
# check the ROC curve:
plot_roc_curve(model, Xtest_Scaled_new, y_test)
plt.show()

# ************************* Conclusions *************************

# 1. I've used LogisticRegression model with Principal Component Analysis (PCA) for build a predictive model
# for the income level
# 2. I've found  maximum accuracy of 0.82 and F1 score of 0.84.
# 3. The AUC was found to be 0.87

# There are more option to optimize the model  and the results,
# as try to check different methods for the optimal features (g=feature selection methods)
# or build different model as boosting models (XGBoost)

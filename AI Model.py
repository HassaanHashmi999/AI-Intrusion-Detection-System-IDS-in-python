import numpy as np
import time
import optuna
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
import warnings
import joblib
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree  import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import BernoulliNB
from lightgbm import LGBMClassifier
from sklearn.feature_selection import RFE
import itertools
from xgboost import XGBClassifier
from tabulate import tabulate



train = pd.read_csv("Train_data.csv")
test = pd.read_csv("Test_data.csv")
#checking the head of the dataset
print(train.head(4))
#shape of the dataset
print("Training data has {} rows & {} columns".format(train.shape[0],train.shape[1]))

print(test.head(4))
train.info()
train.describe(include='object')

total = train.shape[0]
missing_columns = [col for col in train.columns if train[col].isnull().sum() > 0]
for col in missing_columns:
    null_count = train[col].isnull().sum()
    per = (null_count/total) * 100
    print(f"{col}: {null_count} ({round(per, 3)}%)")


#Checking for class if its an anomaly or not in the dataset
print(train['class'])
sns.countplot(x=train['class'])
#ploting the graph for the class
plt.show()
#count of anamoly and normal
print('Class distribution Training set:')
print(train['class'].value_counts())


#Label encoding to all object-type columns in the dataset
def le(df):
    for col in df.columns:
        if df[col].dtype == 'object':
                label_encoder = LabelEncoder()
                df[col] = label_encoder.fit_transform(df[col])

le(train)
le(test)

train.head()

X_train = train.drop(['class'], axis=1)
Y_train = train['class']


#rfe is recursive feature elimination used to select the best features of the dataset
rfc = RandomForestClassifier()

rfe = RFE(rfc, n_features_to_select=10)
rfe = rfe.fit(X_train, Y_train)

feature_map = [(i, v) for i, v in itertools.zip_longest(rfe.get_support(), X_train.columns)]
selected_features = [v for i, v in feature_map if i==True]
print(selected_features)

#selecting the best features from the dataset
X_train = X_train[selected_features]


#Standardization of the dataset
scale = StandardScaler()
X_train = scale.fit_transform(X_train)
test = scale.fit_transform(test)
#splitting the dataset into train and test 70% and 30% respectively
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, train_size=0.70, random_state=2)
#checking the shape of the dataset again after splitting
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

#Decision Tree
clfd = DecisionTreeClassifier(criterion ="entropy", max_depth = 4)
start_time = time.time()
clfd.fit(x_train, y_train.values.ravel())
end_time = time.time()
print("Training time: ", end_time-start_time)
start_time = time.time()
y_test_pred = clfd.predict(x_train)
end_time = time.time()
print("Testing time: ", end_time-start_time)


#Accuracy of the model for 30 trials
def objective(trial):
    dt_max_depth = trial.suggest_int('dt_max_depth', 2, 32, log=False)
    dt_max_features = trial.suggest_int('dt_max_features', 2, 10, log=False)
    classifier_obj = DecisionTreeClassifier(max_features = dt_max_features, max_depth = dt_max_depth)
    classifier_obj.fit(x_train, y_train)
    accuracy = classifier_obj.score(x_test, y_test)
    return accuracy
study_dt = optuna.create_study(direction='maximize')
study_dt.optimize(objective, n_trials=30)#trails is the number of times the model is run
print(study_dt.best_trial)



#print("Accuracy on training set: ", clfd.score(x_train, y_train))
#print("Accuracy on test set: ", clfd.score(x_test, y_test))


dt = DecisionTreeClassifier(max_features = study_dt.best_trial.params['dt_max_features'], max_depth = study_dt.best_trial.params['dt_max_depth'])
dt.fit(x_train, y_train)

dt_train, dt_test = dt.score(x_train, y_train), dt.score(x_test, y_test)

print(f"Train Score: {dt_train}")
print(f"Test Score: {dt_test}")
     
joblib.dump(dt, 'decision_tree_model_2.joblib')
#clf = joblib.load('decision_tree_model.joblib')
#['protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'count', 'same_srv_rate', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_same_src_port_rate']
#Above mentioned are the best features for the model
#test another data with the model created
new_data = pd.read_csv("Train_data.csv")
le(new_data)
new_data = new_data[selected_features]
new_data = scale.fit_transform(new_data)

predictions=dt.predict(new_data)
#1 is normal and 0 is attack

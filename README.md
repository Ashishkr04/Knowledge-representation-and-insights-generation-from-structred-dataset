 %%
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
%matplotlib inline

# %%
df = pd.read_csv('breast-cancer.csv')

# %%
df

# %%
column_names = ["class","age","menopause","tumor-size","inv-nodes","node","deg","breast","breast-quad","irradiat"]

# %%
df = pd.read_csv('breast-cancer.csv', names= column_names)

# %%
df

# %%
df.info()

# %%
missing_counts = df.isnull().sum()

# %%
missing_counts

# %%
df.describe ()

# %%
column_with_qm = df.columns[df.isin(["?"]).any()]

# %%
column_with_qm

# %%
for col in column_with_qm:
    mode = df[col].mode()[0]
    df.loc[df[col]=="?",col] = mode

# %%
from sklearn.preprocessing import LabelEncoder

# %%
le = LabelEncoder()

# %%
column_with_qm

# %%
df['node'] = le.fit_transform(df['node'])

# %%
df['breast-quad'] = le.fit_transform(df['breast-quad'])

# %%
df['menopause'] = le.fit_transform(df['menopause'])

# %%
df['breast'] = le.fit_transform(df['breast'])

# %%
df['irradiat'] = le.fit_transform(df['irradiat'])


# %%
df['tumor-size'] = le.fit_transform(df['tumor-size'])

# %%
df['inv-nodes'] = le.fit_transform(df['inv-nodes'])

# %%
df['irradiat'] = le.fit_transform(df['irradiat'])

# %%
df['class'] = le.fit_transform(df['class'])

# %%
df['age'] = le.fit_transform(df['age'])

# %%
df

# %% [markdown]
# ## Data Visualization
# 

# %%
#count plot to check distribution of breast cancer in specific age groups
sns.countplot(x = 'age', data = df)

# %%
#countplot to count the number of breast cancer in menopause
sns.countplot(x = 'menopause', data = df)

# %%
#countplot to count the number of breast cancer in specific tumour size
sns.countplot(x = 'tumor-size', data = df)

# %%
#countplot to identify the specific number with breast cancer in left and right breast respectively
sns.countplot(x = 'breast', data = df)

# %%
#countplot to identify the number of breast cancer in specific breast quadrant
sns.countplot(x = 'breast-quad', data = df)

# %%
# checking the distribution plot of all the columns
for column in df:
    print(column)

# %%
#creating for loop for getting distribution plot for all the column
for column in df:
    sns.displot(x =column, data = df)

# %%
# creating a pair plot for the dataset
sns.pairplot(df)

# %%
correlation_matrix = df.corr()

# %%
#constructing a heatmap for the correlation matrix
plt.figure(figsize = (10,10))
sns.heatmap(correlation_matrix, cbar = True, fmt = '.1f', annot = True, cmap = 'Blues')

# %%
#boxplot for outliers detection in the datasest
for column in df:
    plt.figure()
    df.boxplot([column])

# %%
## XG BOOST

# %%
# Step 1: Install xgboost in the correct environment
# Using sys.executable to ensure installation in the active environment
# as directly installing xgboost was not working
import sys
!{sys.executable} -m pip install xgboost


# %%
# Step 3: Import the required libraries
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


# %%
print("Unique values in target column:", df['class'].unique())

# %%
# Preprocessing (Assuming data is already preprocessed and cleaned)
X = df.drop('breast-quad', axis=1)  # Features
y = df['breast-quad']  # Target variable



# %%
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=4, random_state=42)

# %%
model = XGBClassifier()

# %%
model.fit(X_train, y_train)

# %%
# Make predictions
y_pred = model.predict(X_test)

# %%
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# %%
# Plotting the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show() 


# %%
print("Unique values in target column:", df['age'].unique())

# %%
# Preprocessing (Assuming data is already preprocessed and cleaned)
X = df.drop('age', axis=1)  # Features
y = df['age']  # Target variable

# %%
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=6, random_state=42)

# %%
model = XGBClassifier()

# %%
model.fit(X_train, y_train)

# %%
# Make predictions
y_pred = model.predict(X_test)

# %%
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# %%
# Plotting the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show() 


# %%
#loading scikit random forest classifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
#setting random seed
np.random.seed(0)

# %%
df['results'] = pd.Categorical(df['breast-quad'])
df.head()

# %%
#creating train and test data 
df['is_train']= np.random.uniform(0, 1, len(df)) <=0.75
df.head()

# %%
#creating dataframes with test rows and training rows
train,test = df[df['is_train']== True], df[df['is_train']==False]
print('Number of observation in the training data is:', len(train))
print('Number of observation in the test data is:', len(test))

# %%
#create a list of feature column's name
features = df.columns[:4]
#view features
print(features)

# %%
#converting each name into digits
y = pd.factorize(train['breast-quad'])[0]
print(y)

# %%
#Creating a random forest Classifier
clf = RandomForestClassifier(n_jobs=2, random_state = 0)
#training the classifier
clf.fit(train[features],y)

# %%
#applying the trained classifier to the test
clf.predict(test[features])

# %%




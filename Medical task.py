## DATASET DESCRIPTION

# =============================================================================
# 1. Age: displays the age of the individual
# 2. Sex: displays the gender of the individual using the following format :
#     1 = male
#     0 = female
# 3. Chest-pain type: displays the type of chest-pain experienced by the individual using the following format :
#     1 = typical angina
#     2 = atypical angina
#     3 = non — anginal pain
#     4 = asymptotic
# 4. Resting Blood Pressure: displays the resting blood pressure value of an individual in mmHg (unit)
# 5. Serum Cholestrol: displays the serum cholesterol in mg/dl (unit)
# 6. Fasting Blood Sugar: compares the fasting blood sugar value of an individual with 120mg/dl.
#     If fasting blood sugar > 120mg/dl then : 1 (true)
#     else : 0 (false)
# 7. Resting ECG : displays resting electrocardiographic results
#     0 = normal
#     1 = having ST-T wave abnormality
#     2 = left ventricular hyperthrophy
# 8. Max heart rate achieved : displays the max heart rate achieved by an individual.
# 9. Exercise induced angina :
#     1 = yes
#     0 = no
# 10. ST depression induced by exercise relative to rest: displays the value which is an integer or float.
# 11. Peak exercise ST segment :
#     1 = upsloping
#     2 = flat
#     3 = downsloping
# 12. Number of major vessels (0–3) colored by flourosopy : displays the value as integer or float.
# 13. Thal : displays the thalassemia :
#     3 = normal
#     6 = fixed defect
#     7 = reversible defect
# 14. Diagnosis of heart disease : Displays whether the individual is suffering from heart disease or not :
#     0 = absence
#     1, 2, 3, 4 = present.
# =============================================================================

##DATA PROCESSING

import pandas as pd
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category = FutureWarning)
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category = FutureWarning)

df = pd.read_csv('heart_data.csv', header = None)

df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol',
              'fbs', 'restecg', 'thalach', 'exang', 
              'oldpeak', 'slope', 'ca', 'thal', 'target']
print(df)

df['target'] = df.target.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})
df['sex'] = df.sex.map({0: 'female', 1: 'male'})
df['thal'] = df.thal.fillna(df.thal.mean())
df['ca'] = df.ca.fillna(df.ca.mean())

print(df.isnull().sum())

##DATA VISUALIZATION


import matplotlib.pyplot as plt
import seaborn as sns

# distribution of target vs age 
# =============================================================================
# We see that most people who are suffering are of the age of 58, followed by 57.
# Majorly, people belonging to the age group 50+ are suffering from the disease.
# =============================================================================
sns.set_context("paper", font_scale = 2, rc = {"font.size": 20,"axes.titlesize": 25,"axes.labelsize": 20}) 
sns.catplot(kind = 'count', data = df, x = 'age', hue = 'target', order = df['age'].sort_values().unique())
plt.title('Variation of Age for each target class')
plt.show()

# barplot of age vs sex with hue = target
# =============================================================================
# We see that for females who are suffering from the disease are older than males.
# =============================================================================
sns.catplot(kind = 'bar', data = df, y = 'age', x = 'sex', hue = 'target')
plt.title('Distribution of age vs sex with the target class')
plt.show()

df['sex'] = df.sex.map({'female': 0, 'male': 1})


##DATA PREPROCESSING

# =============================================================================
# We see that there are only 6 cells with null values with 4 belonging to attribute ca and 2 to thal.
# As the null values are very less we can either drop them or impute them. I have imputed the mean in place of the null values however one can also delete these rows entirely.
# Now let us divide the data in the test and train set.
# In this project, I have divided the data into an 80: 20 ratio. That is, the training size is 80% and testing size is 20% of the whole data.
# =============================================================================

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler as ss
sc = ss()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#########################################   SVM   #############################################################
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf')
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = classifier.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)

print()
print('Accuracy for training set for svm = {}'.format((cm_train[0][0] + cm_train[1][1])/len(y_train)))
print('Accuracy for test set for svm = {}'.format((cm_test[0][0] + cm_test[1][1])/len(y_test)))

# =============================================================================
# Accuracy for training set for svm = 0.9256198347107438
# Accuracy for test set for svm = 0.8032786885245902
# =============================================================================

#########################################   Logistic Regression  #############################################################
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = classifier.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)

print()
print('Accuracy for training set for Logistic Regression = {}'.format((cm_train[0][0] + cm_train[1][1])/len(y_train)))
print('Accuracy for test set for Logistic Regression = {}'.format((cm_test[0][0] + cm_test[1][1])/len(y_test)))

# =============================================================================
# Accuracy for training set for Logistic Regression = 0.8636363636363636
# Accuracy for test set for Logistic Regression = 0.8032786885245902
# =============================================================================

#########################################   Decision Tree  #############################################################

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = classifier.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)

print()
print('Accuracy for training set for Decision Tree = {}'.format((cm_train[0][0] + cm_train[1][1])/len(y_train)))
print('Accuracy for test set for Decision Tree = {}'.format((cm_test[0][0] + cm_test[1][1])/len(y_test)))

# =============================================================================
# Accuracy for training set for Decision Tree = 1.0
# Accuracy for test set for Decision Tree = 0.7704918032786885
# =============================================================================

##SO WE CAN CONCLUDE THAT SVM GIVES THE MOST ACCURACY LOOKING AT THE ACCURACY
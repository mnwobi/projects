# interact -gpu 
# singularity shell --nv /ocean/containers/ngc/tensorflow/tensorflow_23.04-tf2-py3.sif
# python
#  import tensorflow

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from xgboost import XGBClassifier


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
##
# reading in data and cleaning it

# reading in data and making mega data 

ddint= pd.read_csv('/jet/home/mmarius/project/ddinter_downloads_code_A.csv')
drug_acid= pd.read_csv('/jet/home/mmarius/project/drug_cids.csv')
pchem= pd.read_csv('/jet/home/mmarius/project/pubchem_data.csv')

ddinter_df = ddint[['Drug_A', 'Drug_B', 'Level']]



columns_to_keep = [
    'cid', 'mw', 'polararea', 'complexity', 'xlogp', 'heavycnt', 
    'hbonddonor', 'hbondacc', 'rotbonds', 'exactmass', 'monoisotopicmass', 
    'charge', 'covalentunitcnt', 'isotopeatomcnt', 'totalatomstereocnt', 
    'definedatomstereocnt', 'undefinedatomstereocnt', 'totalbondstereocnt', 
    'definedbondstereocnt', 'undefinedbondstereocnt'
]

# Filter the pubchem_df by keeping only the required columns
pubchem_df = pchem[columns_to_keep]

ddinter_df = ddinter_df[ddinter_df['Level'] != 'Unknown']

# Map 'Level' column from factors (strings) to integers
level_mapping = {'Minor': 0, 'Moderate': 1, 'Major': 2}
ddinter_df['Level'] = ddinter_df['Level'].map(level_mapping)

# Ensure 'Level' is of integer type for scikit-learn
ddinter_df['Level'] = ddinter_df['Level'].astype(int)

# Display the updated 'Level' column
ddinter_df[['Drug_A', 'Drug_B', 'Level']].head()


pubchem_df = pubchem_df.select_dtypes(include=['number'])


drug_acid.rename(columns={'CIDs': 'cid'}, inplace=True)


ddinter_drugA_joined = pd.merge(ddinter_df, drug_acid, left_on='Drug_A', right_on='Drug Name', how='inner')

# Merge with pubchem_df on 'cid' for Drug_A
ddinter_drugA_joined = pd.merge(ddinter_drugA_joined, pubchem_df, on='cid', how='inner')

# Rename columns for Drug_A (add _A suffix)
drug_a_columns = ['cid'] + list(pubchem_df.columns)  # cid and all the numerical properties from pubchem_df
ddinter_drugA_joined = ddinter_drugA_joined.rename(columns={col: f"{col}_A" for col in drug_a_columns})



ddinter_drugA_drugB_joined = pd.merge(ddinter_drugA_joined, drug_acid, left_on='Drug_B', right_on='Drug Name', how='inner')

# Merge with pubchem_df on 'cid' for Drug_B
ddinter_drugA_drugB_joined = pd.merge(ddinter_drugA_drugB_joined, pubchem_df, on='cid', how='inner')

# Rename columns for Drug_B (add _B suffix)
drug_b_columns = ['cid'] + list(pubchem_df.columns)  # cid and all the numerical properties from pubchem_df
ddinter_drugA_drugB_joined = ddinter_drugA_drugB_joined.rename(columns={col: f"{col}_B" for col in drug_b_columns})



ddinter_drugA_drugB_joined['cid_A'] = ddinter_drugA_drugB_joined['cid_A'].astype('category')
ddinter_drugA_drugB_joined['cid_B'] = ddinter_drugA_drugB_joined['cid_B'].astype('category')

# Columns to drop (non-predictors, excluding `cid_A` and `cid_B`)
columns_to_drop = ['Drug_A', 'Drug_B', 'Drug Name_x', 'Drug Name_y']

# Drop the columns except for `cid_A` and `cid_B` which will be kept as primary keys
ddinter_drugA_drugB_joined.drop(columns=columns_to_drop, inplace=True)

# Show the first few rows after dropping the columns
ddinter_drugA_drugB_joined.head()

ddinter_drugA_drugB_joined.drop(columns=['xlogp_A', 'xlogp_B'], inplace=True)

cid_connection_df = ddinter_drugA_drugB_joined[['cid_A', 'cid_B']].reset_index(drop=True)

# Set up feature matrix (train_X) and target vector (train_y)
train_X = ddinter_drugA_drugB_joined.drop(columns=['Level', 'cid_A', 'cid_B'])  # Drop 'cid_A' and 'cid_B' from features
train_y = ddinter_drugA_drugB_joined['Level']  # The target is the 'Level' column

# Split the dataset into training and testing sets (train_X, train_y, test_X, test_y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.2, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, test_size=0.2, random_state=42)


#===================================
# MAKING MODELS
#===================================

# for weights
class_counts = np.bincount(y_train)
class_weights = {i: len(y_train) / (len(class_counts) * count) for i, count in enumerate(class_counts)}
# creating model
model= XGBClassifier( random_state=42, num_class=3, objective='multi:softmax')
# adding weights n running model
sample_weights = np.array([class_weights[y] for y in y_train])
model.fit(X_train, y_train, sample_weight=sample_weights)

# analysing model 
predictions=model.predict(X_test)
print(metrics.classification_report(y_test,predictions))
d=ConfusionMatrixDisplay(confusion_matrix(y_test,predictions))
display_labels=(['Class 0', 'Class 1'])
d.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.savefig('/jet/home/mmarius/project/xg_with_weight_cm.png')
plt.clf()

# feature importance plot 

importances = model.feature_importances_
# Create DataFrame with features and importances
feature_importance = pd.Series(importances, index=X_train.columns)

# Plot of feature importance 
plt.figure(figsize=(10, 6))
feature_importance.plot(kind='bar')
plt.title("Feature Importances")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig('/jet/home/mmarius/project/xg_with_weight_fi.png')
plt.clf()


# Officail model accuracy 
metrics.accuracy_score(model.predict(X_val),y_val)
#0.8463320604592991 accuracy




# Creating model without weight for comparison 
model= XGBClassifier(random_state=42, num_class=3, objective='multi:softmax')
# adding weights n running model
model.fit(X_train, y_train)

# analysing model 
predictions=model.predict(X_test)
print(metrics.classification_report(y_test,predictions))
d=ConfusionMatrixDisplay(confusion_matrix(y_test,predictions))
display_labels=(['Class 0', 'Class 1','Class 2'])
d.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.savefig('/jet/home/mmarius/project/xg_no_weight_cm.png')
plt.clf()



# feature importance plot 

importances = model.feature_importances_
# Create DataFrame with features and importances
feature_importance = pd.Series(importances, index=X_train.columns)

# Plot of feature importance 
plt.figure(figsize=(10, 6))
feature_importance.plot(kind='bar')
plt.title("Feature Importances")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig('/jet/home/mmarius/project/xg_no_weight_fi.png')
plt.clf()

# Officail model accuracy 
metrics.accuracy_score(model.predict(X_val),y_val)

#

# model= RandomForestClassifier()
# model.fit(X_train, y_train, sample_weight=sample_weights)

# # analyszing model 
# predictions=model.predict(X_test)
# print(metrics.classification_report(y_test,predictions))
# d=ConfusionMatrixDisplay(confusion_matrix(y_test,predictions))
# display_labels=(['Class 0', 'Class 1'])
# d.plot(cmap=plt.cm.Blues)
# plt.title('Confusion Matrix')
# plt.savefig('/jet/home/mmarius/project/rf_with_weight_cm.png')
# plt.clf()


# # Officail model accuracy 
# metrics.accuracy_score(model.predict(X_val),y_val)

# # 0.991572753399505


#+++++++++++++++++++++++++++++++++++++++++

# rf no weights

model= RandomForestClassifier()
model.fit(X_train, y_train)

# analyszing model 
predictions=model.predict(X_test)
print(metrics.classification_report(y_test,predictions))
d=ConfusionMatrixDisplay(confusion_matrix(y_test,predictions))
display_labels=(['Class 0', 'Class 1'])
d.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.savefig('/jet/home/mmarius/project/rf_no_weight_cm.png')
plt.clf()



importances = model.feature_importances_
# Create DataFrame with features and importances
feature_importance = pd.Series(importances, index=X_train.columns)

# Plot of feature importance 
plt.figure(figsize=(10, 6))
feature_importance.plot(kind='bar')
plt.title("Feature Importances")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig('/jet/home/mmarius/project/rf_no_weight_fi.png')
plt.clf()


metrics.accuracy_score(model.predict(X_val),y_val)
# 0.9908858671066287

#==============
# LIGHT GBM 
#==============

from lightgbm import LGBMClassifier

model= LGBMClassifier()
model.fit(X_train, y_train)

# analyszing model 
predictions=model.predict(X_test)
print(metrics.classification_report(y_test,predictions))
d=ConfusionMatrixDisplay(confusion_matrix(y_test,predictions))
display_labels=(['Class 0', 'Class 1'])
d.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.savefig('/jet/home/mmarius/project/rf_no_weight_cm.png')
plt.clf()

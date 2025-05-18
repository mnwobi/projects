#!/usr/bin/env python
# coding: utf-8

# # Importing and reorganizing data if needed 

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
phe=pd.read_csv('phenotype.txt',delimiter='\t')
phe.index=phe.iloc[:,1]

ge=pd.read_csv('genotype.txt',delimiter='\t')
ge.dropna()
exp=pd.read_csv('expression.txt',delimiter='\t')
exp=exp.dropna()


# In[3]:


col=['id']
for i in phe.iloc[:,0]:
    col.append(i)

col
data=pd.DataFrame(columns=col,index=phe.columns.drop('individual_id'))
data['id']=data.index
data.index=range(0,836)

# do this but in a loop 
for i in range(len(data)):
    data.iloc[i,1:]=phe.iloc[:,i+1]
data
data=data.dropna()
data


# # Phenotype connections 

# In[16]:


from pandas.plotting import scatter_matrix
from matplotlib.pyplot import figure

fig = plt.figure(figsize=(5, 10))


scatter_matrix(data.drop(columns=['id','GENDER','self_reported_ethnicity','inferred_population']).astype('float64'),figsize  = [20, 7])



# # heat plot 

# changed genders to 1 if male and 0 if female

# In[5]:


data.index=range(0,119)
for i in range(len (data['GENDER'])):
    if data['GENDER'][i]=='Male':
        data['GENDER'][i]=1
    else:
        data['GENDER'][i]=0
data['GENDER']


# In[6]:


# data=data.astype('float64')
data2=data.drop(columns=['GENDER','self_reported_ethnicity','inferred_population']).astype('float64')
data2['GENDER']=data['GENDER'].astype('float64')
data2['self_reported_ethnicity']=data['self_reported_ethnicity'].astype('str')
data2['inferred_population']=data['inferred_population'].astype('str')



matrix=data2.corr()


# In[7]:


import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1,1, figsize = (10,10) )

sns.heatmap(matrix,annot=True)


# # using machine learnign to do linear regression on pvals based on gender 

# In[17]:


import statsmodels.api as sm
pvals=[]
for i in data2.drop(columns=['GENDER','id','self_reported_ethnicity','inferred_population']).columns:
    X=data2[i]
    Y=data2["GENDER"]
    mod = sm.OLS(Y, sm.add_constant(X)).fit()
    #slope
    (mod.summary2().tables[1]["Coef."][i])
    #pvalue
    pvals.append (mod.summary2().tables[1]['P>|t|'][i])


# In[18]:


pvals= -np.log10(pvals)

expect = np.arange(1, pvals.shape[0]+1)/pvals.shape[0]
expect = -np.log10(expect)
expect=np.sort(expect)
pval =np.sort(pvals)
# plt.scatter(expect,pvals)
plt.scatter(expect,pvals)
plt.plot(expect,expect)
plt.title('QQ Plot')
plt.xlabel('expected p')
plt.ylabel('pvals')


# our observed pvals arent lining up with our expected pvals. Based on this we can tell that the gene expressions are differnt in a male and femlae, they arent expressed the same 

# # Multi variable regression 
# seeing which values can predict a female or a male( this can help us see if certain gene expressions are high in males or females. If the activities are the same in women and men then the specific gene actifity shouldn't have a low pvalue or a high t score. There shouldnt be any correlation with gene activity to women/men 

# In[19]:


import statsmodels.api as sm

result= sm.OLS(data2['GENDER'],sm.add_constant(data2.drop(columns=['GENDER','self_reported_ethnicity','inferred_population']))).fit()
result.summary()


# In  conclusion non of the CYP activity differnt between women and men since non of their P values are lower than .05
# We can only infer that weight, BMI, and age play a role when it comes to classifying things 

# # Logistic regression predicting if its a male or female and ROC

# In[20]:


data2


# In[21]:


from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
train,test,train_class,test_class=train_test_split(data2.drop(columns=['GENDER','self_reported_ethnicity','inferred_population']),data2['GENDER'],test_size=.25)


# In[22]:


log_regression = LogisticRegression()

#fit the model using the training data
mod=log_regression.fit(train,train_class)
y_pred_proba = log_regression.predict_proba(test)



# In[23]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(test_class, mod.predict(test))
fpr, tpr, thresholds = roc_curve(test_class, mod.predict_proba(test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic Curve')
plt.legend(loc="lower right")
plt.show()


# # multi regression for CYP3A4M_activity
# if over 1279.5026319661345 we will classify it as over active(1) and under is regular (0)
# 

# Showing how i made the cut off

# In[24]:


col=['id']
for i in phe.iloc[:,0]:
    col.append(i)

col
data=pd.DataFrame(columns=col,index=phe.columns.drop('individual_id'))
data['id']=data.index
data.index=range(0,836)

# do this but in a loop 
for i in range(len(data)):
    data.iloc[i,1:]=phe.iloc[:,i+1]

data=data.dropna()


# In[25]:


data[' '].astype('float64').hist(bins=25)
plt.axvline(x=data2['CYP3A4M_activity'].mean(),c='r')


# In[26]:


data.index=range(0,119)
for i in range(len (data['CYP3A4M_activity'])):
    if data['CYP3A4M_activity'].astype('float64')[i]>1279.5026319661345:
        data['CYP3A4M_activity'][i]=1
    else:
        data['CYP3A4M_activity'][i]=0
data['CYP3A4M_activity']



data2=data.drop(columns=['GENDER','self_reported_ethnicity','inferred_population']).astype('float64')
# data2['GENDER']=data['GENDER'].astype('str')
# data2['self_reported_ethnicity']=data['self_reported_ethnicity'].astype('str')
# data2['inferred_population']=data['inferred_population'].astype('str')
data2


# In[27]:


import statsmodels.api as sm

result= sm.OLS(data2['CYP3A4M_activity'],sm.add_constant(
    data2.drop(columns=['CYP3A4M_activity']))).fit()
result.summary()


# Based on the results, we can tell that CYP3A4T_activity has a strong correlation with CYP3A4M_activity. Also CYP2E1_activity came close to the .05 cut off but fell shot at .07. So we can assume that if CYP3A4M_activity has a high or low activity then CYP3A4T_activity will be high or low also 

# # finding pval for each gene activity via linear regression 

# In[28]:


pvals=[]
for i in data2.drop(columns=['CYP3A4M_activity','id']).columns:
    X=data2[i]
    Y=data2["CYP3A4M_activity"]
    mod = sm.OLS(Y, sm.add_constant(X)).fit()
    #slope
    (mod.summary2().tables[1]["Coef."][i])
    #pvalue
    pvals.append (mod.summary2().tables[1]['P>|t|'][i])


# In[29]:


pvals= -np.log10(pvals)

expect = np.arange(1, pvals.shape[0]+1)/pvals.shape[0]
expect = -np.log10(expect)
expect=np.sort(expect)
pval =np.sort(pvals)
# plt.scatter(expect,pvals)
plt.scatter(expect,pvals)
plt.plot(expect,expect)
plt.title('QQ Plot')
plt.xlabel('expected p')
plt.ylabel('pvals')


# This shows that our p values do match what we expect except for one. This means that we have 1 outlier.

# # Differnece 
# 
# Showing that there is a diffenrce when it come to gen expression, everything isnt equally ditributed and each type of gene expressions arent distriputed the same way. Showing that there are significant genes in this data set and they do go against our null hypothesis that everyhing is expressed the same or similar. 

# In[30]:


exp2 = exp[exp < 1.0e6]
exp2.iloc[1,:].hist()
exp2.iloc[1,:].mean()


# In[31]:


# exp = exp[exp.iloc[:,1:] < 1.0e6]
exp2.iloc[2000,:].hist(bins=9)
exp2.iloc[2000,:].mean()


# In[32]:


exp2.iloc[:101,:].T.describe()


# In[33]:


exp2=exp2.drop(columns='feature_id')


# In[34]:


ll=[]
for i in range(len(exp)):
    ll.append(exp2.iloc[i,:].mean())
    


# In[35]:


# distripution of all of the means 
mean_list=pd.DataFrame(ll)
mean_list.hist(bins=50)


# In[36]:


mean=mean_list.mean()
std=mean_list.std()


# # Position Weight Matrix 

# In[37]:


ge=ge.dropna()
ge.index=np.arange(0,26993)


# In[38]:


ge


# In[39]:


matrix=pd.DataFrame(columns=ge.columns[1:],index=['A','T','G','C'])


# In[40]:


import math as math 
for i in ge[ge.columns[1:]]:
    T=0
    A=0
    G=0
    C=0
    for h in ge[i]:
        if h=='TT':
            T+=2
        if h=='AA':
            A+=2
        if h=='GG':
            G+=2
        if h=='CC':
            T+=2
        if h=='TA':
            T+=1
            A+=1
        if h=='TG':
            T+=1
            G+=1
        if h=='TC':
            T+=1
            C+=1
        if h=='AT':
            T+=1
            A+=1
        if h=='AG':
            A+=1
            G+=1
        if h=='AC':
            A+=1
            C+=1
        if h=='GA':
            G+=1
            A+=1
        if h=='GT':
            T+=1
            G+=1
        if h=='GC':
            G+=1
            C+=1
        if h=='CT':
            T+=1
            C+=1
        if h=='CG':
            C+=1
            G+=1
        if h=='CA':
            A+=1
            C+=1
    A1=math.log((A/26993)/(1/4),2)
    T1=math.log((T/26993)/(1/4),2)
    G1=math.log((G/26993)/(1/4),2)
    C1=math.log((C/26993)/(1/4),2) 
    matrix.iloc[0,:]=A1
    matrix.iloc[1,:]=T1
    matrix.iloc[2,:]=G1
    matrix.iloc[3,:]=C1


# In[41]:


matrix


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





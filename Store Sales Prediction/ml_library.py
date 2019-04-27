
# coding: utf-8

# In[20]:

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from pandas_summary import DataFrameSummary
import re

from IPython.display import HTML, display


# ## Function to read files

# In[21]:

def load_data(path):
    ''' Takes file as input and return a dataframe '''
    df=pd.read_table(path,sep=';')
    return df


# ## Initial view of data

# In[22]:

def intial_analysis(dataframe):
       
    print('\033[1m' + "\nDisplay the shape (columns and rows) of the dataset:" +'\033[0m' )
    print("\tRows : {}\n\tColumns : {}".format(dataframe.shape[0],dataframe.shape[1]))
    
    print('\033[1m' + "\nInformation about the dataset:" +'\033[0m')
    dataframe.info()
    
    print('\033[1m' + "\nDetails on Numerical and Categorical features within dataset:\n" + '\033[0m')
    #list the number of Numerical Features in our dataset.
    numerical_feature_columns = list(df._get_numeric_data().columns)
    print("Numeric Columns:",numerical_feature_columns)
    
    #let's find out the number of Categorical Features in our dataset.
    categorical_feature_columns = list(set(df.columns) - set(df._get_numeric_data().columns))
    print("Categorical Columns:",categorical_feature_columns)
    

    print('\033[1m' + "\nPrint any null values within dataset:\n" + '\033[0m')
                
    labels = []
    values = []
    for col in dataframe.columns:
        labels.append(col)
        values.append(dataframe[col].isnull().sum())
        if values[-1]!=0:
            print(col, values[-1])


# In[37]:

def numerical_features(df):
    #let's find out the number of numerical Features in our dataset.
    numerical_feature_columns = list(df._get_numeric_data().columns)
    return numerical_feature_columns

def categorical_features(df):
    #let's find out the number of Categorical Features in our dataset.
    categorical_feature_columns = list(set(df.columns) - set(df._get_numeric_data().columns))
    return categorical_feature_columns

def categorical_labels(df,categorical_feature_columns):
    for var in categorical_feature_columns:
        print(var, 'contains :', len(df[var].unique()), ' labels')


# ## Missing Values

# In[23]:

def missing_values(df):
    #check null values 
    df_na = (df.isnull().sum() / len(df)) * 100
    df_na = df_na.drop(df_na[df_na == 0].index).sort_values(ascending=False)[:30]
    missing_data = pd.DataFrame({'Missing Ratio' :df_na})
    
    if (df_na.shape[0] != 0):
        f, ax = plt.subplots(figsize=(8, 6))
        plt.xticks(rotation='90')
        sns.barplot(x=df_na.index, y=df_na)
        plt.xlabel('Features', fontsize=15)
        plt.ylabel('Percent of missing values', fontsize=15)
        plt.title('Percent missing data by feature', fontsize=15)
        
    return missing_data.head(22)


# ## Visualization

# In[24]:

def density_plots(df):
    num_cols = list(df._get_numeric_data().columns)
    for i in range(0,len(num_cols),2):
        if len(num_cols) > i+1:
            plt.figure(figsize=(8,2))
            plt.subplot(121)
            sns.distplot(df[num_cols[i]], hist=True, kde=True)
            plt.subplot(122)            
            sns.distplot(df[num_cols[i+1]], hist=True, kde=True)
            plt.tight_layout()
            plt.show()

        else:
            sns.distplot(df[num_cols[i]], hist=True, kde=True)


# In[25]:

def box_plot(df):
    num_cols = list(df._get_numeric_data().columns)
    for i in range(0,len(num_cols),2):
        if len(num_cols) > i+1:
            plt.figure(figsize=(10,4))
            plt.subplot(121)
            sns.boxplot(x=df[num_cols[i]],orient='v')
            plt.subplot(122)            
            sns.boxplot(x=df[num_cols[i+1]],orient='v')
            plt.tight_layout()
            plt.show()

        else:
            sns.boxplot(x=df[num_cols[i]],orient='v')


# In[26]:

def plot_residuals(y_test,y_pred,name="Residual Plot"):
    residuals = y_test - y_pred
    plt.scatter(y_test,residuals)
    plt.hlines(y=0, xmin=0, xmax=10)
    plt.title(name)
    plt.ylabel('Residual')
    plt.xlabel('Fitted')


# ## Feature Engineering

# In[27]:

def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out


# In[28]:

# Standardizing numerical features
from sklearn.preprocessing import StandardScaler

def standardize(df,columns):
    stand_scale= StandardScaler()
    column_transform = columns
    df.loc[:, column_transform] = stand_scale.fit_transform(df.loc[:, column_transform])
    return df


# ## Data Split for validation 

# In[29]:

from sklearn.model_selection import train_test_split

#Let us break the X and y dataframes into training set and test set. For this we will use
#Sklearn package's data splitting function which is based on random function
# Split X and y into training and test set in 80:20 ratio

def split_dataset(df):
    X=df.iloc[:,:-1]
    y=df.iloc[:,-1]
    
    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1,test_size=0.30)
    return X_train,X_test,y_train,y_test


# ## Linear Regression

# In[30]:

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
def cross_validation_regressor(model,x_train,y_train):
    kf = KFold(n_splits=10, random_state=7)
    score = cross_val_score(model,x_train,y_train,cv=kf)
    return score.mean()


# In[38]:

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
def linear_regression(X_train,y_train,X_test, y_test):
    regressor = LinearRegression()
    model = regressor.fit(X_train,y_train)    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test,y_pred)
    mae = mean_absolute_error(y_test,y_pred)
    r2 = r2_score(y_test,y_pred)
    rmse = np.sqrt(mse)
    val = cross_validation_regressor(regressor,X_train,y_train)
    adj_r_squared = 1 - (1-r2)*(len(y_test)-1)/(len(y_test)- X_test.shape[1]- 1)
    stats = pd.DataFrame({'cross_validation':val,'rmse':rmse,'mse':mse,'mae':mae,'r2':r2,'adj_r_squared':adj_r_squared},index=['name'])
    return model,y_pred,stats


# In[36]:

from sklearn.linear_model import Lasso
def lasso(x_train,x_test,y_train,y_test,alpha):
    lass = Lasso(alpha=alpha,random_state=7,normalize=True)
    model = lass.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test,y_pred)
    mae = mean_absolute_error(y_test,y_pred)
    r2=model.score(x_test,y_test)
    rmse = np.sqrt(mse)
    val = cross_validation_regressor(lass,x_train,y_train)
    adj_r_squared = 1 - (1-r2)*(len(y_test)-1)/(len(y_test)- x_test.shape[1]- 1)
    stats = pd.DataFrame({'cross_validation':val,
                         'rmse':rmse,'mse':mse,'mae':mae,'r2':(model.score(x_test,y_test)),'adj_r_squared':adj_r_squared},index=['name'])
    return model,y_pred,stats


# In[ ]:




---
title: "Machine Learning Template"
date: 2020-05-20
tags: [machine learning, linear regression, classification]
header:
  image: "/images/mltemplate/header.jpg"
excerpt: "Machine Learning Template"
mathjax: "true"
---





### Mohammad Affan Sheikh
<font color='maroon'><h3>Institute of Business Administration Karachi (IBA)</h3></font>
#### LinkedIn: [view profile](https://www.linkedin.com/in/mohammad-affan-sheikh/) 
#### Github: [view profile](https://github.com/Affansheikh21)

<h2>This is a Python Template which provides generic functions for carrying out Data Analysis and developing a Machine Learning Model</h2>

### Steps involved in developing a Machine Learning Model:

### [1- Data Exploration](#de)

<ol>
<li>Gathering Data Sources</li>
<li>Data Analysis</li>
<li>Data Pre-processing</li>
</ol>   
 

  ### [2- Model Development](#md)

<ol>
<li>Features Selection</li>
<li>Model Building</li>
<li>Model Evaluation</li>
</ol>

## Imports


```python
#import basic modules
import pandas as pd 
import numpy as np
import seaborn as sb
import math
import warnings
import matplotlib.pyplot as plt        
%matplotlib inline

from sklearn import preprocessing

#import feature selection modules
from sklearn.feature_selection import mutual_info_classif,RFE,RFECV
from sklearn.feature_selection import mutual_info_regression

#import classification modules
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

# import regression modules
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import VotingRegressor

#import classification evaluation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold 
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve 
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
```


```python
# need to install xgboost first
# pip install xgboost in conda environment
try:
    from xgboost import XGBClassifier
except:
    print("Failed to import xgboost, make sure you have xgboost installed")
    print("Use following command to install it: pip install xgboost")
    XGBClassifier = None
```


```python
warnings.filterwarnings("ignore")
try:
    import lightgbm as lgb
except:
    print("Failed to import lightgbm, make sure that you have lightgbm installed")
    print("Use following command to install it: conda install -c conda-forge lightgbm")
    lgb = None
```

<a id ='de'><h2>Data Exploration</h2> </a>

## Gathering Data Sources


```python
# This function will load data irrespective of the its type. 
# In order to use this function write:
    # df = load_data('file_path')

def load_data(file_name):
    def readcsv(file_name):
        return pd.read_csv(file_name)
    def readexcel(file_name):
        return pd.read_excel(file_name)
    def readjson(file_name):
        return pd.read_json(file_name)
    func_map = {
        "csv": readcsv,
        "xls": readexcel,
        "xlsx": readexcel,
        "txt": readcsv,
        "json": readjson
    }
    
    # default reader = readcsv
    reader = func_map.get("csv")
    
    for k,v in func_map.items():
        if file_name.endswith(k):
            reader = v
            break
    return reader(file_name)
```

## Data Analysis

##### analyze(dataframe)


```python
#This function will help you conduct exploratory data analysis
#In order to use this function:
    #analyze(dataframe)

def analyze(df):
    print("Shape is:\n", df.shape)
    print("Columns are:\n", df.columns)
    print("Types are:\n", df.dtypes)
    print("Statistical Analysis of Numerical Columns:\n", df.describe())
```

##### missing_values(dataframe)


```python
#This function will help you list down the count of missing values in the data
#In order to use this function:
    #missing_values(df)
def missing_values(df):
    #Missing values in Each column
    print('Column Name' + '\t\t\t' + 'Null Values')
    return df.apply(lambda x: sum(x.isnull()),axis=0)
```

##### list = list_missing_cols(dataframe)


```python
# make a list of the variables that contain missing values
def list_missing_cols(df):
    vars_with_na = [var for var in df.columns if df[var].isnull().sum()>1]
    return vars_with_na
```

##### find_unique(dataframe)


```python
#This function will help you list down the unique identifiers in the data
#In order to use this function:
    #find_unique(df)
def find_unique(df):
    #Checking Unique Identifiers in our Data
    print('Column Name' + '\t\t\t' + 'IsUnique')
    return df.apply(lambda x: x.is_unique,axis=0)
```

### *Numerical Variables*

#####  list = list_numerical_cols(dataframe)


```python
# This function will return all the numerical columns in the dataset
def list_numerical_cols(df):
    num_vars = [var for var in df.columns if df[var].dtypes != 'O']
    return num_vars
```

##### numcolanalysis(dataframe)


```python
#numerical analysis
#histograms and boxplots for all numerical columns
def numcolanalysis(df):
    numcols = df.select_dtypes(include=np.number)
    
    # Box plot for numerical columns
    for col in numcols:
        fig = plt.figure(figsize = (5,5))
        sb.boxplot(df[col], color='grey', linewidth=1)
        plt.tight_layout()
        plt.title(col)
        plt.savefig("Numerical.png")
    
    # Lets also plot histograms for these numerical columns
    df.hist(column=list(numcols.columns),bins=25, grid=False, figsize=(15,12),
                 color='#86bf91', zorder=2, rwidth=0.9)
```

### *String Variables*

##### list = list_categorical_cols(dataframe)


```python
# This function will return all the numerical columns in the dataset
def list_categorical_cols(df):
    cat_vars = [var for var in df.columns if df[var].dtypes == 'O']
    return cat_vars
```

##### stringcolanalysis(dataframe)


```python
#string column analysis analysis
def stringcolanalysis(df):
    stringcols = df.select_dtypes(exclude=[np.number, "datetime64"])
    fig = plt.figure(figsize = (8,10))
    for i,col in enumerate(stringcols):
        fig.add_subplot(4,2,i+1)
        fig.savefig('Categorical.png')
        df[col].value_counts().plot(kind = 'bar', color='black' ,fontsize=10)
        plt.tight_layout()
        plt.title(col)
```

### *Discrete Variables*


```python
#  list of discrete variables
def list_discrete(df):
    temp_list = list_numerical_cols(df)
    discrete_vars = [var for var in temp_list if len(df[var].unique())<20]
    return discrete_vars
```

### *Continuous Variables*


```python
# list of continuous variables
def list_continuous(df):
    num_vars = list_numerical_cols(df)
    discrete_vars = list_discrete(df)
    cont_vars = [var for var in num_vars if var not in discrete_vars]
    return cont_vars
```

### *Outliers*


```python
# let's make boxplots to visualise outliers in the continuous variables 
def plot_outliers(df):
    # log does not take negative values, so let's be careful and skip those variables
    cont_vars = list_continuous(df)
    for var in cont_vars:
        if 0 in df[var].unique():
            pass
        else:
            df[var] = np.log(df[var])
            df.boxplot(column=var)
            plt.title(var)
            plt.ylabel(var)
            plt.show()
```

### *Correlation*


```python
# Helper function to perform correlation analysis over numerical columns
def correlation_anlysis(df):
    # NOTE: If label column is non numeric, we need to 'encode' it before calling this function to have a better visibility
    numcols = df.select_dtypes(include=np.number)
    corr = numcols.corr()
    ax = sb.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sb.diverging_palette(20, 220, n=200),
    square=True
    )
    
    ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right')
```

## Data Pre-Processing

### *Missing Values*


```python
# function to replace NA in categorical variables
def fill_categorical_na(df):
    var_list = list_categorical_cols(df)
    df[var_list] = df[var_list].fillna('Missing')
    return df


#function to replace NA with dedicated value
def fill_categorical(df,value):
    var_list = list_categorical_cols(df)
    df[var_list] = df[var_list].fillna(value)
    return df

```


```python
#function to replace NA in numerical variables with 0
def fill_numerical_na(df):
    vars_with_na = list_numerical_cols(df)
    # replace the missing values
    for var in vars_with_na:
        df[var].fillna(0, inplace=True)


#function to replace NA in numerical variables with mode
def fill_numerical_mode(df):
    vars_with_na = list_numerical_cols(df)
    # replace the missing values
    for var in vars_with_na:
    # calculate the mode
        mode_val = df[var].mode()[0]
        df[var].fillna(mode_val, inplace=True)
        
#function to replace NA in numerical variables with mean
def fill_numerical_mean(df):
    vars_with_na = list_numerical_cols(df)
    # replace the missing values
    for var in vars_with_na:
    # calculate the mode
        mean_val = df[var].mean()[0]
        df[var].fillna(mean_val, inplace=True)

        
#function to replace NA in numerical variables with dedicated value
def fill_numerical(df,value):
    vars_with_na = list_numerical_cols(df)
    # replace the missing values
    for var in vars_with_na:
        df[var].fillna(value, inplace=True)
        
```

### *Other Cleaning Functions*


```python
def dtype_date(df, to_date):
    # Deal with incorrect data in date column
    for i in to_date:
        df[i] = pd.to_datetime(df[i], errors='coerce')
    return df
            
def dtype_numeric(df, to_numeric):
    # Deal with incorrect data in numeric columns
    for i in to_numeric:
        df[i] = pd.to_numeric(df[i], errors='coerce')
    return df


def drop_useless_colums(df, cols_to_delete):
    # Drop useless columns before dealing with missing values
    for i in cols_to_delete:
        df = df.drop(i, axis=1)
    return df
            
    
def drop_useless_rows(df):
    # Drop useless rows before dealing with missing values
    # Delete all rows containing 40% or more missing data
    min_threshold = math.ceil(len(df.columns)*0.4)
    df = df.dropna(thresh=min_threshold)
    return df
    
    
def drop_na_rows(df, cols_to_drop_na_rows):
    # Drop rows with missing values for the columns specifically provided by the driver program
    for i in cols_to_drop_na_rows:
        df = df.drop(df[df[i].isnull()].index)
    return df
```

### *Encoding*


```python
def apply_label_encoding(df, cols=[]):
    le = preprocessing.LabelEncoder()
    for i in cols:
        le.fit(df[i])
        #print(list(le.classes_))
        df[i] = le.transform(df[i])
    return df
```


```python
#One hot encoding
def stringcolencoding(df, cols=[]):
    label_encoder = preprocessing.LabelEncoder()
    onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
    for col in cols:
        integer_encoded = label_encoder.fit_transform(df[col])
        ohe = onehot_encoder.fit_transform(integer_encoded.reshape(-1,1))
        dfOneHot = pd.DataFrame(ohe, columns = [col+"_"+str(int(i)) for i in range(ohe.shape[1])])
        df = pd.concat([df, dfOneHot], axis=1)
        df = df.drop(col, axis=1)
```


```python
# One Hot encoding with Pandas categorical dtype
def stringcolencoding_v2(df, cols=[]):
    for col in cols:
        df[col] = pd.Categorical(df[col])
        dfDummies = pd.get_dummies(df[col], prefix = col)
        df = pd.concat([df, dfDummies], axis=1)
        df = df.drop(col, axis=1)
    return df
```

<a id ='md'><h2>Model Development</h2> </a>

### *X-y Split*


```python
def XYsplit(df, label_col):
    y = df[label_col].copy()
    X = df.drop(label_col,axis=1)
    return X,y
```

### *Train-Test Split*


```python
def traintestsplit(df,split,random=None, label_col=''):
    #make a copy of the label column and store in y
    y = df[label_col].copy()
    #now delete the original
    X = df.drop(label_col,axis=1)
    #manual split
    trainX, testX, trainY, testY= train_test_split(X, y, test_size=split, random_state=random)
    return X, trainX, testX, trainY, testY
```

### *Cross Validation techniques*


```python
def cross_valid_kfold(X, y, split=10, random=None, shuffle=False):
    """
    Generator function for KFold cross validation
    """
    kf = KFold(n_splits=split, random_state=random, shuffle=shuffle)
    for train_index, test_index in kf.split(X):
        trainX, testX = X.iloc[train_index], X.iloc[test_index] 
        trainY, testY = y.iloc[train_index], y.iloc[test_index]
        yield trainX,trainY,testX,testY
    

def cross_valid_repeated_kf(X, y, split=10, random=None, repeat=10):
    """
    Generator function for Repeated KFold cross validation
    """
    kf = RepeatedKFold(n_splits=split, random_state=random, n_repeats=repeat)
    for train_index, test_index in kf.split(X):
        trainX, testX = X.iloc[train_index], X.iloc[test_index] 
        trainY, testY = y.iloc[train_index], y.iloc[test_index]
        yield trainX,trainY,testX,testY
        

def cross_valid_stratified_kf(X, y, split=10, random=None, shuffle=False):
    """
    Generator function for Stratified KFold cross validation
    """
    skf = StratifiedKFold(n_splits=split, random_state=random, shuffle=shuffle)
    for train_index, test_index in skf.split(X, y):
        trainX, testX = X.iloc[train_index], X.iloc[test_index] 
        trainY, testY = y.iloc[train_index], y.iloc[test_index]
        yield trainX,trainY,testX,testY


def cross_valid_strat_shuffle_kf(X, y, split=10, random=None):
    """
    Generator function for StratifiedShuffle cross validation
    """
    sss = StratifiedShuffleSplit(n_splits=split, random_state=random)
    for train_index, test_index in sss.split(X, y):
        trainX, testX = X.iloc[train_index], X.iloc[test_index] 
        trainY, testY = y.iloc[train_index], y.iloc[test_index]
        yield trainX,trainY,testX,testY
```

## Feature Selection

### *Random Forest Feature Selection*


```python
def RFfeatureimportance(df, trainX, testX, trainY, testY, trees=10, random=None, regression=False):
    if regression:
        clf  = RandomForestRegressor(n_estimators=trees, random_state=random)
    else:
        clf  = RandomForestClassifier(n_estimators=trees, random_state=random)
    clf.fit(trainX,trainY)
    #validationmetrics(clf,testX,testY)
    res = pd.Series(clf.feature_importances_, index=df.columns.values).sort_values(ascending=False)*100
    print(res)
    return res
```

### *Run Algo with selected features*


```python
#select features with importance >=threshold
def MachineLearningwithRFFS(df, label_col, threshold=5, algo_list=get_supported_algorithms(), regression=False):
    # lets create a copy of this dataframe and perform feature selection analysis over that
    df_cpy = df.copy()
    df_cpy, trainX, testX, trainY, testY = traintestsplit(df_cpy, 0.2, 91, label_col=label_col)
    res = RFfeatureimportance(df_cpy, trainX, testX, trainY, testY, trees=10, regression=regression)
    
    impftrs = list(res[res > threshold].keys())
    #impftrs.append(label_col)
    
    print ("Selected Features =" + str(impftrs))
    print(df.shape)
    results = run_algorithms(df, label_col, algo_list=algo_list, feature_list=impftrs)
    return {"selected_features": impftrs, "results": results}

```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-29-d14427beebec> in <module>
          1 #select features with importance >=threshold
    ----> 2 def MachineLearningwithRFFS(df, label_col, threshold=5, algo_list=get_supported_algorithms(), regression=False):
          3     # lets create a copy of this dataframe and perform feature selection analysis over that
          4     df_cpy = df.copy()
          5     df_cpy, trainX, testX, trainY, testY = traintestsplit(df_cpy, 0.2, 91, label_col=label_col)
    

    NameError: name 'get_supported_algorithms' is not defined


### *With Cross Validation*


```python
#select features with importance >=threshold
def MachineLearningwithRFFS_CV(df, label_col, threshold=5, algo_list=get_supported_algorithms(), regression=False):
    # lets create a copy of this dataframe and perform feature selection analysis over that
    df_cpy = df.copy()
    df_cpy, trainX, testX, trainY, testY = traintestsplit(df_cpy, 0.2, 91, label_col=label_col)
    res = RFfeatureimportance(df_cpy, trainX, testX, trainY, testY,
                              trees=10, regression=regression)

    impftrs = list(res[res > threshold].keys())
    
    print ("Selected Features =" + str(impftrs))
    print(df.shape)
    if regression:
        cross_valid_method = cross_valid_kfold
    else:
        cross_valid_method = cross_valid_stratified_kf
    results = run_algorithms_cv(df, label_col, algo_list=algo_list, feature_list=impftrs, cross_valid_method=cross_valid_method)
    return {"selected_features": impftrs, "results": results}
    
```

### *Mutual Information Feature Selection*


```python
#determine the important features given by MIFS
def mutualinformation(df, label_col, regression=False):
    df_cpy = df.copy()
    y = df_cpy[label_col].copy()
    X = df_cpy.drop(label_col,axis=1)
    if regression:
        mutual_info = mutual_info_regression(X,y,random_state=35)
    else:
        mutual_info = mutual_info_classif(X,y,random_state=35)
    results = pd.Series(mutual_info, index=X.columns).sort_values(ascending=False)*100
    print(results)
    return results
```

### *Without Cross Validation*


```python
#### Without Cross Validation
#select features with importance >=threshold
def MachineLearningwithMIFS(df, label_col, threshold=5, algo_list=get_supported_algorithms(), regression=False):
    
    # lets create a copy of this dataframe and perform feature selection analysis over that
    df_cpy = df.copy()
    res = mutualinformation(df_cpy, label_col=label_col, regression=regression)
    
    #include all selected features in impftrs
    impftrs = list(res[res > threshold].keys())
    
    print ("Selected Features =" + str(impftrs))
    
    results = run_algorithms(df, label_col, algo_list=algo_list, feature_list=impftrs)
    return {"selected_features": impftrs, "results": results}
```

### *With Cross Validation*


```python
#### With Cross Validation
#select features with importance >=threshold
def MachineLearningwithMIFS_CV(df, label_col, threshold=5, algo_list=get_supported_algorithms(), regression=False):
    
    # lets create a copy of this dataframe and perform feature selection analysis over that
    df_cpy = df.copy()
    res = mutualinformation(df_cpy, label_col=label_col, regression=regression)
    
    #include all selected features in impftrs
    impftrs = list(res[res > threshold].keys())
    
    print ("Selected Features =" + str(impftrs))
    if regression:
        cross_valid_method = cross_valid_kfold
    else:
        cross_valid_method = cross_valid_stratified_kf
    results = run_algorithms_cv(df, label_col, algo_list=algo_list, feature_list=impftrs, cross_valid_method=cross_valid_method)
    return {"selected_features": impftrs, "results": results}

```

### *Recursive Elimination Feature Selection*


```python
def GenericREFS(df, label_col,
                algo_list=get_supported_algorithms(),
                re_algo=RandomForestClassifier,
                **kwargs):
    
    X,y = XYsplit(df, label_col)
    clf = re_algo(**kwargs)
    selector = RFE(estimator=clf, step=1)
    selector = selector.fit(X,y)
    feature_list = X.columns[selector.support_].tolist()
    
    results = run_algorithms(df, label_col, algo_list=algo_list, feature_list=feature_list)
    return {"selected_features": feature_list, "results": results}
```

### *With Cross Validation*


```python
#### With Cross Validation
def GenericREFS_CV(df, label_col,
                algo_list=get_supported_algorithms(),
                regression=False,
                re_algo=RandomForestClassifier,
                **kwargs):
    
    X,y = XYsplit(df, label_col)
    clf = re_algo(**kwargs)
    selector = RFECV(estimator=clf, step=1, cv=10)
    selector = selector.fit(X,y)
    feature_list = X.columns[selector.support_].tolist()
    if regression:
        cross_valid_method = cross_valid_kfold
    else:
        cross_valid_method = cross_valid_stratified_kf
    results = run_algorithms_cv(df, label_col, algo_list=algo_list, feature_list=feature_list, cross_valid_method=cross_valid_method)
    return {"selected_features": feature_list, "results": results}

```


```python
# Helper function to provide list of classification algorithms to be used for recursive elimination feature selection
def get_supported_algorithms_refs():
    algo_list = [LogisticRegression, GradientBoostingClassifier, AdaBoostClassifier,
                          DecisionTreeClassifier, RandomForestClassifier]
    return algo_list

# Helper function to provide list of regression algorithms to be used for recursive elimination feature selection
def get_supported_reg_algorithms_refs():
    algo_list = [LinearRegression, RandomForestRegressor,
                 DecisionTreeRegressor, GradientBoostingRegressor, AdaBoostRegressor]
    return algo_list
```

### *Feature Selection Using PCA*


```python
#Without Cross Validation
def PCA_FS(df, label_col, n_components, algo_list=get_supported_algorithms()):
    df_cpy = df.copy()
    X,y = XYsplit(df_cpy, label_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # First we need to normalize the data
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # Now perform PCA
    pca = PCA(n_components=n_components)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    
    algo_model_map = {}
    # At this stage we apply alogorithms
    for algo in algo_list:
        print("============ " + algo.__name__ + " ===========")
        res = algo(X_train, X_test, y_train, y_test)
        algo_model_map[algo.__name__] = res.get("model_obj", None)
        
        print("============================== \n")
    return {"n_components": n_components, "results": algo_model_map}
```

### *With Cross Validation*


```python
#### With Cross Validation
def PCA_FS_CV(df, label_col, n_components, algo_list=get_supported_algorithms(), regression=False):
    df_cpy = df.copy()
    X,y = XYsplit(df_cpy, label_col)
    
    cross_valid_method = cross_valid_kfold if regression else cross_valid_stratified_kf 
    result = {}
    algo_model_map = {}
    for algo in algo_list:
        clf = None
        result[algo.__name__] = dict()
        for X_train,y_train,X_test,y_test in cross_valid_method(X, y, split=10):
            # First we need to normalize the data
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
            
            # Now perform PCA
            pca = PCA(n_components=n_components)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
            
            # apply algo on this fold and save result for later usage
            res_algo = algo(X_train, X_test, y_train, y_test, verbose=False, clf=clf)
            # Get trained model so we could use it again in the next iteration
            clf = res_algo.get("model_obj", None)
            
            for k,v in res_algo.items():
                if k == "model_obj":
                    continue
                if k not in result[algo.__name__].keys():
                    result[algo.__name__][k] = list()
                result[algo.__name__][k].append(v)
            
        algo_model_map[algo.__name__] = clf
        
    
    score_map = dict()
    # let take average scores for all folds now
    for algo, metrics in result.items():
        print("============ " + algo + " ===========")
        score_map[algo] = dict()
        for metric_name, score_lst in metrics.items():
            score_map[algo][metric_name] = np.mean(score_lst)
        print(score_map[algo])
        print ("============================== \n")
        score_map[algo]["model_obj"] =  algo_model_map[algo]
    return {"n_components": n_components, "results": algo_model_map}
```

## Model Building

<font color='red'><h2>Classification</h2></font>

###  *Logistic Regression*


```python
def LogReg(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf  = LogisticRegression()
    clf.fit(trainX , trainY)
    return validationmetrics(clf,testX,testY,verbose=verbose)
```

### *KNN*


```python
def KNN(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf = KNeighborsClassifier()
    clf.fit(trainX , trainY)
    return validationmetrics(clf,testX,testY,verbose=verbose)
```

### *Gradient Boosting*


```python
def GadientBoosting(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf = GradientBoostingClassifier()
    clf.fit(trainX , trainY)
    return validationmetrics(clf,testX,testY,verbose=verbose)
```

### *AdaBoost*


```python
def AdaBoost(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    clf.fit(trainX , trainY)
    return validationmetrics(clf,testX,testY,verbose=verbose)
```

### *Support Vector Machine*


```python
def SVM(trainX, testX, trainY, testY, svmtype="SVC", verbose=True, clf=None):
    # for one vs all
    if not clf:
        if svmtype == "Linear":
            clf = svm.LinearSVC()
        else:
            clf = svm.SVC()
    clf.fit(trainX , trainY)
    return validationmetrics(clf,testX,testY,verbose=verbose)
```

### *Decision Tree*


```python
def DecisionTree(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf = DecisionTreeClassifier()
    clf.fit(trainX , trainY)
    return validationmetrics(clf,testX,testY,verbose=verbose)
```

### *Random Forest*


```python
def RandomForest(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf  = RandomForestClassifier()
    clf.fit(trainX , trainY)
    return validationmetrics(clf,testX,testY,verbose=verbose)
```

### *Naive Bayes*


```python
def NaiveBayes(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf = GaussianNB()
    clf.fit(trainX , trainY)
    return validationmetrics(clf,testX,testY,verbose=verbose)
```

### *MultiLayerPerceptron*


```python
def MultiLayerPerceptron(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf = MLPClassifier(hidden_layer_sizes=5)
    clf.fit(trainX,trainY)
    return validationmetrics(clf,testX,testY,verbose=verbose)
```

### *XgBoost*


```python
def XgBoost(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf = XGBClassifier(random_state=1,learning_rate=0.01)
    clf.fit(trainX,trainY)
    return validationmetrics(clf,testX,testY,verbose=verbose)
```

### *LightGbm*


```python
def LightGbm(trainX, testX, trainY, testY, verbose=True, clf=None):
    d_train = lgb.Dataset(trainX, label=trainY)
    params = {}
    params['learning_rate'] = 0.003
    params['boosting_type'] = 'gbdt'
    params['objective'] = 'binary'
    params['metric'] = 'binary_logloss'
    params['sub_feature'] = 0.5
    params['num_leaves'] = 10
    params['min_data'] = 50
    params['max_depth'] = 10
    clf = lgb.train(params, d_train, 100)
    return validationmetrics(clf,testX,testY,verbose=verbose)
```

<font color='red'><h2>Regression</h2></font>

### *Linear Regression*


```python
def LinearReg(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf  = LinearRegression()
    clf.fit(trainX , trainY)
    return validationmetrics_reg(clf, testX, testY, verbose=verbose)
```

### *Random Forest Regressor*


```python
def RandomForestReg(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf = RandomForestRegressor(n_estimators=100)
    clf.fit(trainX , trainY)
    return validationmetrics_reg(clf, testX, testY, verbose=verbose)


```

### *Polynomial Regression*


```python
def PolynomialReg(trainX, testX, trainY, testY, degree=3, verbose=True, clf=None):
    poly = PolynomialFeatures(degree = degree)
    X_poly = poly.fit_transform(trainX)
    poly.fit(X_poly, trainY)
    if not clf:
        clf = LinearRegression() 
    clf.fit(X_poly, trainY)
    return validationmetrics_reg(clf, poly.fit_transform(testX), testY, verbose=verbose)
```

### *Support Vector Regression*


```python
def SupportVectorRegression(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf = SVR(kernel="rbf")
    clf.fit(trainX , trainY)
    return validationmetrics_reg(clf, testX, testY, verbose=verbose)
```

### *Decision Tree Regressor*


```python
def DecisionTreeReg(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf = DecisionTreeRegressor()
    clf.fit(trainX , trainY)
    return validationmetrics_reg(clf, testX, testY, verbose=verbose)
```

### *Gradient Boosting Regression*


```python
def GradientBoostingReg(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf = GradientBoostingRegressor()
    clf.fit(trainX , trainY)
    return validationmetrics_reg(clf, testX, testY, verbose=verbose)
```

### *AdaBoost Regression*


```python
def AdaBooostReg(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf = AdaBoostRegressor(random_state=0, n_estimators=100)
    clf.fit(trainX , trainY)
    return validationmetrics_reg(clf, testX, testY, verbose=verbose)
```

### *Voting Regressor*


```python

def VotingReg(trainX, testX, trainY, testY, verbose=True, clf=None):
    lr = LinearRegression()
    rf = RandomForestRegressor(n_estimators=100)
    sv = SVR(kernel="rbf")
    dt = DecisionTreeRegressor()
    gb = GradientBoostingRegressor()
    ab = AdaBoostRegressor(random_state=0, n_estimators=100)
    if not clf:
        clf = VotingRegressor([('rf', rf), ('dt', dt), ('gb', gb), ('ab', ab)])
    clf.fit(trainX , trainY)
    return validationmetrics_reg(clf, testX, testY, verbose=verbose)
```

<font color='red'><h2>Choosing the Best Model</h2></font>


```python
def get_supported_classification_algorithms():
    covered_algorithms = [LogReg, KNN, GadientBoosting, AdaBoost,
                          SVM, DecisionTree, RandomForest, NaiveBayes,
                          MultiLayerPerceptron]
    if XGBClassifier:
        covered_algorithms.append(XgBoost)
    if lgb:
        covered_algorithms.append(LightGbm)
    return covered_algorithms
```


```python
def get_supported_regression_algorithms():
    covered_algorithms = [LinearReg, RandomForestReg, PolynomialReg, SupportVectorRegression,
                          DecisionTreeReg, GradientBoostingReg, AdaBooostReg, VotingReg]
    return covered_algorithms
```


```python
def get_supported_algorithms(classification=False,regression=False):
    if classification:
        algo_list = get_supported_classification_algorithms()
    else:
        algo_list = get_supported_regression_algorithms()
    return algo_list
    
```


```python
def find_best_model(df,label_col,feature_list=[]):
    algo_list = get_supported_algorithms()
    """
    Run Algorithms with manual split
    """
    # Lets make a copy of dataframe and work on that to be on safe side 
    _df = df.copy()
    
    if feature_list:
        impftrs = feature_list
        impftrs.append(label_col)
        _df = _df[impftrs]
    
    _df, trainX, testX, trainY, testY = traintestsplit(_df, 0.2, 91, label_col=label_col)
    algo_model_map = {}
    for algo in algo_list:
        print("============ " + algo.__name__ + " ===========")
        res = algo(trainX, testX, trainY, testY)
        algo_model_map[algo.__name__] = res.get("model_obj", None)
        print ("============================== \n")
    
    return algo_model_map
    
    
```


```python
# With stratified kfold validation support
def find_best_model_cv(df, label_col, algo_list=get_supported_algorithms(), feature_list=[], cross_valid_method=cross_valid_stratified_kf):
    """
    Run Algorithms with cross validation
    """
    _df = df.copy()
    X,y = XYsplit(_df, label_col)
    
    # Select features if specified by driver program
    if feature_list:
        X = X[feature_list]
    
    result = {}
    algo_model_map = {}
    for algo in algo_list:
        clf = None
        result[algo.__name__] = dict()
        for trainX,trainY,testX,testY  in cross_valid_method(X, y, split=10):
            res_algo = algo(trainX, testX, trainY, testY, verbose=False, clf=clf)
            # Get trained model so we could use it again in the next iteration
            clf = res_algo.get("model_obj", None)
            
            for k,v in res_algo.items():
                if k == "model_obj":
                    continue
                if k not in result[algo.__name__].keys():
                    result[algo.__name__][k] = list()
                result[algo.__name__][k].append(v)
                
        algo_model_map[algo.__name__] = clf
        
    
    score_map = dict()
    # let take average scores for all folds now
    for algo, metrics in result.items():
        print("============ " + algo + " ===========")
        score_map[algo] = dict()
        for metric_name, score_lst in metrics.items():
            score_map[algo][metric_name] = np.mean(score_lst)
        print(score_map[algo])
        print ("============================== \n")
        score_map[algo]["model_obj"] = algo_model_map[algo]
    
    return score_map

```

## Model Evaluation

## *Classification*


```python
def validationmetrics(model,testX,testY, verbose=True):
    
    predictions = model.predict(testX)
    
    if model.__class__.__module__.startswith('lightgbm'):
        for i in range(0, predictions.shape[0]):
            predictions[i]= 1 if predictions[i] >= 0.5 else 0
    
    #Accuracy
    accuracy = accuracy_score(testY, predictions)*100
    
    #Precision
    precision = precision_score(testY, predictions,pos_label=1,labels=[0,1])*100
    
    #Recall
    recall = recall_score(testY, predictions,pos_label=1,labels=[0,1])*100
    
    #get FPR (specificity) and TPR (sensitivity)
    fpr , tpr, _ = roc_curve(testY, predictions)
    
    #AUC
    auc_val = auc(fpr, tpr)
    
    #F-Score
    f_score = f1_score(testY, predictions)
    
    if verbose:
        print("Prediction Vector: \n", predictions)
        print("Accuracy: \n", accuracy)
        print("Precision of event Happening: \n", precision)
        print("Recall of event Happening: \n", recall)
        print("AUC: \n",auc_val)
        print("F-Score:\n", f_score)
        #confusion Matrix
        print("Confusion Matrix: \n", confusion_matrix(testY, predictions,labels=[0,1]))
    
    res_map = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "auc_val": auc_val,
                "f_score": f_score,
                "model_obj": model
              }
    return res_map
```

## *Regression*


```python
def validationmetrics_reg(model,testX,testY, verbose=True):
    predictions = model.predict(testX)
    
    # R-squared
    r2 = r2_score(testY,predictions)
    
    # Adjusted R-squared
    r2_adjusted = 1-(1-r2)*(testX.shape[0]-1)/(testX.shape[0]-testX.shape[1]-1)
    
    # MSE
    mse = mean_squared_error(testY,predictions)
    
    #RMSE
    rmse = math.sqrt(mse)
    
    if verbose:
        print("R-Squared Value: ", r2)
        print("Adjusted R-Squared: ", r2_adjusted)
        print("RMSE: ", rmse)
    
    res_map = {
                "r2": r2,
                "r2_adjusted": r2_adjusted,
                "rmse": rmse,
                "model_obj": model
              }
    return res_map
```


```python

```

"""
Job Application Rate Predicted Modeling
Author: Hongfei Yang <yangh4@seattleu.edu>
Date: 03/27/2018
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold

print(time.ctime())


###############################################################################

#=============#
# Import Data #
#=============#


np.random.seed(10)
filepath = './ozan_p_pApply_intern_challenge_03_26_min.csv'

def get_raw_data(filepath):
    try:
        dataset = pd.read_csv(filepath)
    except FileNotFoundError:
        raise Exception(
                'Fail to find data file from path %s' % filepath )
    return dataset

def shuffle_data(data_set):
    idx = np.random.choice(range(len(data_set)), len(data_set), replace=False)
    return data_set.iloc[idx]

raw_data = get_raw_data(filepath)  # Original Data
raw_data = shuffle_data(raw_data)  # Shuffle for preventing pre-existed order 

raw_data.head(3)

##############################################################################

#===============#
# Data Analysis #
#===============#

raw_data.describe()
# NOTE: 1) Scales are different
#       2) Features are heavily right skewed


# Print out missing values
for feature in raw_data.columns:
    na_count = raw_data[feature].isnull().sum()
    if na_count != 0:
        print(na_count, 'missing values in', feature)

# Check if it is the case that title_proximity_tfidf and 
# description_proximity_tfidf are always absense at the same time.
null_title = raw_data['title_proximity_tfidf'].isnull()
null_description = raw_data['description_proximity_tfidf'].isnull()
all_equal = null_title == null_description
if all_equal.sum() == len(all_equal):
    print('Two variables are always missing at the same time')
else:
    print('Two variables are NOT always missing at the same time')   
    

# Imput missing values for city_match by creating a new category for missing
raw_data.fillna(value={'city_match': -1}, inplace=True)
raw_data['city_match'] = raw_data['city_match'].astype('int')

# Feature Analysis

num_features = raw_data.columns[raw_data.dtypes == 'float64']

# Plot correlation matrix between numeric features
def visualize_correlation_matrix(data_set, features):
    cor_mat = data_set[features].corr()
    sns.heatmap(cor_mat, cmap='coolwarm', annot=True)
    
visualize_correlation_matrix(raw_data, num_features)
# NOTE: Moderate postive correlation between main_query_tfidf
#       and title_proximity_tfidf


# Plot the distribution of each numeric feature
def numerical_analysis(data_set, feature, exclude=None):
    na_cond = np.asarray(raw_data[feature].notnull())
    if exclude != None:
        exc_cond = np.asarray(raw_data[feature] != exclude)
        boo = na_cond & exc_cond
    else:
        boo = na_cond
        
    data = raw_data[feature][boo]
    
    applied = np.asarray(raw_data['apply'][boo] == 1)
    not_applied = np.asarray(raw_data['apply'][boo] == 0)
    applied_data = raw_data[feature][boo][applied]
    not_applied_data = raw_data[feature][boo][not_applied]
    
    sns.kdeplot(data, label='all')
    sns.kdeplot(applied_data, label='applied')
    sns.kdeplot(not_applied_data, label='not applied')
    plt.title(feature)
    plt.legend()
    #sns.FacetGrid(raw_data[boo], col='apply').map(sns.distplot, feature)

for feature in num_features:
    plt.figure()
    numerical_analysis(raw_data, feature)

for feature in num_features:
    plt.figure()
    numerical_analysis(raw_data, feature, 0)
    
# NOTE: 1) The distribution is quiet different with and without 0
#       2) Most variables are right skewed
#       3) For main_query_tfidf, when the score is less than 2/3 user is
#          less likely to apply
#       4) Lower query_title_score higher likelihood of apply


# Log transform nueric features
def log_transform_features(data_set, features):
    for feature in features:
        data_set[feature] = data_set[feature].map(lambda x: np.log(x + 0.01) if x > 0 else x)
    return data_set

log_features = ['description_proximity_tfidf', 'query_jl_score', 
                    'query_title_score', 'job_age_days']

raw_data = log_transform_features(raw_data, log_features)
    
# Min-max normalize features
#def min_max_scale(data_set, features):
#    for feature in features:
#        boo = data_set[feature].notnull()
#        min_val = min(data_set[feature][boo])
#        max_val = max(data_set[feature][boo])
#        
#        data_set[feature] = (data_set[feature] - min_val) / (max_val - min_val)
#        
for feature in log_features:
    plt.figure()
    numerical_analysis(raw_data, feature, 0)
    
# Plot city_match across apply
sns.FacetGrid(raw_data, col='apply').map(sns.countplot, 'city_match')

# Discard 

##############################################################################

#=====================#
# Feature engineering #
#=====================#

## Create a binary variable to indicate whether a feautre is 0
#def create_is_zero_feature(data_set, feature):
#    return data_set[feature] == 0

# Create a binary variable to indicate whether a feautre is Missing
def create_is_null_feature(data_set, feature):
    return data_set[feature].isnull()

#raw_data['description_proximity_tfidf_is_zero'] = create_is_zero_feature(raw_data, 'description_proximity_tfidf')
raw_data['description_proximity_tfidf_is_null'] = create_is_null_feature(raw_data, 'description_proximity_tfidf')
raw_data.fillna(value={'description_proximity_tfidf': 0}, inplace=True)
#raw_data.fillna(value={'title_proximity_tfidf': 0}, inplace=True)

def one_hot_encode(data_set, feature, drop_col):
    res = pd.get_dummies(data_set, columns=[feature], prefix=feature)
    if drop_col:
        res.drop(labels=[drop_col], axis=1, inplace=True)
    return res

raw_data = one_hot_encode(raw_data, 'city_match', 'city_match_-1')


# Create user segmentations
raw_data.fillna(value={'u_id': 0}, inplace=True)

def recode_uid(data_set, colname='u_id'):
    """Recode the u_id column into integers"""
    col = data_set[colname].astype('category').cat.codes
    return data_set.assign(uid_recoded=col)

raw_data = recode_uid(raw_data)

def make_pivot_table(data_set, uid_colname, classid_colname):
    """Make a pivot table of u_id and class_id"""
    idx, col = uid_colname, classid_colname, 
    res = data_set[[idx, col]].pivot_table(index=idx, columns=col, aggfunc=len, 
                                               fill_value=0)
    return res

pivot_tbl = make_pivot_table(raw_data, 'uid_recoded', 'mgoc_id')

def make_usr_class_lookup(data_set):
    res = {}
    for i in range(len(data_set)):
        u_id, max_class_id = data_set.index[i], data_set.iloc[i].idxmax()
        if u_id not in res and u_id != 0:
            res[u_id] = max_class_id
    return res

def make_class_segments_lookup(data_set, n_cluster):
    res = {}
    kmeans = KMeans(n_clusters=n_cluster, random_state=10)
    data_set = data_set[data_set.index != 0]
    
    kmeans.fit(data_set)
    labels = kmeans.labels_
    
    for i, class_id in enumerate(data_set.index):
        if class_id not in res:
            res[class_id] = labels[i]
            
    return res

usr_cls_lookup = make_usr_class_lookup(pivot_tbl)

cls_segmt_lookup = make_class_segments_lookup(pivot_tbl.transpose(), 10)

def get_segment_id(uid, dic1=usr_cls_lookup, dic2=cls_segmt_lookup):
    """Return the segment id a user belongs to."""
    if uid != 0:
        max_class_id = dic1[uid]
        return dic2[max_class_id]
    else:
        return None
    
def add_segments_column(data_set, key_colname):
    return data_set.assign(usr_segments=data_set[key_colname].apply(lambda x: get_segment_id(x)))    

raw_data = add_segments_column(raw_data, 'uid_recoded')

raw_data = one_hot_encode(raw_data, 'usr_segments', None)


def add_squared_feature(data_set, features):
    for feature in features:
        colname = 'sqr_' + feature
        data_set[colname] = np.square(raw_data[feature])
    return data_set

raw_data = add_squared_feature(raw_data, ['description_proximity_tfidf', 
                                              'main_query_tfidf',
                                              'query_jl_score' ])


##############################################################################


#=======================#
# Create Train/Test Set #
#=======================#


def select_by_date(data_set, date_col, min_date, max_date):
    """ Select the data of which the search dates are in a given range."""
    dates = pd.to_datetime(data_set[date_col])
    boo = (dates <= max_date) & (dates >= min_date)
    res = data_set[boo].copy()
    res.drop(labels=[date_col], axis=1, inplace=True)
    return res

def get_features(data_set, exclude):
    res = [x for x in data_set.columns if x not in exclude]
    return res

train_dataset = select_by_date(raw_data, 'search_date_pacific', 
                                   '2018-01-21', '2018-01-26')
test_dataset = select_by_date(raw_data, 'search_date_pacific', 
                                  '2018-01-27', '2018-01-27')

features = get_features(train_dataset, ['apply', 'u_id', 'uid_recoded', 
                                            'mgoc_id', 'title_proximity_tfidf'])

# Create a X_train, Y_train, X_test and Y_test
train_data, train_labels = train_dataset[features], train_dataset['apply']
test_data, test_labels = test_dataset[features], test_dataset['apply']


##############################################################################

#================#
# Model Learning #
#================#


kfold = KFold(n_splits=5)  # Set for 5-Fold CV 
classifiers = {'Linear_Discriminant_Analysis': LinearDiscriminantAnalysis(),
               'Logistic_Regression': LogisticRegression(),
               'AdaBoost': AdaBoostClassifier(),
               'Random_Forest': RandomForestClassifier(),
               'GBDT': GradientBoostingClassifier()}

def get_cv_score(models, kfold, score, x, y):
    for model in models:
        print('Calculating CV score for the model, this may take some time.')
        res = cross_val_score(models[model], x, y, cv=kfold, scoring=score)
        cv_score = round(np.asarray(res).mean(), 5)
        print('CV score is', cv_score, 'for the', model, 'model.')

get_cv_score(classifiers, kfold, 'roc_auc', train_data, train_labels)

# Cross Validate to choose the best ratio for under-sampling
def under_sample_modeling(model, train_data, train_lables, ratio, kfold=kfold):
    res = []
    for train_idx, valid_idx in kfold.split(train_data):
        train_set = train_data.iloc[train_idx]
        train_label = train_labels.iloc[train_idx]
        
        valid_set = train_data.iloc[valid_idx]
        valid_label = train_labels.iloc[valid_idx]
        
        under_sampled_set = make_under_sample(train_set, train_label, 1, ratio)
        train_set = under_sampled_set[0]
        train_label = under_sampled_set[1]
        
        mod = model.fit(train_set, train_label)
        auc = roc_auc_score(valid_label, mod.predict_proba(valid_set)[:, 1])
        
        res.append(auc)
        
    return round(np.array(res).mean(), 5)
        
def print_under_sample_result(ratios, models):
    for model in models:
        for ratio in ratios:
            res = under_sample_modeling(model, train_data, train_labels, ratio)
            print('Ratio is', ratio, 'to 1, CV auc =', res)

#under_sample_ratios = [1, 3, 5, 8]  # Change me to test different ratios for under-sampling
#print_under_sample_result(under_sample_ratios, classifiers)


def make_under_sample(data, label, minor_class, sample_ratio): 
    """Under sampling the major class of a data set such that the 
    ratio of the major class to the minor is equal to the input ratio.
    """
    mjr_cls_data, mjr_cls_label = data[label != minor_class], label[label != minor_class]
    mnr_cls_data, mnr_cls_label = data[label == minor_class], label[label == minor_class]
    
    population = range(len(mjr_cls_data))
    sample_size = sample_ratio * len(mnr_cls_data)

    sample_idx = np.random.choice(population, sample_size, replace=False)
    
    new_mjr_cls_data = mjr_cls_data.iloc[sample_idx]
    new_mjr_cls_label = mjr_cls_label.iloc[sample_idx]
    
    new_data = pd.concat([new_mjr_cls_data, mnr_cls_data])
    new_label = pd.concat([new_mjr_cls_label, mnr_cls_label])
    
    shuffle_idx = np.random.choice(range(len(new_data)), len(new_data), replace=False)
    new_data = new_data.iloc[shuffle_idx]
    new_label = new_label.iloc[shuffle_idx]
    
    return [new_data, new_label]

sampled_dataset = make_under_sample(train_data, train_labels, 1, 3)
sampled_train_data = sampled_dataset[0]
sampled_train_labels = sampled_dataset[1]

get_cv_score(classifiers, kfold, 'roc_auc', sampled_train_data, sampled_train_labels)


# Train model on whole training set
def train_model(models, train_data, train_labels):
    res = {}
    for model in models:
        res[model] = (models[model].fit(train_data, train_labels))
    return res

#trained_classifiers = train_model(classifiers, train_data, train_labels)
trained_classifiers = train_model(classifiers, sampled_train_data, sampled_train_labels)

# Predict results on test set
def get_predicted_result(models, test_data, test_labels):
    res = {}
    for model in models:
        score = roc_auc_score(test_labels, models[model].predict_proba(test_data)[:, 1])
        #score = roc_auc_score(test_labels, models[model].predict(test_data))
        res[model] = round(score, 5)
        print('Test score is ', res[model], ' for ', model, ' model.' )
    return res
       
predicted_auc = get_predicted_result(trained_classifiers, test_data, test_labels) 

# Get feature importance
def get_feature_importance(classifier):
    res = {}
    for i, feature_imp in enumerate(classifier.feature_importances_):
        res[features[i]] = feature_imp
    return res

feature_imp = get_feature_importance(trained_classifiers['GBDT'])
feature_imp = pd.DataFrame.from_dict(feature_imp, orient='index')
feature_imp = feature_imp.sort_values(0, ascending=False)

# Plot feature importance
sns.barplot(y=feature_imp.index, x = feature_imp[0])    


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

print(time.ctime())


#plot_learning_curve(LogisticRegression(), 'rf', train_data, train_labels, ylim=(0.7, 1.01), cv=kfold, n_jobs=1)

# Compute ROC curve and ROC area for each class
#fpr, tpr, thresholds = roc_curve(test_labels, y_score[:, 1], pos_label=1)

#plt.figure()
#lw = 2
#plt.plot(fpr, tpr, color='darkorange',
#         lw=lw)
#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic example')
#plt.legend(loc="lower right")
#plt.show()


# Tune Hyperparameters
#RFC = RandomForestClassifier()
#
### Search grid for optimal parameters
#rf_param_grid = {"max_depth": [None],
#                 "max_features": [1, 3, 10],
#                 "min_samples_split": [2, 3, 10],
#                 "min_samples_leaf": [1, 3, 10],
#                 "n_estimators" :[100,300],
#                 "criterion": ["gini"]}
#gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="roc_auc", n_jobs= 4, verbose = 1)
#
#gsRFC.fit(sampled_train_data,sampled_train_labels)
#
#RFC_best = gsRFC.best_estimator_
#
## Best score
#gsRFC.best_score_
#
## Adaboost
#DTC = DecisionTreeClassifier()
#ada = AdaBoostClassifier(DTC, random_state=7)
#ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
#                  "base_estimator__splitter" :   ["best", "random"],
#                  "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}
#ada_grid = GridSearchCV(ada,
#                        param_grid=ada_param_grid,
#                        scoring="roc_auc")
#ada_grid.fit(sampled_train_data, sampled_train_labels)
#ada_best = ada_grid.best_estimator_

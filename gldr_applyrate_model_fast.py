"""
Job Application Rate Predicted Modeling
@author: Hongfei Yang <yangh4@seattleu.edu>
03/27/2018
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


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

raw_data = get_raw_data(filepath)
raw_data = shuffle_data(raw_data)  # Shuffle to prevent pre-existing order 

raw_data.fillna(value={'u_id': 0}, inplace=True)
raw_data.fillna(value={'city_match': -1}, inplace=True)
raw_data['city_match'] = raw_data['city_match'].astype('int')


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


def log_transform_features(data_set, features):
    for feature in features:
        data_set[feature] = data_set[feature].map(lambda x: np.log(x + 0.01) if x > 0 else x)
    return data_set

log_features = ['description_proximity_tfidf', 'query_jl_score', 
                    'query_title_score', 'job_age_days']

raw_data = log_transform_features(raw_data, log_features)


def create_is_null_feature(data_set, feature):
    return data_set[feature].isnull()

raw_data['description_proximity_tfidf_is_null'] = create_is_null_feature(raw_data, 'description_proximity_tfidf')
raw_data = raw_data.fillna(value={'description_proximity_tfidf': 0}, inplace=True)


def add_squared_feature(data_set, features):
    for feature in features:
        colname = 'sqr_' + feature
        data_set[colname] = np.square(raw_data[feature])
    return data_set

raw_data = add_squared_feature(raw_data, ['description_proximity_tfidf', 
                                              'main_query_tfidf',
                                              'query_jl_score' ])

    
def one_hot_encode(data_set, feature, drop_col):
    res = pd.get_dummies(data_set, columns=[feature], prefix=feature)
    if drop_col:
        res.drop(labels=[drop_col], axis=1, inplace=True)
    return res

raw_data = one_hot_encode(raw_data, 'city_match', 'city_match_-1')
raw_data = one_hot_encode(raw_data, 'usr_segments', None)


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
    
train_data, train_labels = train_dataset[features], train_dataset['apply']
test_data, test_labels = test_dataset[features], test_dataset['apply']


kfold = KFold(n_splits=5)  # Set for 5-Fold CV 
classifiers = {'Linear_Discriminant_Analysis': LinearDiscriminantAnalysis(), 
               'Gradient_Boosted_Decision_Tree': GradientBoostingClassifier()}

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


def get_cv_score(models, kfold, score, x, y):
    for model in models:
        print('Calculating CV score for the model, this may take some time.')
        res = cross_val_score(models[model], x, y, cv=kfold, scoring=score)
        cv_score = round(np.asarray(res).mean(), 5)
        print('CV score is', cv_score, 'for the', model, 'model.')

get_cv_score(classifiers, kfold, 'roc_auc', sampled_train_data, sampled_train_labels)


def train_model(models, train_data, train_labels):
    res = {}
    for model in models:
        res[model] = (models[model].fit(train_data, train_labels))
    return res

trained_classifiers = train_model(classifiers, sampled_train_data, sampled_train_labels)


def get_predicted_result(models, test_data, test_labels):
    res = {}
    for model in models:
        score = roc_auc_score(test_labels, models[model].predict_proba(test_data)[:, 1])
        #score = roc_auc_score(test_labels, models[model].predict(test_data))
        res[model] = round(score, 5)
        print('Test score is ', res[model], ' for ', model, ' model.' )
    return res
       
predicted_auc = get_predicted_result(trained_classifiers, test_data, test_labels) 
 
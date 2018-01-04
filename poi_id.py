#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedShuffleSplit, ParameterGrid, cross_val_score


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi','salary',  'total_payments',
             'expenses', 'exercised_stock_options', 'other',
            'long_term_incentive', 'to_messages', 'from_poi_to_this_person', 'from_messages',
             'from_this_person_to_poi', 'shared_receipt_with_poi','fraction_stock_incentive']

n = len(features_list)

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Task 2: Remove outliers
data_dict.pop('TOTAL', 0 )
data_dict.pop('LOCKHART EUGENE E', 0)

### Task 3: Create new feature(s)
def computeFraction(nume, demo):
    """ given nume (numerator) and demo (denominator),
        return the fraction
    """

    fraction = 0.
    if nume != 'NaN' and demo != 'NaN':
        fraction = float(nume) / demo
    else:
        pass
    return fraction

for name in data_dict:

    data_point = data_dict[name]

    # Create feature fraction_stock_incentive
    fraction_stock_incentive = computeFraction(data_point['exercised_stock_options'], data_point['long_term_incentive'])
    data_point['fraction_stock_incentive'] = fraction_stock_incentive

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


# Example starting point. Try investigating other evaluation techniques!

# using StratifiedShuffleSplit to split data
sss = StratifiedShuffleSplit(test_size=0.3, random_state=42)
for train_index, test_index in sss.split(features, labels):
    features_train = []
    features_test  = []
    labels_train   = []
    labels_test    = []
    for ii in train_index:
        features_train.append( features[ii] )
        labels_train.append( labels[ii] )
    for jj in test_index:
        features_test.append( features[jj] )
        labels_test.append( labels[jj] )

# parameter tuning using f1 score
best_score = 0
best_param = {}
param_grid = {'C': [5, 10, 20, 40 , 80, 160, 320],
'penalty': ['l1', 'l2']
}
for param in list(ParameterGrid(param_grid)):
    clf = LogisticRegression(**param)
    score = cross_val_score(clf, features_train, labels_train, scoring='f1').mean()
    if score > best_score:
        best_score = score
        best_param = param
print best_param
print best_score

# fit final model
clf = LogisticRegression(**best_param)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

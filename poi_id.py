#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split,  ParameterGrid
from sklearn.metrics import f1_score


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary',  'total_payments', 'bonus','restricted_stock',
            'total_stock_value', 'expenses', 'exercised_stock_options', 'other',
            'long_term_incentive', 'to_messages', 'from_poi_to_this_person', 'from_messages',
             'from_this_person_to_poi', 'shared_receipt_with_poi',
                 'fraction_from_poi', 'fraction_to_poi', 'fraction_stock_incentive',
                'ratio_salary_bonus', 'ratio_salary_restricted_stock']

n = len(features_list)

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Task 2: Remove outliers
data_dict.pop('TOTAL', 0 )


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

    # Create feature fraction_from_poi
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    data_point["fraction_from_poi"] = fraction_from_poi

    # Create feature fraction_to_poi
    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    data_point["fraction_to_poi"] = fraction_to_poi

    # Create feature fraction_stock_incentive
    fraction_stock_incentive = computeFraction(data_point['exercised_stock_options'], data_point['long_term_incentive'])
    data_point['fraction_stock_incentive'] = fraction_stock_incentive

    # Create feature ratio_salary_bonus
    ratio_salary_bonus = computeFraction(data_point['bonus'], data_point['salary'])
    data_point['ratio_salary_bonus'] = ratio_salary_bonus

    # Create feature ratio_salary_restricted_stock
    ratio_salary_restricted_stock = computeFraction(data_point['restricted_stock'], data_point['salary'])
    data_point['ratio_salary_restricted_stock'] = ratio_salary_restricted_stock

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
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# parameter tuning using f1 score
best_score = 0
best_param = {}
param_grid = {'C': [10, 40 , 80, 160, 320],
             'penalty': ['l1', 'l2']}
for param in list(ParameterGrid(param_grid)):
    clf = LogisticRegression(**param)
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    score = f1_score(labels_test, pred)
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

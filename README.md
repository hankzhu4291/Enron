# Enron
Identify Fraud from Enron Email using machine learning

## Project Background
In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives.
In this project, the goal is to build a classifier to predict if a person is Person of Interest in the fraud case. Machine Learning could be help to accomplish the goal since it can learn the pattern from training data given and with the pattern it would predict Person of Interest. Features are required for Machine Learning. In this project, features involve Enron email information and financial data. I used all 23 features at the beginning as exploration. 

## Data Exploration:
  - Total data points: 146
  - Poi: 18
  - Non-poi: 127
  - Number of features(final): 19
  - Features with most missing values: 
  loan_advances(142), director_fees(129), restricted_stock_deferred(128), deferral_payments(108), deferred_income(97)

## Final Features used:
'salary', 'total_payments', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'to_messages',
 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi', 
'fraction_stock_incentive'

## Final algorithm used
Logistic Regression('penalty'='l1', 'C'=160)

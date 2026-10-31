# Bank Marketing Dataset – Data Dictionary

## Dataset Overview

- Dataset: Bank Marketing
- Source: UCI Bank Marketing dataset documentation
- Records: 4,521
- Columns: 17
- Target variable: y
- Problem type: Binary Classification

## Feature Descriptions

| Feature | Description | Data Type | Role |
|---|---|---|---|
| age | Age of the bank client | Numerical | Predictor |
| job | Type of job | Categorical | Predictor |
| marital | Marital status; divorced includes divorced or widowed | Categorical | Predictor |
| education | Education level | Categorical | Predictor |
| default | Whether the client has credit in default | Categorical | Predictor |
| balance | Average yearly balance in euros | Numerical | Predictor |
| housing | Whether the client has a housing loan | Categorical | Predictor |
| loan | Whether the client has a personal loan | Categorical | Predictor |
| contact | Communication type used for the last contact | Categorical | Predictor |
| day | Last contact day of the month | Numerical | Predictor |
| month | Last contact month of the year | Categorical | Predictor |
| duration | Duration of the last contact in seconds | Numerical | Predictor |
| campaign | Number of contacts during the current campaign, including the last contact | Numerical | Predictor |
| pdays | Days since the client was last contacted in a previous campaign; -1 means not previously contacted | Numerical | Predictor |
| previous | Number of contacts before the current campaign | Numerical | Predictor |
| poutcome | Outcome of the previous marketing campaign | Categorical | Predictor |
| y | Whether the client subscribed to a term deposit (yes/no) | Categorical | Target |

## Feature Categories

### Numerical Features

1. age
2. balance
3. day
4. duration
5. campaign
6. pdays
7. previous

Total: 7

### Categorical Predictors

1. job
2. marital
3. education
4. default
5. housing
6. loan
7. contact
8. month
9. poutcome

Total: 9

### Target Variable

- y: yes / no

## Important Categorical Values

- job: admin., unknown, unemployed, management, housemaid,
  entrepreneur, student, blue-collar, self-employed,
  retired, technician, services
- marital: married, divorced, single
- education: unknown, secondary, primary, tertiary
- default: yes, no
- housing: yes, no
- loan: yes, no
- contact: unknown, telephone, cellular
- month: jan, feb, mar, ..., nov, dec
- poutcome: unknown, other, failure, success

## Important Dataset Considerations

1. The value -1 in pdays indicates that the client was not
   previously contacted during an earlier campaign.
2. The category unknown is an explicit categorical value
   in several features; it is not automatically a missing value.
3. The duration feature represents the duration of the last
   contact in seconds. Since it is only known after the contact
   ends, using it to predict subscription before a call begins
   may cause data leakage. Its inclusion must be carefully
   evaluated for prospective prediction.
4. The target variable y contains two classes: yes and no.
5. The target distribution is imbalanced:
   - no: 4,000 records (88.48%)
   - yes: 521 records (11.52%)

## Data Quality Summary

- Total records: 4,521
- Total columns: 17
- Missing values: 0
- Exact duplicate records: 0

## Original Dataset Citation

Moro, S., Laureano, R. and Cortez, P. (2011).
Using Data Mining for Bank Direct Marketing: An Application
of the CRISP-DM Methodology.

Proceedings of the European Simulation and Modelling
Conference – ESM'2011, pp. 117–121.

Dataset documentation: data/raw/bank-names.txt

---

Project: Bank Marketing Subscription Prediction
Phase: 2 – Dataset Collection and Understanding
Author: Ajeet Sherkar
Degree: B.Tech – Artificial Intelligence and Machine Learning

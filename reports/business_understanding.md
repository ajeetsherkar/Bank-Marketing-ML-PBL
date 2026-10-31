# Bank Marketing Subscription Prediction
## Phase 1: Business Understanding

### 1. Project Overview

This project aims to develop a Machine Learning system that predicts
whether a bank customer is likely to subscribe to a term deposit
during a marketing campaign.

The system will analyze available customer and campaign information
to support better customer targeting and marketing decisions.

### 2. Business Problem

Banks conduct marketing campaigns to promote term deposits.
However, not every customer contacted is interested in subscribing.

Contacting customers who are unlikely to subscribe can waste
marketing resources, increase campaign costs and reduce efficiency.

The business problem is to identify potential customers who are
more likely to subscribe to a term deposit.

A predictive system can help the bank prioritize customer contacts
and improve marketing campaign effectiveness.

### 3. Business Objectives

1. Identify customers who are more likely to subscribe.
2. Improve customer targeting during marketing campaigns.
3. Reduce unnecessary and ineffective marketing calls.
4. Allocate marketing resources more efficiently.
5. Improve overall campaign performance.

### 4. Stakeholder Analysis

| Stakeholder | Interest |
|---|---|
| Bank management | Campaign performance |
| Marketing team | Customer targeting |
| Sales employees | Potential customer identification |
| Customers | Relevant financial offers |
| Data science team | Prediction and model monitoring |

### 5. Machine Learning Problem Definition

Problem Type: Supervised Machine Learning

Learning Task: Binary Classification

Objective: Predict whether a customer will subscribe to a
bank term deposit.

Target Variable: y

Possible Target Classes:
- yes: Customer subscribed to a term deposit.
- no: Customer did not subscribe to a term deposit.

The model will learn patterns from historical customer and
marketing campaign data to predict the target class.

### 6. Key Performance Indicators (KPIs)

#### 6.1 Model Evaluation KPIs

- Precision: Measures how many predicted positive customers
  actually subscribed.
- Recall: Measures how many actual subscribers were correctly
  identified by the model.
- F1-score: Harmonic mean of precision and recall.
- ROC-AUC: Measures the model's ability to distinguish between
  positive and negative classes across classification thresholds.
- PR-AUC: Summarizes the precision-recall trade-off.

#### 6.2 Business KPI

Campaign Conversion Rate:

Campaign Conversion Rate =
(Number of successful subscriptions / Number of contacted customers)
* 100

This metric measures the proportion of contacted customers who
successfully subscribe to a term deposit.

### 7. Project Scope

The project will include:

1. Dataset understanding and exploration.
2. Data cleaning and preprocessing.
3. Exploratory Data Analysis (EDA).
4. Feature engineering and selection.
5. Development of classification models.
6. Model evaluation and comparison.
7. Business insights and interpretation.
8. Model packaging and reproducibility.
9. Streamlit application development.
10. Documentation and final presentation.

### 8. Expected Business Outcome

The expected outcome is a machine learning solution that helps
the bank identify customers who are more likely to subscribe
to a term deposit.

The solution aims to support improved customer targeting,
more efficient marketing resource allocation and better
campaign decision-making.

### 9. Limitations and Considerations

- Predictions are decision-support outputs, not guarantees
  of customer subscription.
- Model performance depends on historical data quality.
- Customer privacy and responsible data usage must be considered.
- Business decisions should consider model predictions alongside
  other relevant information.

---

**Project:** Bank Marketing Subscription Prediction  
**Phase:** 1 - Business Understanding  
**Author:** Ajeet Sherkar  
**Degree:** B.Tech - Artificial Intelligence and Machine Learning

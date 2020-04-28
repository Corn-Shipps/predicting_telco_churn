# Why are our customers churning?.

To answer this question, our data science team contructed a model to predict customer churn using classification techniques.

# Deliverables

### The team was asked to provide:
> - A detailed report answering the question, _"Why are our customers churning"?_ which includes analysis depicting who is most likely to churn along with insight on why they may be choosing to do so
> - A .csv file with uniquely identifiable customer id, the probability and predictioin of churn using our best model
> - Slide deck describing the metrics of the chosen model along with a presentation to explain how we arrived at our decision
> - `telco_churn.customers.py` file that will take in a new dataset, (in the exact same form of the one you acquired from telco_churn.customers) and perform all the transformations necessary to run the model we have developed on this new dataset to provide probabilities and predictions.

# Executive Summary
### Key Takeaways

- The most important features for determining customer churn include:
> 1. Contract type 
> 2. Tenure of three months or less
> 3. Monthly charges
> 4. Automatic payment
> 5. Senior Citizen status

- The question about why our customers are churning is answered in two parts: Who is most likely to churn and what is it about that group that may be motivating them to churn
- Essentially, customers with a month-to-month, not using automatic payments tend to churn within the first three months and many of them are classifed as senior citizens. 
- A look at thier monthly charges revealed, that they pay on average 13 dollars more per month than customers that did not churn

#### _Further analysis recommended:_

- Determine which services those customers are using at a higher rate


# Data Dictionary
|          Feature         |    Count/data type    | Description                                                                                                                                                                                                |
|:------------------------:|:---------------------:|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| customer_id              | 7043 non-null object  | Unique identifier of each customer at the company                                                                                                                                                          |
| gender                   | 7043non-null int64    | Gender of the customer. 0 indicates female,1 indicates male                                                                                                                                                |
| senior_citizen           | 7043 non-null int64   | Indicates if the customer is classified as a senior citizen. 1 indicates is a senior citizen, 0 indicates is not a senior citizen                                                                          |
| partner                  | 7043 non-null int64   | Indicates if the customer has a partner. 1 indicates has a partner, 0 indicates does not have a partner                                                                                                    |
| dependents               | 7043 non-null int64   | Indicates if the customer has dependents. 1 indicates has dependents, 0 indicates does not have dependents                                                                                                 |
| tenure                   | 7043 non-null int64   | Indicates the number of months the customer has being paying for services at the company                                                                                                                   |
| monthly_charges          | 7043 non-null float64 | Indicates how much the customer pays per month                                                                                                                                                             |
| total_charges            | 7043 non-null float64 | Indicates how much the customer has paid to date since becoming a customer                                                                                                                                 |
| phone_service            | 7043 non-null int64   | Indicates if the customer pays for phone service. 1 indicates does pay for phone service, 0 indicates does pay for phone service                                                                           |
| multiple_lines           | 7043 non-null int64   | Indicates if the customer pays for multiple phone lines. 2 indicates does pay for multiple phone lines, 1 indicates does pay for multiple lines, 0 indicates has no phone service                          |
| online_security          | 7043 non-null int64   | Indicates if the customer pays for online security. 2 indicates does pay for online security, 1 indicates does not pay for online security, 0 indicates does not have internet service                     |
| online_backup            | 7043 non-null int64   | Indicates if the customer pays for online backup service. 2 indicates does pay for online backup service, 1 indicates does not pay for online backup, 0 indicates does not have internet service           |
| device_protection        | 7043 non-null int64   | Indicates if the customer pays for device protection (router?). 2 indicates does not pay for device protection, 1 indicates does not pay for device protection, 0 indicates does not have internet service |
| tech_support             | 7043 non-null int64   | Indicates if the customer pays for technical support. 2 indicates does pay for online technical support, 1 indicates does not pay for online technical support 0 indicates does not have internet service  |
| streaming_tv             | 7043 non-null int64   | Indicates if the customer pays to stream tv online. 2 indicates does pay to stream tv online, 1 indicates does not pay to stream tv online, 0 indicates does not have internet service                     |
| streaming_movies         | 7043 non-null int64   | Indicates if the customer pays to stream movies online. 2 indicates does pay to stream movies online, 1 indicates does not pay to stream movies online, 0 indicates does not have internet service         |
| paperless_billing        | 7043non-null int64    | Indicates if the customer uses paperless billing. 1 indicates they do use paperless billing, 0 indicates they do not use paperless billing                                                                 |
| contract_type_id         | 7043 non-null int64   | Indicates the type of contract the customer has signed. 1 indicates month-to-month contract, 2 indicates one year contract, 3 indicates two year contract                                                  |
| payment_type_id          | 7043 non-null int64   | Indicates the type of payment method the customer uses. 1 indicates electronic check, 2 indicates mailed check, 3 indicates bank transfer (automatic), 4 indicates credit card (automatic)                 |
| internet_service_type_id | 7043 non-null int64   | Indicates the type of internet service the customer has, 1 is DLS, 2 is fiber optic, 3 is none                                                                                                             |
| churn                    | 7043 non-null int64   | Indicates if the customer stopped using the company's product or service. 1 indicates churn, 2 indicates no churn                                                                                          |
| automatic_payment        | 7043 non-null bool    | Indicates if the customer participates in an automatic form of payment. True indicates they use a form of automatic payment, False indicates they do not                                                   |
| monthly_charges_scaled   | 7043 non-null float64 | Monthly charges column scaled using the MinMax Scaler from Sklearn                                                                                                                                         |
| tenure_3_or_less         | 7043 non-null bool    | Indicates if the customer has been with the company for 3 months or less                                                                                                                                   |



# Pipeline

## Aquisition

## Preparation

### Hypothesis testing

## Exploration

## Feature Selection

## Modeling

## Evaluation


# Techinical Skills and Programs used

- Python (including libraries)
- SQL
- Hypothesis testing
- Logistic Regression, Decision Tree, Random Forest, K Nearest Neighbors models
- Canva

To view the presentation, please [click here](https://www.canva.com/design/DAD6kLeGec8/JUm9G7dao7PNw8tl8H3pqA/view?utm_content=DAD6kLeGec8&utm_campaign=designshare&utm_medium=link&utm_source=sharebutton)
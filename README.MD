# To Churn or Not to Churn.

The purpose of this project is to build a model which predicts customer churn using classification strategies.




# Data Dictionary
|          Feature         |    Count/data type    | Description                                                                                                                                                                                                |
|:------------------------:|:---------------------:|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| customer_id              | 5026 non-null object  | Unique identifier of each customer at the company                                                                                                                                                          |
| gender                   | 5026 non-null int64   | Gender of the customer. 0 indicates female,1 indicates male                                                                                                                                                |
| senior_citizen           | 5026 non-null int64   | Indicates if the customer is classified as a senior citizen. 1 indicates is a senior citizen, 0 indicates is not a senior citizen                                                                          |
| partner                  | 5026 non-null int64   | Indicates if the customer has a partner. 1 indicates has a partner, 0 indicates does not have a partner                                                                                                    |
| dependents               | 5026 non-null int64   | Indicates if the customer has dependents. 1 indicates has dependents, 0 indicates does not have dependents                                                                                                 |
| tenure                   | 5026 non-null int64   | Indicates the number of months the customer has being paying for services at the company                                                                                                                   |
| monthly_charges          | 5026 non-null float64 | Indicates how much the customer pays per month                                                                                                                                                             |
| total_charges            | 5026 non-null float64 | Indicates how much the customer has paid to date since becoming a customer                                                                                                                                 |
| phone_service            | 5026 non-null int64   | Indicates if the customer pays for phone service. 1 indicates does pay for phone service, 0 indicates does pay for phone service                                                                           |
| multiple_lines           | 5026 non-null int64   | Indicates if the customer pays for multiple phone lines. 2 indicates does pay for multiple phone lines, 1 indicates does pay for multiple lines, 0 indicates has no phone service                          |
| online_security          | 5026 non-null int64   | Indicates if the customer pays for online security. 2 indicates does pay for online security, 1 indicates does not pay for online security, 0 indicates does not have internet service                     |
| online_backup            | 5026 non-null int64   | Indicates if the customer pays for online backup service. 2 indicates does pay for online backup service, 1 indicates does not pay for online backup, 0 indicates does not have internet service           |
| device_protection        | 5026 non-null int64   | Indicates if the customer pays for device protection (router?). 2 indicates does not pay for device protection, 1 indicates does not pay for device protection, 0 indicates does not have internet service |
| tech_support             | 5026 non-null int64   | Indicates if the customer pays for technical support. 2 indicates does pay for online technical support, 1 indicates does not pay for online technical support 0 indicates does not have internet service  |
| streaming_tv             | 5026 non-null int64   | Indicates if the customer pays to stream tv online. 2 indicates does pay to stream tv online, 1 indicates does not pay to stream tv online, 0 indicates does not have internet service                     |
| streaming_movies         | 5026 non-null int64   | Indicates if the customer pays to stream movies online. 2 indicates does pay to stream movies online, 1 indicates does not pay to stream movies online, 0 indicates does not have internet service         |
| paperless_billing        | 5026 non-null int64   | Indicates if the customer uses paperless billing. 1 indicates they do use paperless billing, 0 indicates they do not use paperless billing                                                                 |
| contract_type_id         | 5026 non-null int64   | Indicates the type of contract the customer has signed. 1 indicates month-to-month contract, 2 indicates one year contract, 3 indicates two year contract                                                  |
| payment_type_id          | 5026 non-null int64   | Indicates the type of payment method the customer uses. 1 indicates electronic check, 2 indicates mailed check, 3 indicates bank transfer (automatic), 4 indicates credit card (automatic)                 |
| internet_service_type_id | 5026 non-null int64   | Indicates the type of internet service the customer has, 1 is DLS, 2 is fiber optic, 3 is none                                                                                                             |
| churn                    | 5026 non-null int64   | Indicates if the customer stopped using the company's product or service. 1 indicates churn, 2 indicates no churn                                                                                          |





To view the presentation, please click here(https://www.canva.com/design/DAD6kLeGec8/JUm9G7dao7PNw8tl8H3pqA/view?utm_content=DAD6kLeGec8&utm_campaign=designshare&utm_medium=link&utm_source=sharebutton)
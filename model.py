import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier



# Programmatically get all the model columns
# .loc[starting_row:ending_row, starting_column:ending_column]
def model_accuracy(df):
    '''
    This function takes in all the model predictions and baseline predictions 
    and returns a dataframe with each model's accuracy in order
    '''
    models = df.loc[:, "baseline":"Random_Forest"].columns.tolist()
    models
    
    output = []
    for model in models:
        output.append({
            "model": model,
            "accuracy%": ((df[model] == df.actual).mean())*100
        })
    
    
    accuracy = pd.DataFrame(output)
    accuracy = accuracy.sort_values(by="accuracy%", ascending=False)
    return accuracy



# programatically create logit model
def logit_metrics(train, test):
    '''
    Takes in a train and test dataframe
    Defines the X and y train and test variables
    Creates a model
    Fits the model
    Makes predictions using the model
    Returns a classification report for the model
    '''
# Create X and y variables for each dataset
    X_train = train[['contract_type_id', 'senior_citizen',  'tenure_3_or_less', 'monthly_charges_scaled']]
    y_train = train[['churn']]
    X_test = test[['contract_type_id', 'senior_citizen',  'tenure_3_or_less', 'monthly_charges_scaled']]
    y_test = test[['churn']]
    
    # Create and fit the model; set a random seed for reproducibility
    logit = LogisticRegression(random_state = 123)
    logit.fit(X_train, y_train)
    
    # Use the model to make predictions
    y_pred = logit.predict(X_test)
    y_pred[:10]
    
    # Print a classification report
    target_names = ["did not churn", "churn"]
    print(classification_report(y_test, y_pred, target_names = target_names))


# programatically create decision_tree model
def decision_tree_metrics(train, test):
    '''
    Takes in a train and test dataframe
    Defines the X and y train and test variables
    Creates a model
    Fits the model
    Makes predictions using the model
    Returns a classification report for the model
    '''
# Create X and y variables for each dataset
    X_train = train[['contract_type_id', 'senior_citizen',  'tenure_3_or_less', 'monthly_charges_scaled']]
    y_train = train[['churn']]
    X_test = test[['contract_type_id', 'senior_citizen',  'tenure_3_or_less', 'monthly_charges_scaled']]
    y_test = test[['churn']]
    
    # Create and fit the model; set a random seed for reproducibility
    clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=123)
    clf.fit(X_train, y_train)
    
    # Use the model to make predictions
    y_pred = clf.predict(X_test)
    y_pred[:10]
    
    # Print a classification report
    target_names = ["did not churn", "churn"]
    print(classification_report(y_test, y_pred, target_names = target_names))
    
    
    
# Programmatically create K-Nearest Neighbors model
def knn_metrics(train, test):
    '''
    Takes in a train and test dataframe
    Defines the X and y train and test variables
    Creates a model
    Fits the model
    Makes predictions using the model
    Returns a classification report for the model
    '''
# Create X and y variables for each dataset
    X_train = train[['contract_type_id', 'senior_citizen',  'tenure_3_or_less', 'monthly_charges_scaled']]
    y_train = train[['churn']]
    X_test = test[['contract_type_id', 'senior_citizen',  'tenure_3_or_less', 'monthly_charges_scaled']]
    y_test = test[['churn']]
    
    # Create and fit the model; set a random seed for reproducibility
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X_train, y_train)
    
    # Use the model to make predictions
    y_pred = knn.predict(X_test)
    y_pred[:10]
    
    # Print a classification report
    target_names = ["did not churn", "churn"]
    print(classification_report(y_test, y_pred, target_names = target_names))
    
    

# Programmatically create Random Forest model
def rf_metrics(train, test):
    '''
    Takes in a train and test dataframe
    Defines the X and y train and test variables
    Creates a model
    Fits the model
    Makes predictions using the model
    Returns a classification report for the model
    '''
# Create X and y variables for each dataset
    X_train = train[['contract_type_id', 'senior_citizen',  'tenure_3_or_less', 'monthly_charges_scaled']]
    y_train = train[['churn']]
    X_test = test[['contract_type_id', 'senior_citizen',  'tenure_3_or_less', 'monthly_charges_scaled']]
    y_test = test[['churn']]
    
    # Create and fit the model; set a random seed for reproducibility
    rf = RandomForestClassifier(bootstrap=True, 
                            class_weight=None, 
                            criterion='gini',
                            min_samples_leaf=3,
                            n_estimators=100,
                            max_depth=3, 
                            random_state=123)

    rf.fit(X_train, y_train)
    # Use the model to make predictions
    y_pred = rf.predict(X_test)
    y_pred[:10]
    
    # Print a classification report
    target_names = ["did not churn", "churn"]
    print(classification_report(y_test, y_pred, target_names = target_names))
    
    
def predictions_csv(df):
    # create predicions dataframe by adding in customer_id from original df
    predictions = pd.DataFrame(df.customer_id)

    # Split the data between train and test
    train, test = train_test_split(df, random_state=123, train_size=.80)
    
    # Define X, X_train, y_train variables
    X = df[['contract_type_id', 'senior_citizen',  'tenure_3_or_less', 'monthly_charges_scaled', 'payment_type_id']]
    X_train = train[['contract_type_id', 'senior_citizen',  'tenure_3_or_less', 'monthly_charges_scaled', 'payment_type_id']]
    y_train = train[['churn']]

    # Create model and fit it to the training set
    logit = LogisticRegression(random_state = 123)
    logit.fit(X_train, y_train)

    # Predict probability of churn
    y_pred_proba = logit.predict_proba(X)

    # Predict churn
    y_pred_logit = logit.predict(X)

    # Add probabilities and predictions to dataframe
    predictions = pd.DataFrame(
    {'Customer_ID': df.customer_id,
    'Probability_of_churn': y_pred_proba[:,1],
    'Probability_of_not_churning': y_pred_proba[:,0],
    'Churn_Prediction': y_pred_logit})

    # Create csv file

    return predictions.to_csv('telco_churn_predictions.csv')

 
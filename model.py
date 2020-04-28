import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier



# Programmatically get all the model columns
# .loc[starting_row:ending_row, starting_column:ending_column]
def model_accuracy(df):
    models = df.loc[:, "baseline":"Decision_Tree"].columns.tolist()
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
    
    
    
    
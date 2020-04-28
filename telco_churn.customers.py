import pandas as pd
import numpy as np
import acquire
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def wrangle_telco():
    df = acquire.get_telco_data()
    df.tenure.replace(0, 1, inplace=True)
    df.total_charges = df.total_charges.str.strip()
    df.total_charges.replace('', df.monthly_charges, inplace=True)
    df.total_charges = df.total_charges.astype(float)
    df['automatic_payment'] = ((df['payment_type_id'] == 3) | (df['payment_type_id'] == 4))
    scaler = MinMaxScaler()
    df['monthly_charges_scaled'] = scaler.fit_transform(df['monthly_charges'].values.reshape(-1,1))
    df['tenure_3_or_less'] = df['tenure']<=3
    return df



def encode_columns():
    train, test = train_test_split(df, random_state=123, train_size=.80)

    encoder = LabelEncoder()

    encode_list = [
    'gender', 'partner', 'dependents', 'phone_service'
    , 'multiple_lines', 'online_security', 'online_backup'
    , 'device_protection', 'tech_support'
    , 'streaming_movies', 'streaming_tv', 'paperless_billing', 'churn'
    ]
             
    for e in encode_list:
        train[e] = encoder.fit_transform(train[e])
        test[e] = encoder.transform(test[e])
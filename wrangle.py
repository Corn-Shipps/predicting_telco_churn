import pandas as pd
import numpy as np
import acquire
from sklearn.preprocessing import MinMaxScaler


def wrangle_telco():
    df = acquire.get_telco_data()
    df.tenure.replace(0, 1, inplace=True)
    df.total_charges = df.total_charges.str.strip()
    df.total_charges.replace('', df.monthly_charges, inplace=True)
    df.total_charges = df.total_charges.astype(float)
    df['automatic_payment'] = ((df['payment_type_id'] == 3) | (df['payment_type_id'] == 4))
    scaler = MinMaxScaler()
    df['total_charges_scaled'] = scaler.fit_transform(df['total_charges'].values.reshape(-1,1))
    return df
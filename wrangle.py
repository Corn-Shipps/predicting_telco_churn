import pandas as pd
import numpy as np
from acquire import get_telco_data



def wrangle_telco():
    df = get_telco_data()
    df.tenure.replace(0, 1, inplace=True)
    df.total_charges = df.total_charges.str.strip()
    df.total_charges.replace('', df.monthly_charges, inplace=True)
    df.total_charges = df.total_charges.astype(float)
    return df
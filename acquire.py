import pandas as pd 
import numpy as np 
from env import user, password, host

def get_db_url(dbname) -> str:
    url = 'mysql+pymysql://{}:{}@{}/{}'
    return url.format(user, password, host, dbname)


def get_telco_data():
    query = '''
    SELECT customers.customer_id, gender, senior_citizen, partner, dependents, tenure, monthly_charges, total_charges, phone_service, multiple_lines, online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies, paperless_billing, contract_types.contract_type, payment_types.payment_type,internet_service_types.internet_service_type, churn
FROM customers
LEFT JOIN contract_types ON customers.contract_type_id=contract_types.contract_type_id
LEFT JOIN internet_service_types ON customers.internet_service_type_id=internet_service_types.internet_service_type_id
LEFT JOIN payment_types ON customers.payment_type_id=payment_types.payment_type_id
    '''
    df = pd.read_sql(query, get_db_url('telco_churn'))
    return df 
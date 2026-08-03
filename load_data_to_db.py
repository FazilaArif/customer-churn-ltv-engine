import pandas as pd
from sqlalchemy import create_engine

# PostgreSQL connection
engine = create_engine(
    "postgresql://postgres:3008@localhost:5432/customer_churn"
)

# Load CSV file
df = pd.read_csv("Data/Telco_customer_churn.csv")

print("CSV loaded successfully")
print(df.head())

# Upload dataframe to PostgreSQL
df.to_sql(
    "customer_churn_data",
    engine,
    if_exists="replace",
    index=False
)

print("Data inserted into PostgreSQL successfully ✅")
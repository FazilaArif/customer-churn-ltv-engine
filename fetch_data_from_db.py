import pandas as pd
from sqlalchemy import create_engine

engine = create_engine(
    "postgresql://postgres:3008@localhost:5432/customer_churn"
)

query = "SELECT * FROM customer_churn_data"

df = pd.read_sql(query, engine)

print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)
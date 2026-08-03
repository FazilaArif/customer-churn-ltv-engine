from sqlalchemy import create_engine, text

engine = create_engine(
    "postgresql://postgres:3008@localhost:5432/customer_churn"
)

try:
    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1"))
        print("PostgreSQL connected successfully ✅")

except Exception as e:
    print("Connection failed ❌")
    print(e)
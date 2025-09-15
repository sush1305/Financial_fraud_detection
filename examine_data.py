import pandas as pd

# Load the data
data_path = 'data/raw/creditcard.csv'
df = pd.read_csv(data_path)

# Display basic information
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())
print("\nClass distribution:")
print(df['Class'].value_counts(normalize=True))

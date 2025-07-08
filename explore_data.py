import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("creditcard.csv")

# Show basic info
print("Shape of dataset:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nMissing values:\n", df.isnull().sum())
print("\nClass distribution:\n", df['Class'].value_counts())

# Visualize class distribution
sns.countplot(x='Class', data=df)
plt.title("Fraud (1) vs Not Fraud (0)")
plt.show()

# Summary statistics
print("\nSummary stats:\n", df.describe())

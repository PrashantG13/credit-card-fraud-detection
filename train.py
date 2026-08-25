# Zaroori Libraries Import Karna
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

print("Credit Card Fraud Detection Project Start Ho Gaya Hai!")

# Dummy/Sample Data Create kar rahe hain (Taaki bina heavy CSV ke code run ho sake)
# Real project mein yahan Kaggle ki CSV file load ki jati hai (pd.read_csv('creditcard.csv'))
np.random.seed(42)
data = {
    'Time': np.random.randint(0, 172800, 1000),
    'V1': np.random.randn(1000),
    'V2': np.random.randn(1000),
    'Amount': np.random.uniform(1.0, 500.0, 1000),
    'Class': np.random.choice([0, 1], size=1000, p=[0.99, 0.01]) # 0 = Normal, 1 = Fraud
}

df = pd.DataFrame(data)
print("Data ki pehli 5 rows:")
print(df.head())

# Features (X) aur Target (y) alag karna
X = df.drop('Class', axis=1)
y = df['Class']

# Training aur Testing data mein divide karna
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Model Train karna
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions karna
y_pred = model.predict(X_test)

# Model ki performance dekhna
print("\nModel Training Successful!")
print("Accuracy Score:", accuracy_score(y_test, y_pred))
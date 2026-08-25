# Zaroori Libraries Import Karna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

print("Credit Card Fraud Detection Project Start Ho Gaya Hai!")

# Sample Data Create kar rahe hain (Fraud Detection Simulation)
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

# --- GRAPH BANANE KA CODE ---
plt.figure(figsize=(8, 5))
feature_importances = model.feature_importances_
features = X.columns

sns.barplot(x=features, y=feature_importances, palette='viridis', hue=features, legend=False)
plt.title('Fraud Detection - Feature Importance Graph')
plt.xlabel('Features (Variables)')
plt.ylabel('Importance Score')
plt.grid(True, linestyle='--', alpha=0.6)

# Graph ko save karna
plt.savefig('fraud_plot.png')
print("\nFraud Analysis Graph 'fraud_plot.png' naam se successfully save ho gaya hai!")
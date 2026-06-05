import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the Dataset
# We are using a widely available raw CSV link for the Titanic dataset
print("Loading data...")
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)

# 2. Data Cleaning & Preprocessing
print("Cleaning data...")
# Select only the features we want to use for this simple model
features = ['Survived', 'Pclass', 'Sex', 'Age', 'Fare']
data = data[features]

# Handle missing values (Fill missing Ages with the median age)
data['Age'] = data['Age'].fillna(data['Age'].median())

# Convert categorical text data into numbers ('male' -> 0, 'female' -> 1)
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# 3. Split the Data
# X contains our inputs (features), y contains our output (survival)
X = data.drop('Survived', axis=1)
y = data['Survived']

# Split into training data (80%) and testing data (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Build and Train the Model
print("Training the Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Make Predictions and Evaluate
print("Evaluating model...")
predictions = model.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"\n--- Results ---")
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")
print("Detailed Classification Report:")
print(classification_report(y_test, predictions))

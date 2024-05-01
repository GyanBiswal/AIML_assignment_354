
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

np.random.seed(0)

# Generating random data for the Titanic dataset
n_passengers = 1000

# Age between 0 and 80
age = np.random.randint(0, 80, n_passengers)

# Gender: 0 for male, 1 for female
gender = np.random.randint(0, 2, n_passengers)

# Ticket class: 1, 2, or 3
ticket_class = np.random.randint(1, 4, n_passengers)

# Generating random survival values (0 or 1)
survived = np.random.randint(0, 2, n_passengers)


data = pd.DataFrame({
    'Age': age,
    'Gender': gender,
    'Class': ticket_class,
    'Survived': survived
})


X = data.drop('Survived', axis=1)
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)



accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

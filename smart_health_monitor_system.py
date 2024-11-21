import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE  # For balancing classes

# Generate Synthetic Health Data with Clearer Risk Levels
def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    heart_rate = np.random.randint(50, 120, size=num_samples)  # heart rate in beats per minute
    calories_burned = np.random.randint(100, 500, size=num_samples)  # calories burned
    steps_taken = np.random.randint(1000, 15000, size=num_samples)  # steps taken
    age = np.random.randint(18, 80, size=num_samples)  # age of user
    gender = np.random.choice([0, 1], size=num_samples)  # 0 for male, 1 for female
    
    # Risk level rules
    risk_level = np.where(
        (heart_rate < 70) & (steps_taken > 8000) & (calories_burned > 300) & (age < 40), 0,  # Low risk
        np.where((heart_rate > 100) | (age > 60) | (steps_taken < 4000), 2, 1)  # High risk or Moderate
    )
    
    data = pd.DataFrame({
        'heart_rate': heart_rate,
        'calories_burned': calories_burned,
        'steps_taken': steps_taken,
        'age': age,
        'gender': gender,
        'risk_level': risk_level
    })
    return data

# Step 1: Generate the data
data = generate_synthetic_data()

# Step 2: Features and Labels
X = data[['heart_rate', 'calories_burned', 'steps_taken', 'age', 'gender']]
y = data['risk_level']

# Step 3: Scale the features before applying SMOTE
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Balance the dataset using SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_scaled, y)

# Check class distribution after SMOTE
print("Class Distribution After SMOTE:")
print(pd.Series(y_balanced).value_counts())

# Step 5: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Step 6: Train a Random Forest model with balanced class weighting
model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Step 7: Evaluate the model
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 8: Visualize Health Metrics
plt.figure(figsize=(14, 7))

# Heart Rate vs. Calories Burned
plt.subplot(1, 2, 1)
sns.scatterplot(x='heart_rate', y='calories_burned', data=data, hue='risk_level', palette='coolwarm')
plt.title("Heart Rate vs Calories Burned")
plt.xlabel('Heart Rate (BPM)')
plt.ylabel('Calories Burned')

# Steps Taken vs. Age
plt.subplot(1, 2, 2)
sns.scatterplot(x='steps_taken', y='age', data=data, hue='risk_level', palette='coolwarm')
plt.title("Steps Taken vs Age")
plt.xlabel('Steps Taken')
plt.ylabel('Age')

plt.tight_layout()
plt.show()

# Step 9: Predict Health Risk for a New User Input
def predict_health_risk(heart_rate, calories_burned, steps_taken, age, gender):
    input_data = pd.DataFrame([[heart_rate, calories_burned, steps_taken, age, gender]],
                              columns=['heart_rate', 'calories_burned', 'steps_taken', 'age', 'gender'])
    input_scaled = scaler.transform(input_data)  # Scale the input
    risk_prediction = model.predict(input_scaled)
    risk_level = ['Low Risk', 'Moderate Risk', 'High Risk']
    return risk_level[risk_prediction[0]], risk_prediction[0]

# Step 10: Get User Inputs
print("Please enter the following health metrics:")
user_heart_rate = int(input("Enter Heart Rate (BPM): "))
user_calories_burned = int(input("Enter Calories Burned: "))
user_steps_taken = int(input("Enter Steps Taken: "))
user_age = int(input("Enter Age: "))
user_gender = int(input("Enter Gender (0 for Male, 1 for Female): "))

# Step 11: Predict the health risk for the user
health_risk = predict_health_risk(user_heart_rate, user_calories_burned, user_steps_taken, user_age, user_gender)
print(f"The predicted health risk for the user is: {health_risk}")

# Step 12: Visualization of User Data
labels = ['Heart Rate (BPM)', 'Calories Burned', 'Steps Taken', 'Age', 'Gender']
values = [user_heart_rate, user_calories_burned, user_steps_taken, user_age, user_gender]

# Bar Chart Visualization
plt.figure(figsize=(10, 6))
plt.bar(labels, values, color=['blue', 'green', 'red', 'purple', 'orange'])
plt.title("User Health Metrics")
plt.ylabel("Values")
plt.show()

# Radar Chart Visualization
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
values += values[:1]  # Close the loop
angles += angles[:1]  # Close the loop

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)
ax.fill(angles, values, color='cyan', alpha=0.6)
ax.plot(angles, values, color='blue', linewidth=2)
ax.set_yticklabels([])  # Hide radial ticks
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

plt.title("User Health Metrics - Radar Chart")
plt.show()

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# --------- 1. Sample Training Data ---------
data = {
    'age': [22, 30, 28, 35, 40, 24, 29, 33],
    'gender': ['male', 'female', 'male', 'female', 'male', 'female', 'male', 'female'],
    'bmi': [22.0, 28.5, 26.4, 21.8, 30.2, 24.5, 27.6, 23.2],
    'activity_level': ['moderate', 'sedentary', 'light', 'active', 'moderate', 'active', 'light', 'moderate'],
    'goal': ['muscle_gain', 'weight_loss', 'weight_loss', 'maintain', 'weight_loss', 'muscle_gain', 'weight_loss', 'maintain']
}

df = pd.DataFrame(data)

# --------- 2. Preprocessing ---------
label_encoders = {}
for column in ['gender', 'activity_level', 'goal']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# --------- 3. Train Model ---------
X = df[['age', 'gender', 'bmi', 'activity_level']]
y = df['goal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# --------- 4. New User Input ---------
new_user = {
    'age': 26,
    'gender': 'male',
    'height_cm': 180,
    'weight_kg': 75,
    'activity_level': 'moderate',
    'diet_pref': 'vegetarian'
}

# Calculate BMI
height_m = new_user['height_cm'] / 100
bmi = new_user['weight_kg'] / (height_m ** 2)

# Encode input as DataFrame with correct column names
input_features = pd.DataFrame([{
    'age': new_user['age'],
    'gender': label_encoders['gender'].transform([new_user['gender']])[0],
    'bmi': bmi,
    'activity_level': label_encoders['activity_level'].transform([new_user['activity_level']])[0]
}])

# Predict fitness goal
predicted_goal_code = model.predict(input_features)[0]
predicted_goal = label_encoders['goal'].inverse_transform([predicted_goal_code])[0]

print(f"\n🏁 Predicted Fitness Goal: {predicted_goal}")

# --------- 5. Calorie Calculation ---------
activity_multiplier = {
    'sedentary': 1.2,
    'light': 1.375,
    'moderate': 1.55,
    'active': 1.725
}

bmr = 10 * new_user['weight_kg'] + 6.25 * new_user['height_cm'] - 5 * new_user['age']
bmr += 5 if new_user['gender'] == 'male' else -161
tdee = bmr * activity_multiplier[new_user['activity_level']]

if predicted_goal == 'weight_loss':
    target_calories = tdee - 500
elif predicted_goal == 'muscle_gain':
    target_calories = tdee + 300
else:
    target_calories = tdee

# --------- 6. Macronutrient Split ---------
macros_percent = {
    'weight_loss': (0.4, 0.3, 0.3),
    'muscle_gain': (0.35, 0.45, 0.2),
    'maintain': (0.3, 0.4, 0.3)
}

protein_pct, carb_pct, fat_pct = macros_percent[predicted_goal]

protein_g = round((protein_pct * target_calories) / 4)
carb_g = round((carb_pct * target_calories) / 4)
fat_g = round((fat_pct * target_calories) / 9)

# --------- 7. Foods List ---------
veg_sources = ['Lentils', 'Chickpeas', 'Paneer', 'Tofu', 'Quinoa']
non_veg_sources = ['Chicken', 'Eggs', 'Fish', 'Greek Yogurt', 'Turkey']
protein_sources = veg_sources if new_user['diet_pref'] == 'vegetarian' else non_veg_sources

# --------- 8. Fitness Plan ---------
fitness_plan = {
    'weight_loss': ['Cardio (30 mins/day)', 'Calorie Deficit Diet', 'HIIT (3x/week)'],
    'muscle_gain': ['Strength Training (5x/week)', 'High-Protein Diet', 'Progressive Overload'],
    'maintain': ['Mixed Workouts (3x/week)', 'Balanced Diet', 'Flexibility Training']
}

# --------- 9. Output ---------
print("\n🏋️ Fitness Plan:")
for activity in fitness_plan[predicted_goal]:
    print("-", activity)

print("\n🥗 Nutrition Overview:")
print(f"- Daily Calorie Target: {int(target_calories)} kcal")
print(f"- Macronutrients: Protein: {protein_g}g, Carbs: {carb_g}g, Fats: {fat_g}g")
print(f"- Protein Sources ({new_user['diet_pref']}): {', '.join(protein_sources)}")
print("- Tips: Eat whole foods, stay hydrated, avoid processed sugar.")

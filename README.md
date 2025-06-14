# 🧠 Nutrition Plan Recommendation using Machine Learning

This project uses a **K-Nearest Neighbors (KNN)** classifier to predict a user's fitness goal (e.g., weight loss, muscle gain, or maintain) based on personal health data such as age, gender, BMI, and activity level. It then generates a personalized **nutrition plan**, **calorie requirements**, and **fitness routine** tailored to the prediction.

---

## 🚀 Features

- Predicts a user's **fitness goal** using a trained ML model  
- Calculates **BMI**, **BMR**, and **TDEE**  
- Suggests a daily **calorie target** and **macronutrient split**  
- Recommends food sources based on diet preference (vegetarian/non-vegetarian)  
- Provides a basic **fitness workout plan** aligned with the user's goal

---

## 🛠️ Technologies Used

- Python 🐍  
- pandas, NumPy  
- scikit-learn (KNeighborsClassifier, LabelEncoder)  

---

## 📊 Sample Input

```python
new_user = {
    'age': 26,
    'gender': 'male',
    'height_cm': 180,
    'weight_kg': 75,
    'activity_level': 'moderate',
    'diet_pref': 'vegetarian'
}
```

---

## 🧪 Output Example

```
🏁 Predicted Fitness Goal: muscle_gain

🏋️ Fitness Plan:
- Strength Training (5x/week)
- High-Protein Diet
- Progressive Overload

🥗 Nutrition Overview:
- Daily Calorie Target: 2816 kcal
- Macronutrients: Protein: 246g, Carbs: 316g, Fats: 63g
- Protein Sources (vegetarian): Lentils, Chickpeas, Paneer, Tofu, Quinoa
- Tips: Eat whole foods, stay hydrated, avoid processed sugar.
```

---

## 📁 How to Run

1. Clone the repository  
2. Install dependencies:  
   ```bash
   pip install pandas scikit-learn numpy
   ```
3. Run the script:  
   ```bash
   python nutrition_plan.py
   ```

---

## 🏆 Author

**Purav Bhatt**  
- [LinkedIn](https://www.linkedin.com/in/pb0504/)  
- Finalist – HackJNU 3.0 (Jan 2024)  
- 3rd Position – Inter-College Table Tennis Tournament (2024)

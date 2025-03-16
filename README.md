# README: Nutrition Analysis AI

## Introduction  
The **Nutrition Analysis AI** is a Flask-based web application designed to evaluate users' daily food intake, generate a **health rating**, and provide **personalized nutritional insights**. It allows users to select foods for different meals and receive an analysis of nutrient intake, **identify deficiencies**, and get suggestions for improving their diet.  

The application uses **data visualization techniques** such as pie charts and bar graphs to display nutrient distribution and suggest foods that best cover the identified deficiencies.  

---

## Features  
### 🔹 Food Selection  
Users can input their daily meals by selecting food items for **breakfast, lunch, snacks, and dinner**.  

### 🔹 Nutrition Analysis  
The app computes the total intake of various nutrients and **identifies deficiencies**.  

### 🔹 Health Rating  
A **normalized health rating** is calculated based on the **Nutrition Density** of the selected foods, on a scale of **0 to 10**.  

### 🔹 Nutrient Deficiency Detection  
The app determines nutrients that fall below **50% of the average intake**, labeling them as **deficient**.  

### 🔹 Food Suggestions  
For each deficient nutrient, **top 3 foods** rich in those nutrients are suggested.  

### 🔹 Top 5 Foods for Nutritional Balance  
The app identifies the **top 5 foods** that cover most of the deficiencies and presents them in a **bar chart**.  

### 🔹 Data Visualization  
- **Pie Chart** for **nutrient distribution**.  
- **Bar Chart** for **nutrient breakdown**.  
- **Top 5 Deficiency-Covering Foods Chart**.  

### 🔹 Model Performance Comparison  
The app includes pre-generated **ML model performance** visualizations for **Random Forest, LightGBM, and SVC**.  

---

## Installation and Setup  

### Prerequisites  
Ensure you have the following installed on your system:  
- Python 3.x  
- Flask (`pip install Flask`)  
- Pandas (`pip install pandas`)  
- Matplotlib (`pip install matplotlib`)  

### Steps to Install and Run  

1. **Clone the repository**  
   ```sh
   git clone https://github.com/yourusername/nutrition-analysis-app.git
   cd nutrition-analysis-app
   ```

2. **Install dependencies**  
   ```sh
   pip install -r requirements.txt
   ```

3. **Run the Flask app**  
   ```sh
   python app.py
   ```
   The app will be available at `http://127.0.0.1:8080/`.  

---

## Usage  

1. Open the app in a web browser.  
2. Select food items for **breakfast, lunch, snacks, and dinner**.  
3. Click **Analyze** to generate the report.  
4. View:  
   - **Health Rating**  
   - **Nutrient Breakdown (Pie Chart & Bar Chart)**  
   - **Deficient Nutrients**  
   - **Recommended Foods**  
   - **Top 5 Foods Covering Deficiencies**  
   - **Model Performance Comparisons**  

---

## File Structure  

```
/nutrition-analysis-app
│── app.py              # Main Flask application  
│── FOOD-DATA-GROUP11.csv  # Food dataset with nutrition values  
│── templates/  
│   ├── index.html      # Homepage with food selection form  
│   ├── results.html    # Results page displaying analysis  
│── static/  
│   ├── top_5_deficiency_covering_foods.png  # Chart for top 5 deficiency-covering foods  
│   ├── nutrient_pie_chart.png  # Pie chart for nutrient distribution  
│   ├── nutrient_bar_chart.png  # Bar chart for nutrient breakdown  
│   ├── figure_1.png  # ML Model: Random Forest  
│   ├── figure_2.png  # ML Model: LightGBM  
│   ├── figure_3.png  # ML Model: SVC  
│── requirements.txt  # Python dependencies  
│── README.md         # Project documentation  
```

---

## Key Functionalities  

### 🔹 Data Processing  
- The app reads **FOOD-DATA-GROUP11.csv**, which contains food items with **nutrient values** and a **Nutrition Density** score.  
- User-selected foods are **filtered**, and total nutrient intake is computed.  

### 🔹 Health Rating Calculation  
- The app **normalizes** the **Nutrition Density** score based on the highest available density.  
- The health rating is computed as:  
  \[
  \text{Normalized Rating} = \left( \frac{\sum \text{Selected Food Densities}}{\text{Total Selected Items} \times \text{Max Density}} \right) \times 10
  \]
- The final rating is **capped between 0 and 10**.  

### 🔹 Identifying Deficient Nutrients  
- Nutrients **below 50% of the mean intake** are marked as deficient.  
- A list of these nutrients is generated.  

### 🔹 Food Recommendations  
- **Top 3 foods** rich in each deficient nutrient are suggested.  
- **Top 5 foods** covering most deficiencies are determined and displayed using a bar chart.  

### 🔹 Data Visualization  
- **Pie Chart** for nutrient distribution.  
- **Bar Chart** for nutrient breakdown.  
- **Bar Chart for top 5 deficiency-covering foods**.  

---

## Technologies Used  
- **Flask** (Backend Framework)  
- **Pandas** (Data Processing)  
- **Matplotlib** (Data Visualization)  
- **HTML, CSS, JavaScript** (Frontend)  

---

## Potential Enhancements  
- **Integration of a larger food database** for better recommendations.  
- **User authentication** for personalized tracking.  
- **Calorie tracking and macronutrient goals** for fitness planning.  
- **AI-powered meal recommendations** using ML models.  

---

## Authors & Contributions  
- **Swagnik Dey** – Backend, Data Processing, Visualization  
- **Contributors Welcome!** Feel free to contribute by submitting **Pull Requests**.  

---

## License  
This project is licensed under the **MIT License**.  

---

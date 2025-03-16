from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load the processed food data
data_path = "FOOD-DATA-GROUP11.csv"
food_data = pd.read_csv(data_path)

# Clean column names to avoid issues
food_data.columns = food_data.columns.str.strip()

# Check column names for debugging
print("Available columns:", food_data.columns)

# Extract the list of foods for dropdown options
food_options = sorted(food_data["food"].unique())

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html', food_options=food_options)

# Route to process user input and return analysis
@app.route('/analyze', methods=['POST'])
def analyze():
    # Get selected foods from the form for each meal
    breakfast = request.form.getlist("breakfast")
    lunch = request.form.getlist("lunch")
    snacks = request.form.getlist("snacks")
    dinner = request.form.getlist("dinner")

    # Combine all selected foods
    selected_foods = breakfast + lunch + snacks + dinner
    selected_foods = [food.lower() for food in selected_foods]

    # Filter dataset for the selected foods
    filtered_data = food_data[food_data['food'].str.lower().isin(selected_foods)]

    # Check if "Nutrition Density" column exists
    if "Nutrition Density" in filtered_data.columns:
        nutrients = filtered_data.drop(columns=["food", "Nutrition Density"]).sum()

        # Calculate normalized health rating
        total_density = filtered_data["Nutrition Density"].sum()
        max_density = food_data["Nutrition Density"].max()
        normalized_rating = (total_density / (len(selected_foods) * max_density)) * 10

        # Ensure the rating is within the range of 0 to 10
        health_rating = min(max(normalized_rating, 0), 10)
    else:
        return "Error: 'Nutrition Density' column not found in the dataset."

    # Identify nutrient deficiencies (low nutrient scores)
    low_nutrients = nutrients[nutrients < nutrients.mean() * 0.5].index.tolist()

    # Suggest foods rich in deficient nutrients (top 3 per nutrient)
    suggested_foods = {}
    for nutrient in low_nutrients:
        top_foods = food_data.sort_values(by=nutrient, ascending=False).head(3)
        suggested_foods[nutrient] = top_foods["food"].tolist()

    # Suggest Top 5 Foods Covering Most Deficient Nutrients
    deficiency_scores = {}
    for _, row in food_data.iterrows():
        score = sum(row[nutrient] for nutrient in low_nutrients if nutrient in row)
        deficiency_scores[row["food"]] = score

    # Get top 5 foods with the highest total for deficient nutrients
    top_5_foods = sorted(deficiency_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    top_5_foods_names = [food for food, _ in top_5_foods]
    top_5_foods_scores = [score for _, score in top_5_foods]

    # Plot Bar Chart for Top 5 Foods Covering Deficiencies
    plt.figure(figsize=(10, 6))
    plt.bar(top_5_foods_names, top_5_foods_scores, color='green')
    plt.xlabel("Food Items")
    plt.ylabel("Deficiency Coverage Score")
    plt.title("Top 5 Foods Covering Deficient Nutrients")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    top_5_bar_chart_path = "static/top_5_deficiency_covering_foods.png"
    plt.savefig(top_5_bar_chart_path)
    plt.close()

    # Plot pie chart of nutrients
    plt.figure(figsize=(8, 6))
    nutrients.plot.pie(autopct='%1.1f%%', labels=nutrients.index)
    plt.title("Nutrient Distribution")
    pie_chart_path = "static/nutrient_pie_chart.png"
    plt.savefig(pie_chart_path)
    plt.close()

    # Plot bar chart of nutrients
    plt.figure(figsize=(10, 6))
    nutrients.plot(kind='bar', color='skyblue')
    plt.xlabel("Nutrients")
    plt.ylabel("Amount")
    plt.title("Nutrient Breakdown")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    bar_chart_path = "static/nutrient_bar_chart.png"
    plt.savefig(bar_chart_path)
    plt.close()

    # Use pre-generated model comparison figures
    model_comparison_charts = {
        "random_forest": "figure_1.png",
        "lightgbm": "figure_2.png",
        "svc": "figure_3.png"
    }

    # Prepare results for display
    results = {
        "health_rating": round(health_rating, 2),
        "low_nutrients": low_nutrients,
        "suggested_foods": suggested_foods,
        "top_5_foods": top_5_foods_names,
        "pie_chart": pie_chart_path,
        "bar_chart": bar_chart_path,
        "top_5_bar_chart": top_5_bar_chart_path,
        "model_comparison_charts": model_comparison_charts
    }

    return render_template("results.html", results=results)

if __name__ == '__main__':
    if not os.path.exists("static"):
        os.makedirs("static")
    app.run(debug=True, port=8080)

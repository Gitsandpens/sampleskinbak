"""Simple rule-based diet plan suggestions for over/underweight users."""


def compute_bmi(height_cm: float, weight_kg: float) -> float:
    height_m = height_cm / 100
    return weight_kg / (height_m ** 2)


def categorize_bmi(bmi: float) -> str:
    if bmi < 18.5:
        return "underweight"
    elif bmi < 25:
        return "normal"
    elif bmi < 30:
        return "overweight"
    else:
        return "obese"


DIET_SUGGESTIONS = {
    "underweight": {
        "summary": "Focus on calorie-dense, nutrient-rich foods to gain weight healthily.",
        "tips": [
            "Add healthy fats: nuts, peanut butter, avocado, olive oil.",
            "Eat frequent, smaller meals (5-6/day) instead of 3 large ones.",
            "Include protein with every meal: eggs, dairy, legumes, lean meat.",
            "Add smoothies with milk, banana, and nut butter as snacks.",
            "Strength training can help convert extra calories into muscle.",
        ],
    },
    "normal": {
        "summary": "Maintain your current balance with varied, whole-food meals.",
        "tips": [
            "Keep a balanced plate: half vegetables, quarter protein, quarter whole grains.",
            "Stay hydrated and maintain regular physical activity.",
            "Limit ultra-processed foods and added sugar.",
        ],
    },
    "overweight": {
        "summary": "Aim for a modest calorie deficit with nutrient-dense, high-fiber foods.",
        "tips": [
            "Prioritize vegetables, lean protein, and whole grains over refined carbs.",
            "Practice portion control; consider smaller plates.",
            "Cut sugary drinks; prefer water or unsweetened beverages.",
            "Aim for 150+ minutes/week of moderate activity (brisk walking, cycling).",
        ],
    },
    "obese": {
        "summary": "A structured, moderate calorie deficit combined with regular activity works best — consider consulting a nutritionist.",
        "tips": [
            "Build meals around vegetables, lean protein, and fiber to stay full longer.",
            "Reduce refined carbs and fried foods gradually rather than abruptly.",
            "Track meals for a couple of weeks to understand current intake patterns.",
            "Combine cardio with light resistance training, building up gradually.",
            "Consider professional guidance for a personalized plan.",
        ],
    },
}


def get_diet_plan(height_cm: float, weight_kg: float) -> dict:
    bmi = compute_bmi(height_cm, weight_kg)
    category = categorize_bmi(bmi)
    plan = DIET_SUGGESTIONS[category]
    return {
        "bmi": round(bmi, 1),
        "category": category,
        "summary": plan["summary"],
        "tips": plan["tips"],
    }

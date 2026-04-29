from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html", user_text="", prediction_text=None)

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("text", "").strip()
    if not text:
        return render_template("index.html", user_text="", prediction_text=None)

    cleaned = vectorizer.transform([text])
    prediction = model.predict(cleaned)[0]

    # Label mapping: 0 = Positive (normal), 1 = Negative (hate speech)
    result = "Positive 😊" if prediction == 0 else "Negative 😡"

    return render_template(
        "index.html",
        prediction_text=f"Sentiment: {result}",
        user_text=text
    )

if __name__ == "__main__":
    app.run(debug=True)
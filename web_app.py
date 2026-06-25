import os
import pickle
import pandas as pd
from flask import Flask, render_template, request, jsonify
from preprocess import preprocess_text

app = Flask(__name__)

SPAM_KEYWORDS = [
    "lottery", "won a", "you won", "winner", "prize", "claim",
    "congratulations", "selected", "lucky winner", "you have been selected",
    "account blocked", "account suspended", "will be suspended",
    "will be blocked", "will be cut", "kyc expired", "kyc update",
    "pan card blocked", "verify now", "update now", "immediately",
    "earn daily", "earn 5000", "earn 10000", "work from home",
    "double your money", "investment plan", "no investment",
    "salary 50000", "job offer", "apply now", "part time job",
    "get rich", "guaranteed",
    "free recharge", "free data", "free gift", "free iphone",
    "free prize", "free entry",
    "click here", "click now", "call now", "call 9", "call 8", "call 7",
    "sms back", "reply win", "reply yes",
    "customs fee", "package on hold", "delivery fee", "pay now",
    "cheap loan", "loan approved", "no documents",
    "send otp", "share otp",
    "casino", "real money", "betting",
]


def load_model():
    brain_path = "model/spam_brain.pkl"
    vec_path = "model/tfidf_engine.pkl"
    with open(brain_path, "rb") as f:
        brain = pickle.load(f)
    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)
    return brain, vectorizer


brain, vectorizer = load_model()


def check_keywords(message):
    msg_lower = message.lower()
    matched = [kw for kw in SPAM_KEYWORDS if kw in msg_lower]
    return len(matched) > 0, matched


def check_with_model(message):
    cleaned = preprocess_text(message)
    as_vector = vectorizer.transform([cleaned])
    proba = brain.predict_proba(as_vector)[0]
    classes = brain.classes_.tolist()
    spam_prob = proba[classes.index("spam")] * 100
    ham_prob = proba[classes.index("ham")] * 100
    is_spam = spam_prob >= 30.0
    return is_spam, spam_prob, ham_prob


def detect_spam(message):
    keyword_spam, matched_keywords = check_keywords(message)
    ml_spam, spam_prob, ham_prob = check_with_model(message)
    final_is_spam = keyword_spam or ml_spam

    if keyword_spam:
        detection_reason = f"keyword match: {', '.join(matched_keywords[:3])}"
        confidence = max(spam_prob, 85.0)
    elif ml_spam:
        detection_reason = "ML model flagged"
        confidence = spam_prob
    else:
        detection_reason = "Looks safe"
        confidence = ham_prob

    return {
        "is_spam": final_is_spam,
        "confidence": round(confidence, 1),
        "spam_probability": round(spam_prob, 1),
        "ham_probability": round(ham_prob, 1),
        "reason": detection_reason,
        "matched_keywords": matched_keywords,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    message = request.form.get("message", "").strip()
    if not message:
        return jsonify({"error": "No message provided"}), 400
    result = detect_spam(message)
    result["message"] = message
    return jsonify(result)


@app.route("/bulk", methods=["POST"])
def bulk():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    messages = []
    if file.filename.endswith(".csv"):
        df = pd.read_csv(file)
        col = request.form.get("column", "message")
        if col not in df.columns:
            return jsonify({"error": f"Column '{col}' not found"}), 400
        messages = df[col].dropna().tolist()
    else:
        content = file.read().decode("utf-8", errors="replace")
        messages = [line.strip() for line in content.splitlines() if line.strip()]

    results = []
    spam_count = 0
    for msg in messages:
        r = detect_spam(msg)
        r["message"] = msg
        results.append(r)
        if r["is_spam"]:
            spam_count += 1

    return jsonify({
        "total": len(results),
        "spam_count": spam_count,
        "legit_count": len(results) - spam_count,
        "results": results,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

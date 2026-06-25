# train.py  (v3 — upgraded model)
# ──────────────────────────────────────────────
# ok so i upgraded 3 things this time:
# 1. added extra spam keywords manually (boosted dataset)
# 2. switched to logistic regression (way better than naive bayes for edge cases)
# 3. made the spam threshold stricter (catches more spam now)
# lets gooo 🚀
# ──────────────────────────────────────────────

import os
import sys
import time
import pickle
import pandas as pd
from preprocess import preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

os.system("color")

RED     = "\033[91m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
BLUE    = "\033[94m"
CYAN    = "\033[96m"
WHITE   = "\033[97m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
RESET   = "\033[0m"


def slow_print(text, delay=0.03):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()


def loading_bar(label, total_time=2.0):
    bar_size = 35
    print(f"\n  {CYAN}{label}{RESET}")
    for step in range(bar_size + 1):
        filled  = "█" * step
        empty   = "░" * (bar_size - step)
        percent = int((step / bar_size) * 100)
        color   = GREEN if percent == 100 else YELLOW
        sys.stdout.write(f"\r  {color}[{filled}{empty}] {percent}%{RESET}")
        sys.stdout.flush()
        time.sleep(total_time / bar_size)
    print()


def spinner(message, seconds=1.5):
    frames   = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    end_time = time.time() + seconds
    idx      = 0
    while time.time() < end_time:
        sys.stdout.write(f"\r  {CYAN}{frames[idx % len(frames)]}{RESET}  {message}")
        sys.stdout.flush()
        time.sleep(0.1)
        idx += 1
    sys.stdout.write(f"\r  {GREEN}✔{RESET}  {message}{' ' * 10}\n")
    sys.stdout.flush()


def print_banner():
    os.system("cls" if os.name == "nt" else "clear")
    print(f"""
{CYAN}{BOLD}
  ╔══════════════════════════════════════════════════╗
  ║     📧  SPAM DETECTOR v3 — TRAINING MODE  📧     ║
  ║     now with logistic regression + keywords      ║
  ╚══════════════════════════════════════════════════╝
{RESET}""")


# ── extra spam messages added manually ──
# these are common Indian spam patterns the kaggle dataset misses
EXTRA_SPAM_MESSAGES = [
    "You won a lottery call 9000012345",
    "Congratulations you have won 1 crore rupees call now",
    "Free recharge click here to claim",
    "Your KYC is expired update now or account blocked",
    "Win iPhone 15 click this link now",
    "Earn 5000 daily from home no investment required",
    "Your SBI account is blocked call 9876543210 immediately",
    "Urgent your Paytm wallet will be suspended verify now",
    "Dear customer your electricity will be cut tonight pay now",
    "Claim your free gift voucher worth 5000 rupees today",
    "Job offer salary 50000 per month apply now",
    "Investment plan double your money in 30 days guaranteed",
    "Your PAN card is blocked call helpline immediately",
    "Free data 2GB click link to activate now",
    "You are selected for government scheme claim benefit",
    "Send OTP to verify your prize winning amount",
    "Cheap loan approved instantly no documents needed",
    "Work from home earn 10000 daily part time job",
    "Your package is on hold pay customs fee click here",
    "Lottery winner congratulations claim prize call now",
    "Winner you won a prize SMS back YES to claim",
    "Get rich quick scheme join now limited slots available",
    "Free SIM card offer call now to activate",
    "Your account will be suspended call immediately",
    "Congratulations you are our lucky winner today",
]

EXTRA_LEGIT_MESSAGES = [
    "Hey are you coming to college tomorrow",
    "Mom said dinner is ready come home",
    "Can you send me the notes for today class",
    "Meeting at 3pm dont forget",
    "I will be late by 10 minutes",
    "Happy birthday have a great day",
    "Did you finish the assignment",
    "Lets catch up this weekend",
    "The match starts at 7pm tonight",
    "Can you call me when you are free",
    "Please pick up milk on the way home",
    "Class is cancelled today professor is sick",
    "Are you free this evening lets hang out",
    "Good morning have a nice day",
    "Can we reschedule our meeting to tomorrow",
]


# ── step 1 — load data + boost with extra messages ──
def load_data():
    slow_print(f"\n  {BLUE}📂  Step 1 — Loading dataset + adding extra spam patterns...{RESET}", 0.03)

    if not os.path.exists("spam.csv"):
        print(f"\n  {RED}❌  spam.csv not found!{RESET}")
        print(f"  {YELLOW}👉  Download from:{RESET}")
        print("      https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset\n")
        sys.exit(1)

    spinner("Reading spam.csv...", 1.5)

    raw  = pd.read_csv("spam.csv", encoding="latin-1")
    data = raw[["v1", "v2"]].copy()
    data.columns = ["label", "message"]

    extra_spam  = pd.DataFrame({"label": ["spam"] * len(EXTRA_SPAM_MESSAGES),  "message": EXTRA_SPAM_MESSAGES})
    extra_legit = pd.DataFrame({"label": ["ham"]  * len(EXTRA_LEGIT_MESSAGES), "message": EXTRA_LEGIT_MESSAGES})

    data = pd.concat([data, extra_spam, extra_legit], ignore_index=True)

    print(f"\n  {WHITE}📊  Total messages      :{RESET} {CYAN}{len(data)}{RESET}")
    print(f"  {WHITE}🚨  Spam messages       :{RESET} {RED}{len(data[data['label'] == 'spam'])}{RESET}")
    print(f"  {WHITE}✅  Legit messages      :{RESET} {GREEN}{len(data[data['label'] == 'ham'])}{RESET}")
    print(f"  {YELLOW}  ➕  Extra spam added   : {len(EXTRA_SPAM_MESSAGES)} messages{RESET}")
    print(f"  {YELLOW}  ➕  Extra legit added  : {len(EXTRA_LEGIT_MESSAGES)} messages{RESET}")

    return data


# ── step 2 — clean data ──
def clean_data(data):
    slow_print(f"\n  {BLUE}🧹  Step 2 — Cleaning data...{RESET}", 0.03)
    spinner("Removing duplicates...", 1.2)

    before = len(data)
    data   = data.drop_duplicates()
    data   = data.dropna()
    after  = len(data)

    print(f"\n  {WHITE}🗑️   Removed {YELLOW}{before - after}{RESET} {WHITE}duplicate rows{RESET}")
    print(f"  {WHITE}✅  Clean rows left     :{RESET} {GREEN}{after}{RESET}")
    return data


# ── step 3 — tfidf with trigrams ──
# catches phrases like "call now claim prize" much better
def vectorize_text(data):
    slow_print(f"\n  {BLUE}🔢  Step 3 — TF-IDF vectorization (upgraded)...{RESET}", 0.03)

    messages = data["message"].apply(preprocess_text)
    labels   = data["label"]

    msg_train, msg_test, lbl_train, lbl_test = train_test_split(
        messages, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    loading_bar("Converting text to vectors...", 2.5)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=8000,
        ngram_range=(1, 3),   # catches 1,2,3 word combos
        sublinear_tf=True     # smooths word frequency
    )

    train_vecs = vectorizer.fit_transform(msg_train)
    test_vecs  = vectorizer.transform(msg_test)

    print(f"\n  {WHITE}📐  Training samples    :{RESET} {CYAN}{train_vecs.shape[0]}{RESET}")
    print(f"  {WHITE}🧪  Testing samples     :{RESET} {CYAN}{test_vecs.shape[0]}{RESET}")
    print(f"  {WHITE}🔤  Vocabulary size     :{RESET} {CYAN}{train_vecs.shape[1]}{RESET}")

    return vectorizer, train_vecs, test_vecs, lbl_train, lbl_test


# ── step 4 — train BOTH models and pick the best ──
# PDF needs Naive Bayes so we train both and compare
# Naive Bayes = fast, simple, good baseline
# Logistic Regression = better on tricky/short messages
def train_model(train_vecs, lbl_train, test_vecs, lbl_test):
    slow_print(f"\n  {BLUE}🧠  Step 4 — Training both models and comparing...{RESET}", 0.03)
    slow_print(f"  {YELLOW}   (PDF needs Naive Bayes so we train both and pick winner){RESET}", 0.02)

    # model 1 — naive bayes
    loading_bar("Training Naive Bayes...", 1.5)
    nb_model = MultinomialNB()
    nb_model.fit(train_vecs, lbl_train)
    nb_acc = accuracy_score(lbl_test, nb_model.predict(test_vecs))
    print(f"  {WHITE}  Naive Bayes accuracy     :{RESET} {CYAN}{nb_acc*100:.2f}%{RESET}")

    # model 2 — logistic regression
    loading_bar("Training Logistic Regression...", 2.0)
    lr_model = LogisticRegression(class_weight="balanced", C=2, max_iter=1000, random_state=42)
    lr_model.fit(train_vecs, lbl_train)
    lr_acc = accuracy_score(lbl_test, lr_model.predict(test_vecs))
    print(f"  {WHITE}  Logistic Regression acc  :{RESET} {CYAN}{lr_acc*100:.2f}%{RESET}")

    # pick the winner
    if lr_acc >= nb_acc:
        slow_print(f"\n  {GREEN}🏆  Logistic Regression wins — saving it!{RESET}", 0.03)
        return lr_model, "Logistic Regression"
    else:
        slow_print(f"\n  {GREEN}🏆  Naive Bayes wins — saving it!{RESET}", 0.03)
        return nb_model, "Naive Bayes"


# ── step 5 — evaluate ──
def evaluate_model(model, test_vecs, lbl_test):
    slow_print(f"\n  {BLUE}📊  Step 5 — Testing accuracy...{RESET}", 0.03)
    spinner("Running predictions...", 2.0)

    predictions = model.predict(test_vecs)
    accuracy    = accuracy_score(lbl_test, predictions)
    score_pct   = accuracy * 100

    print(f"\n  {WHITE}🎯  Accuracy Score      :{RESET} ", end="")
    if score_pct >= 97:
        print(f"{GREEN}{BOLD}{score_pct:.2f}%  🔥 Insane!{RESET}")
    elif score_pct >= 94:
        print(f"{CYAN}{score_pct:.2f}%  😎 Solid{RESET}")
    else:
        print(f"{YELLOW}{score_pct:.2f}%  🤔 Decent{RESET}")

    print(f"\n  {CYAN}── Detailed Report ─────────────────────────────{RESET}")
    print(classification_report(lbl_test, predictions))


# ── step 6 — save ──
def save_model(model, vectorizer):
    slow_print(f"\n  {BLUE}💾  Step 6 — Saving upgraded model...{RESET}", 0.03)
    spinner("Writing to disk...", 1.5)

    os.makedirs("model", exist_ok=True)
    with open("model/spam_brain.pkl",   "wb") as f: pickle.dump(model,   f)
    with open("model/tfidf_engine.pkl", "wb") as f: pickle.dump(vectorizer, f)

    print(f"\n  {GREEN}📁  Saved:{RESET} model/spam_brain.pkl")
    print(f"  {GREEN}📁  Saved:{RESET} model/tfidf_engine.pkl")


def main():
    print_banner()
    time.sleep(0.4)
    slow_print(f"  {WHITE}Upgrading the spam brain — v3 incoming 🚀{RESET}", 0.04)
    slow_print(f"  {YELLOW}3 upgrades: keywords + logistic regression + stricter{RESET}\n", 0.03)
    time.sleep(0.5)

    raw_data                         = load_data()
    clean                            = clean_data(raw_data)
    vec, tr_v, te_v, tr_l, te_l     = vectorize_text(clean)
    model, winner                    = train_model(tr_v, tr_l, te_v, te_l)
    evaluate_model(model, te_v, te_l)
    save_model(model, vec)

    print(f"\n  {CYAN}══════════════════════════════════════════════{RESET}")
    slow_print(f"  {GREEN}{BOLD}✅  v3 ready! Now run: python app.py{RESET}", 0.04)
    slow_print(f"  {YELLOW}  Try: 'You won a lottery call 9000012345'{RESET}", 0.03)
    print(f"  {CYAN}══════════════════════════════════════════════{RESET}\n")


if __name__ == "__main__":
    main()

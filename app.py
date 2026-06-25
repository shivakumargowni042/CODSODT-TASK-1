# app.py (v3 — 3-layer spam detection)
# ──────────────────────────────────────────────
# ok so the old version was only using the ML model
# which is trained on UK SMS data from kaggle
# but indian spam like "lottery call 9000012345" looks different
# so i added 3 layers now:
#   layer 1 = keyword rules  (hard rules, always fires first)
#   layer 2 = ML model       (logistic regression)
#   layer 3 = low threshold  (flags if 30%+ spam probability)
# all 3 work together so barely anything slips through now
# ──────────────────────────────────────────────

import os
import sys
import time
import pickle
import datetime
import subprocess
import pandas as pd
from preprocess import preprocess_text

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


# ── helpers ──────────────────────────────────

def slow_print(text, delay=0.03):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
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
    sys.stdout.write(f"\r  {GREEN}✔{RESET}  {message}{' ' * 15}\n")
    sys.stdout.flush()


# ── layer 1: keyword rules ────────────────────
# these are hard-coded spam signals
# if ANY of these appear in the message → instant spam flag
# no need to wait for ML model
SPAM_KEYWORDS = [
    # prize / winning
    "lottery", "won a", "you won", "winner", "prize", "claim",
    "congratulations", "selected", "lucky winner", "you have been selected",
    # urgency / account threats
    "account blocked", "account suspended", "will be suspended",
    "will be blocked", "will be cut", "kyc expired", "kyc update",
    "pan card blocked", "verify now", "update now", "immediately",
    # money / jobs
    "earn daily", "earn 5000", "earn 10000", "work from home",
    "double your money", "investment plan", "no investment",
    "salary 50000", "job offer", "apply now", "part time job",
    "get rich", "guaranteed",
    # free stuff
    "free recharge", "free data", "free gift", "free iphone",
    "free prize", "free entry",
    # click / call baits
    "click here", "click now", "call now", "call 9", "call 8", "call 7",
    "sms back", "reply win", "reply yes",
    # payment fraud
    "customs fee", "package on hold", "delivery fee", "pay now",
    "cheap loan", "loan approved", "no documents",
    # otp fraud
    "send otp", "share otp",
    # casino
    "casino", "real money", "betting",
]


def check_keywords(message):
    """
    layer 1 — scans message for known spam keywords
    returns (is_spam, matched_keywords)
    if any keyword found → spam, no need for ML
    """
    msg_lower = message.lower()
    matched   = [kw for kw in SPAM_KEYWORDS if kw in msg_lower]
    return len(matched) > 0, matched


# ── layer 2 + 3: ML model with low threshold ──
def check_with_model(message, brain, vectorizer):
    """
    layer 2 = ML model prediction
    layer 3 = if spam probability > 30% → flag as spam
    default threshold is 50% which misses a lot
    30% is more aggressive and catches edge cases
    """
    cleaned      = preprocess_text(message)
    as_vector    = vectorizer.transform([cleaned])
    proba        = brain.predict_proba(as_vector)[0]
    classes      = brain.classes_.tolist()
    spam_prob    = proba[classes.index("spam")] * 100
    ham_prob     = proba[classes.index("ham")]  * 100

    # layer 3 — lowered threshold: flag as spam if 30%+ confident
    # default is 50% which is why it was missing stuff
    is_spam = spam_prob >= 30.0

    return is_spam, spam_prob, ham_prob


# ── combined 3-layer detection ─────────────────
def detect_spam(message, brain, vectorizer):
    """
    runs all 3 layers and combines results
    layer 1 always takes priority (hard rules)
    """
    # layer 1 — keyword check
    keyword_spam, matched_keywords = check_keywords(message)

    # layer 2 + 3 — ML model with low threshold
    ml_spam, spam_prob, ham_prob = check_with_model(message, brain, vectorizer)

    # final decision — spam if EITHER layer says so
    final_is_spam = keyword_spam or ml_spam

    # figure out which layer caught it (for display)
    if keyword_spam:
        detection_reason = f"keyword match: {', '.join(matched_keywords[:3])}"
        confidence       = max(spam_prob, 85.0)   # keyword match = high confidence
    elif ml_spam:
        detection_reason = "ML model flagged"
        confidence       = spam_prob
    else:
        detection_reason = "ML model — looks safe"
        confidence       = ham_prob

    return final_is_spam, confidence, spam_prob, ham_prob, detection_reason


# ── load saved model ──────────────────────────
def load_model():
    brain_path = "model/spam_brain.pkl"
    vec_path   = "model/tfidf_engine.pkl"

    if not os.path.exists(brain_path) or not os.path.exists(vec_path):
        print(f"\n  {RED}❌  Model files not found!{RESET}")
        print(f"  {YELLOW}👉  Run this first:{RESET}  python train.py\n")
        sys.exit(1)

    spinner("Loading trained model...", 1.5)

    try:
        with open(brain_path, "rb") as f: brain      = pickle.load(f)
        with open(vec_path,   "rb") as f: vectorizer = pickle.load(f)
    except Exception as e:
        print(f"\n  {RED}❌  Failed to load model: {e}{RESET}")
        print(f"  {YELLOW}👉  Run this first:{RESET}  python train.py\n")
        sys.exit(1)

    slow_print(f"  {GREEN}🧠  3-layer detection system ready!{RESET}", 0.02)
    return brain, vectorizer


# ── save to log ───────────────────────────────
def save_to_log(message, is_spam, confidence, reason):
    os.makedirs("logs", exist_ok=True)
    timestamp   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    verdict_str = "SPAM" if is_spam else "LEGIT"
    log_line    = f"[{timestamp}]  [{verdict_str}]  [{confidence:.1f}%]  [{reason}]  {message}\n"
    with open("logs/history.log", "a", encoding="utf-8") as f:
        f.write(log_line)


# ── show history ──────────────────────────────
def show_history():
    log_path = "logs/history.log"
    if not os.path.exists(log_path):
        print(f"\n  {YELLOW}📭  No history yet!{RESET}\n")
        return
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if not lines:
        print(f"\n  {YELLOW}📭  Log file is empty.{RESET}\n")
        return
    print(f"\n  {CYAN}── Last 10 predictions ─────────────────────────{RESET}")
    for line in lines[-10:]:
        line = line.strip()
        color = RED if "[SPAM" in line else GREEN
        print(f"  {color}{line}{RESET}")
    print(f"  {CYAN}────────────────────────────────────────────────{RESET}")
    print(f"  {DIM}  Full log: logs/history.log{RESET}\n")


# ── show result ───────────────────────────────
def show_result(is_spam, confidence, spam_pct, ham_pct, reason):
    print()
    suspense = ["🔍 Scanning...", "🧠 Checking layers...", "📡 Analyzing patterns...", "⚡ Almost done..."]
    for line in suspense:
        sys.stdout.write(f"\r  {CYAN}{line}{RESET}   ")
        sys.stdout.flush()
        time.sleep(0.4)
    print("\n")

    # confidence tier
    if not is_spam:
        badge = f"{GREEN}🟢 SAFE{RESET}"
    elif confidence >= 80:
        badge = f"{RED}🔴 HIGH CONFIDENCE SPAM{RESET}"
    else:
        badge = f"{YELLOW}🟡 SUSPICIOUS{RESET}"

    if is_spam:
        print(f"  {RED}{BOLD}╔══════════════════════════════════════════╗{RESET}")
        print(f"  {RED}{BOLD}║   🚨   SPAM DETECTED — DO NOT TRUST!  🚨  ║{RESET}")
        print(f"  {RED}{BOLD}╚══════════════════════════════════════════╝{RESET}")
        print()
        print(f"  {badge}")
        print()
        slow_print(f"  {RED}  Do not click any links in this message!{RESET}", 0.02)
        slow_print(f"  {RED}  Delete it immediately 🗑️{RESET}", 0.02)
    else:
        print(f"  {GREEN}{BOLD}╔══════════════════════════════════════════╗{RESET}")
        print(f"  {GREEN}{BOLD}║   ✅   LOOKS LEGIT — SEEMS SAFE!   ✅    ║{RESET}")
        print(f"  {GREEN}{BOLD}╚══════════════════════════════════════════╝{RESET}")
        print()
        print(f"  {badge}")
        print()
        slow_print(f"  {GREEN}  This message looks normal 😊{RESET}", 0.02)

    # confidence bar
    bar_len    = 30
    filled_amt = int((confidence / 100) * bar_len)
    bar_color  = RED if is_spam else GREEN
    bar_filled = "█" * filled_amt
    bar_empty  = "░" * (bar_len - filled_amt)
    verdict    = "SPAM" if is_spam else "LEGIT"

    print()
    print(f"  {WHITE}Confidence  [{bar_color}{bar_filled}{RESET}{bar_empty}] {bar_color}{BOLD}{confidence:.1f}%  ({verdict}){RESET}")
    print()
    print(f"  {DIM}  🔍 Detected by    : {reason}{RESET}")
    print(f"  {DIM}  🚨 Spam probability: {spam_pct:.1f}%{RESET}")
    print(f"  {DIM}  ✅ Legit probability: {ham_pct:.1f}%{RESET}")
    print(f"\n  {CYAN}{'─'*44}{RESET}")


# ── welcome banner ────────────────────────────
def print_banner():
    os.system("cls" if os.name == "nt" else "clear")
    lines = [
        f"",
        f"  {CYAN}{BOLD}╔══════════════════════════════════════════════╗{RESET}",
        f"  {CYAN}{BOLD}║   📱   SPAM SMS DETECTOR  v3.0   📱          ║{RESET}",
        f"  {CYAN}{BOLD}║   3-layer detection: keywords + ML + threshold║{RESET}",
        f"  {CYAN}{BOLD}╚══════════════════════════════════════════════╝{RESET}",
        f"",
    ]
    for line in lines:
        slow_print(line, 0.008)
        time.sleep(0.04)


# ── menu ──────────────────────────────────────
def show_menu():
    print(f"  {CYAN}What do you want to do?{RESET}")
    print(f"  {WHITE}  [1]{RESET}  Check a single message")
    print(f"  {WHITE}  [2]{RESET}  View prediction history")
    print(f"  {WHITE}  [3]{RESET}  Bulk scan a file (.txt / .csv)")
    print(f"  {WHITE}  [4]{RESET}  Retrain model")
    print(f"  {WHITE}  [5]{RESET}  Exit")
    print()
    return input(f"  {YELLOW}Your choice (1/2/3/4/5): {RESET}").strip()


# ── get message ───────────────────────────────
def get_user_message():
    print(f"\n  {CYAN}{'─'*44}{RESET}")
    print(f"  {WHITE}Type or paste your SMS message:{RESET}")
    print(f"  {CYAN}{'─'*44}{RESET}")
    return input(f"\n  {YELLOW}📝  Message: {RESET}").strip()


# ── session summary ───────────────────────────
def show_summary(total, spam_count, legit_count):
    print(f"\n  {CYAN}{'═'*44}{RESET}")
    slow_print(f"  {WHITE}{BOLD}  📈  Session Summary{RESET}", 0.03)
    print(f"  {CYAN}{'─'*44}{RESET}")
    print(f"  {WHITE}  🔢  Total checked   :{RESET}  {CYAN}{total}{RESET}")
    print(f"  {WHITE}  🚨  Spam found      :{RESET}  {RED}{spam_count}{RESET}")
    print(f"  {WHITE}  ✅  Legit messages  :{RESET}  {GREEN}{legit_count}{RESET}")
    if total > 0:
        print(f"  {WHITE}  📊  Spam rate       :{RESET}  {YELLOW}{(spam_count/total)*100:.0f}%{RESET}")
    print(f"  {DIM}  📁  Log saved to: logs/history.log{RESET}")
    print(f"  {CYAN}{'═'*44}{RESET}")
    slow_print(f"\n  {GREEN}Stay safe out there! 🛡️{RESET}\n", 0.04)


# ── bulk file scan ────────────────────────────
def bulk_scan(brain, vectorizer):
    path = input(f"\n  {YELLOW}📂  File path: {RESET}").strip().strip('"').strip("'")

    if not os.path.exists(path):
        slow_print(f"\n  {RED}❌  File not found!{RESET}\n", 0.03)
        return 0, 0, 0

    messages = []
    ext = os.path.splitext(path)[1].lower()

    try:
        if ext == ".csv":
            col = input(f"  {YELLOW}Column name with messages (default: message): {RESET}").strip() or "message"
            df = pd.read_csv(path, encoding="utf-8")
            if col not in df.columns:
                print(f"  {RED}❌  Column '{col}' not found! Try: {', '.join(df.columns)}{RESET}")
                return 0, 0, 0
            messages = df[col].dropna().tolist()
        else:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                messages = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"\n  {RED}❌  Error reading file: {e}{RESET}\n")
        return 0, 0, 0

    if not messages:
        slow_print(f"\n  {YELLOW}⚠️  No messages found in file{RESET}\n", 0.03)
        return 0, 0, 0

    print(f"\n  {CYAN}── Scanning {len(messages)} messages ─────────────────{RESET}")

    spam_count = 0
    for i, msg in enumerate(messages, 1):
        is_spam, confidence, spam_pct, ham_pct, reason = detect_spam(msg, brain, vectorizer)
        label = f"{RED}SPAM{RESET}" if is_spam else f"{GREEN}LEGIT{RESET}"
        bar   = "█" * int((confidence / 100) * 20)
        dots  = "░" * (20 - int((confidence / 100) * 20))
        color = RED if is_spam else GREEN
        print(f"  [{i}/{len(messages)}] {label}  {color}{bar}{dots}{RESET} {confidence:5.1f}%  {msg[:60]}{'...' if len(msg) > 60 else ''}")
        save_to_log(msg, is_spam, confidence, reason)
        if is_spam:
            spam_count += 1

    legit_count = len(messages) - spam_count
    print(f"  {CYAN}──────────────────────────────────────────────{RESET}")
    print(f"  {WHITE}Total: {len(messages)}  |  {RED}Spam: {spam_count}{RESET}  |  {GREEN}Legit: {legit_count}{RESET}\n")
    return len(messages), spam_count, legit_count


# ── retrain model ─────────────────────────────
def retrain_model():
    slow_print(f"\n  {YELLOW}⚠️  Retraining will overwrite the current model.{RESET}", 0.03)
    confirm = input(f"  {YELLOW}Continue? (y/n): {RESET}").strip().lower()
    if confirm != "y":
        slow_print(f"\n  {BLUE}  Cancelled.{RESET}\n", 0.03)
        return None, None

    slow_print(f"\n  {CYAN}Launching train.py...{RESET}\n", 0.03)
    try:
        subprocess.run([sys.executable, "train.py"], check=True)
    except subprocess.CalledProcessError:
        print(f"\n  {RED}❌  Training failed!{RESET}\n")
        return None, None

    print()
    try:
        return load_model()
    except SystemExit:
        print(f"\n  {RED}❌  Model files missing after retrain!{RESET}\n")
        return None, None


# ── main ──────────────────────────────────────
def main():
    print_banner()
    time.sleep(0.3)
    slow_print(f"  {WHITE}Hey! Welcome to Spam Detector v3 👋{RESET}", 0.04)
    slow_print(f"  {DIM}  Using 3-layer detection — much harder to fool now{RESET}\n", 0.03)

    brain, vectorizer = load_model()

    total_checked = 0
    spam_found    = 0
    legit_found   = 0

    while True:
        choice = show_menu()

        if choice == "1":
            user_msg = get_user_message()

            if not user_msg:
                slow_print(f"\n  {YELLOW}⚠️  You didn't type anything lol{RESET}\n", 0.03)
                continue

            is_spam, confidence, spam_pct, ham_pct, reason = detect_spam(
                user_msg, brain, vectorizer
            )

            show_result(is_spam, confidence, spam_pct, ham_pct, reason)
            save_to_log(user_msg, is_spam, confidence, reason)

            total_checked += 1
            if is_spam:
                spam_found += 1
            else:
                legit_found += 1

            print(f"  {DIM}  ✔  Saved to logs/history.log{RESET}\n")

        elif choice == "2":
            show_history()

        elif choice == "3":
            t, s, l = bulk_scan(brain, vectorizer)
            total_checked += t
            spam_found    += s
            legit_found   += l

        elif choice == "4":
            result = retrain_model()
            if result[0] is not None:
                brain, vectorizer = result
                slow_print(f"  {GREEN}✅  Model reloaded successfully!{RESET}\n", 0.03)

        elif choice == "5":
            show_summary(total_checked, spam_found, legit_found)
            break

        else:
            slow_print(f"\n  {YELLOW}⚠️  Type 1–5 only lol{RESET}\n", 0.03)


if __name__ == "__main__":
    main()

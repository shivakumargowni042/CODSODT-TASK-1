import re


def preprocess_text(text):
    text = re.sub(r"http[s]?://\S+", " [URL] ", text)
    text = re.sub(r"www\.\S+", " [URL] ", text)
    text = re.sub(r"\b\d{10}\b", " [PHONE] ", text)
    text = re.sub(r"\b\d{5}\s?\d{5}\b", " [PHONE] ", text)
    text = re.sub(r"[₹£€$]\s?\d+[\d,]*", " [MONEY] ", text)
    text = re.sub(r"\b\d{3,}[\d,]*\b", " [MONEY] ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

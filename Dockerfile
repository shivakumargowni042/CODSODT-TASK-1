FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt && python train.py
EXPOSE 7860
CMD gunicorn web_app:app --bind 0.0.0.0:7860

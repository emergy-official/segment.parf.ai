FROM python:3.11.7

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./
COPY inference.py .
COPY iris_model.pkl .

ENTRYPOINT ["gunicorn", "-b", "0.0.0.0:8080", "app:app", "-n"]
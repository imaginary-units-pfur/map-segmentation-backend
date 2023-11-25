FROM python:3.11

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

WORKDIR /app
COPY app.py .
COPY model.py .

ENTRYPOINT gunicorn -w 4 -b 0.0.0.0 app:app --access-logfile /var/log/app.log

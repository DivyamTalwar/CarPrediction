FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/ models/
COPY api/ api/
COPY src/ src/

EXPOSE 8080

CMD ["python", "api/app.py"]

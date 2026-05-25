FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip && \
    pip install numpy>=1.21 pandas>=1.3 matplotlib>=3.4 scikit-learn>=1.0 xgboost>=1.5 && \
    mkdir -p results

ENTRYPOINT ["python", "main.py"]
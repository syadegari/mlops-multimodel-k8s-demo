FROM python:3.11-slim

# Make sure output is unbuffered (better logs)
ENV PYTHONUNBUFFERED=1

# Workdir inside the container
WORKDIR /app

# Install only what we need for inference
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    scikit-learn \
    numpy \
    joblib

# Only copy the FastAPI app into the image because the training happens outside the container
COPY app.py /app/app.py

EXPOSE 8000

# Start the server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

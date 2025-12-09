# CPU-only PyTorch + common vision deps, since we're only doing cpu-inference
FROM huggingface/transformers-torch-light

# Better logging behaviour
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# - fastapi + uvicorn for the API
# - transformers for the LM
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    transformers

COPY app.py /app/app.py

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

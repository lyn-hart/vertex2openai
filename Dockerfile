FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies
COPY app/requirements.txt .
RUN pip cache purge && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ .

# Expose the port (matches the port uvicorn listens on below)
EXPOSE 8050

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8050"]

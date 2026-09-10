FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies
COPY app/requirements.txt .
RUN pip cache purge && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ .

# Default port; overridden at runtime by PORT when the host platform sets one
# (Render and most PaaS providers inject PORT — we must bind to it).
EXPOSE 8050

# Command to run the application
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8050}"]

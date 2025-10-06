# Use official Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for psycopg2 and other packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Create uploads directory
RUN mkdir -p uploads

# Expose port 5000 for the Flask app
EXPOSE 5000

# Set environment variables
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1
ENV DATABASE_URL=postgresql://postgres:root@localhost:5432/retinoscan_db
# Run the app with gunicorn using eventlet worker
CMD ["gunicorn", "-k", "eventlet", "-w", "1", "app:app", "-b", "0.0.0.0:5000"]

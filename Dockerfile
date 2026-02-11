# Use Python 3.10
FROM python:3.10

# Set working directory
WORKDIR /code

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the application code
COPY . .

# Create a writable directory for transformers cache
RUN mkdir -p /code/cache
os.environ['TRANSFORMERS_CACHE'] = '/code/cache'
RUN chmod -R 777 /code/cache

# Expose port
EXPOSE 7860

# Run the application using Gunicorn for production
# Adjust workers/timeout as needed
CMD ["python", "backend.py"]

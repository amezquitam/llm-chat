
# Use Python Alpine as base image
FROM python:3.13 

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
# Expose port 7860
EXPOSE 7860

# Run the converted Python script
CMD ["python", "app.py"]
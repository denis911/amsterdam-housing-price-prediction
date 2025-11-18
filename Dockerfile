# Use official Python slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port used by FastAPI
EXPOSE 9696

# Run FastAPI service
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "9696", "--reload"]

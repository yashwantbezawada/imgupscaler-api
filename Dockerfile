# Use the official Python image as a base
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt into the container at /app/
COPY requirements.txt .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install specific PyTorch, torchvision, and torchaudio versions with CUDA support
RUN pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118

# Copy the FastAPI app code into the container
COPY app.py .

# Copy the models directory into the container
COPY models/ ./models/

# Expose port 8000 (Gunicorn will use this port)
EXPOSE 8001

# Command to run the FastAPI app with Gunicorn and Uvicorn workers in production
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "app:app", "--bind", "0.0.0.0:8001"]

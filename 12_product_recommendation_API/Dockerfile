# Base Image
FROM python:3.11

# Set Working Directory
WORKDIR /app

# Copy Files
COPY . .

# Install Dependencies
RUN pip install -r requirements.txt

# Expose Port
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
# Use lightweight Python image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip cache purge

# Copy the entire codebase into container
COPY . .

# Set environment variables inside Docker
ENV ARTIFACTS_DIR=Artifacts
ENV DATA_DIR=Data
ENV LOGS_DIR=Logs

# Expose Streamlit default port
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "Scripts/app.py"]

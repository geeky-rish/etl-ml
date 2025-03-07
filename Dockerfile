# Use a Debian-based Python image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED = 1

# Set the working directory
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    seaborn \
    matplotlib \
    scikit-learn \
    flask \
    optuna \
    pyarrow \
    joblib \
    pyarrow \
    fastparquet \
    pymongo

# Create necessary directories for artifacts
RUN mkdir -p /app/artifacts

# Copy the local files into the container
COPY . /app/

# Expose the port your app will run on
EXPOSE 5001

# Run the app
CMD ["python", "etl_mk4_draft_final.py"]

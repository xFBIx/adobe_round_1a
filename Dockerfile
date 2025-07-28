FROM --platform=linux/amd64 python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data during build
RUN python -c "import nltk; nltk.download('stopwords', quiet=True); nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# Copy application files
COPY process_pdfs.py .
COPY utils/ ./utils/
COPY models/ ./models/

# Create output directory
RUN mkdir -p /app/output

# Set executable permissions
RUN chmod +x process_pdfs.py

# Run the main script
CMD ["python", "process_pdfs.py"]

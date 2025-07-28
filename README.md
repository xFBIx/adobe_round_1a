# Challenge 1a: PDF Processing Solution

## Overview

This solution implements a machine learning-based PDF processing system for Challenge 1a of the Adobe India Hackathon 2025. The system extracts structured data from PDF documents and outputs JSON files with document titles and hierarchical outlines.

## Solution Architecture

### Machine Learning Approach

- **Model**: XGBoost Classifier for document structure classification
- **Features**: Typography, spatial, and structural features extracted from PDF text runs
- **Classes**: Predicts document elements as title, h1, h2, h3, or body text
- **Preprocessing**: Feature scaling, selection, and class balancing

### Key Libraries Used

- **PyMuPDF (fitz)**: PDF text extraction with formatting metadata
- **XGBoost**: Gradient boosting classifier for structure prediction
- **scikit-learn**: Feature preprocessing and model evaluation
- **NLTK**: Natural language processing for text analysis
- **NumPy/Pandas**: Data manipulation and numerical operations

## Project Structure

Challenge_1a_Submission/
├── Dockerfile
├── README.md
├── process_pdfs.py # Main prediction script
├── requirements.txt
├── models/
│ └── trained_model.pkl # Pre-trained ML model
└── utils/
└── pdf_classifier.py # Core classifier logic

## Features Extracted

### Typography Features

- Font size, font family, bold/italic formatting
- Font size relative to page median
- Typography consistency patterns

### Spatial Features

- Position normalization (x, y coordinates)
- Margins and spacing analysis
- Distance relationships between elements

### Structural Features

- Text length, word count, character patterns
- Numbering patterns (1., 1.1, bullets)
- Capitalization and punctuation patterns
- Stop word ratios

## Build and Run Instructions

### Build Command

docker build --platform linux/amd64 -t pdf-processor .

### Run Command

docker run --rm -v $(pwd)/input:/app/input:ro -v $(pwd)/output:/app/output --network none pdf-processor

## Performance Specifications

- **Processing Time**: <10 seconds for 50-page PDFs
- **Model Size**: <200MB (trained model included)
- **Memory Usage**: Optimized for 16GB RAM limit
- **CPU**: Efficient utilization of 8 CPU cores
- **Architecture**: AMD64 compatible

## Output Format

Each PDF generates a corresponding JSON file with:
{
"title": "Document Title",
"outline": [
{
"level": "H1",
"text": "Main Heading",
"page": 1
},
{
"level": "H2",
"text": "Sub Heading",
"page": 2
}
]
}

## Technical Implementation

### Model Training Pipeline

1. **Text Extraction**: Extract text runs with formatting metadata
2. **Feature Engineering**: Compute 20+ features per text element
3. **Data Augmentation**: Balance classes for better prediction
4. **Model Training**: XGBoost with cross-validation
5. **Feature Selection**: RFE for optimal feature subset

### Prediction Pipeline

1. Load pre-trained model and preprocessors
2. Extract text runs from input PDF
3. Compute same features as training
4. Apply preprocessing transformations
5. Predict document structure labels
6. Generate structured JSON output

## Key Optimizations

- **Efficient PDF Processing**: Batch processing with memory management
- **Feature Caching**: Optimized feature computation
- **Model Compression**: Compact model serialization
- **Error Handling**: Robust processing with fallback mechanisms
- **Duplicate Removal**: Clean output with deduplicated text elements

## Testing and Validation

- Cross-validated training with F1-macro scoring
- Overfitting detection and mitigation
- Performance profiling for time and memory constraints
- Tested on diverse PDF layouts and complexities

---

**Note**: This solution uses a pre-trained machine learning model optimized for document structure classification and meets all specified performance and resource constraints.

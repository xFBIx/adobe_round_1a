#!/usr/bin/env python3
"""
Adobe India Hackathon 2025 - Challenge 1a
PDF Processing Solution - Main Prediction Script
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any

from utils.pdf_classifier import PDFStructureClassifier, convert_predictions_to_json

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def process_single_pdf(
    classifier: PDFStructureClassifier, pdf_path: str, output_path: str
) -> bool:
    """Process a single PDF file and generate JSON output"""
    try:
        logger.info(f"Processing: {os.path.basename(pdf_path)}")

        # Make predictions
        result = classifier.predict_pdf(pdf_path)

        if "error" in result:
            logger.error(f"Error processing {pdf_path}: {result['error']}")
            return False

        if "predictions" not in result:
            logger.warning(f"No predictions generated for {pdf_path}")
            return False

        # Convert predictions to required JSON format
        json_output = convert_predictions_to_json(result["predictions"])

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save JSON output
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(json_output, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Successfully processed: {os.path.basename(pdf_path)}")
        logger.info(f"   Title: {json_output.get('title', 'N/A')}")
        logger.info(f"   Outline items: {len(json_output.get('outline', []))}")

        return True

    except Exception as e:
        logger.error(f"❌ Failed to process {pdf_path}: {str(e)}")
        return False


def process_pdfs():
    """Main function to process all PDFs in input directory"""
    # Define paths according to Docker mount requirements
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    model_path = Path("/app/models/trained_model.pkl")

    logger.info("Starting PDF processing pipeline...")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Model path: {model_path}")

    # Validate input directory
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return

    if not input_dir.is_dir():
        logger.error(f"Input path is not a directory: {input_dir}")
        return

    # Validate model file
    if not model_path.exists():
        logger.error(f"Model file does not exist: {model_path}")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the trained model
    try:
        logger.info("Loading trained model...")
        classifier = PDFStructureClassifier.load_model(str(model_path))
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        return

    # Find all PDF files in input directory
    pdf_files = list(input_dir.glob("*.pdf"))

    if not pdf_files:
        logger.warning("No PDF files found in input directory")
        return

    logger.info(f"Found {len(pdf_files)} PDF files to process")

    # Process each PDF file
    successful_count = 0
    failed_count = 0

    for pdf_file in pdf_files:
        # Generate output filename
        json_filename = f"{pdf_file.stem}.json"
        output_path = output_dir / json_filename

        # Process the PDF
        if process_single_pdf(classifier, str(pdf_file), str(output_path)):
            successful_count += 1
        else:
            failed_count += 1

    # Summary
    logger.info("=" * 50)
    logger.info("PROCESSING SUMMARY")
    logger.info(f"Total PDFs found: {len(pdf_files)}")
    logger.info(f"Successfully processed: {successful_count}")
    logger.info(f"Failed to process: {failed_count}")
    logger.info("=" * 50)

    if failed_count > 0:
        logger.warning(f"⚠️ {failed_count} files failed to process")
    else:
        logger.info("✅ All files processed successfully!")


if __name__ == "__main__":
    try:
        process_pdfs()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"🚨 Critical error in main process: {str(e)}")
        raise

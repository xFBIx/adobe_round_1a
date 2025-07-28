import json
import os
import re
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
import joblib

# PDF processing
import fitz  # PyMuPDF
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TextRun:
    """Represents a text run from PDF with associated metadata"""

    text: str
    font_size: float
    font_name: str
    is_bold: bool
    is_italic: bool
    x0: float
    y0: float
    x1: float
    y1: float
    page_num: int
    page_width: float
    page_height: float


class PDFStructureClassifier:
    def __init__(self):
        self.label_encoder = None
        self.scaler = None
        self.tfidf_vectorizer = None
        self.feature_selector = None
        self.model = None
        self.feature_names = []
        self.is_fitted = False

    def extract_text_runs(self, pdf_path: str) -> List[TextRun]:
        """Extract text runs with formatting metadata from PDF"""
        text_runs = []
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_width = page.rect.width
                page_height = page.rect.height

                # Get text blocks with formatting
                blocks = page.get_text("dict")
                for block in blocks["blocks"]:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                if span["text"].strip():
                                    text_runs.append(
                                        TextRun(
                                            text=span["text"].strip(),
                                            font_size=span["size"],
                                            font_name=span["font"],
                                            is_bold="Bold" in span["font"]
                                            or span["flags"] & 2**4,
                                            is_italic="Italic" in span["font"]
                                            or span["flags"] & 2**1,
                                            x0=span["bbox"][0],
                                            y0=span["bbox"][1],
                                            x1=span["bbox"][2],
                                            y1=span["bbox"][3],
                                            page_num=page_num + 1,
                                            page_width=page_width,
                                            page_height=page_height,
                                        )
                                    )
            doc.close()
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")

        return text_runs

    def compute_typography_features(
        self, text_runs: List[TextRun]
    ) -> List[Dict[str, float]]:
        """Compute typography-based features"""
        features = []

        # Calculate page-level statistics
        page_font_sizes = defaultdict(list)
        for run in text_runs:
            page_font_sizes[run.page_num].append(run.font_size)

        page_medians = {
            page: np.median(sizes) for page, sizes in page_font_sizes.items()
        }

        for run in text_runs:
            page_median = page_medians[run.page_num]
            feat = {
                # Typography features
                "font_size": run.font_size,
                "font_size_delta": run.font_size - page_median,
                "font_size_ratio": (
                    run.font_size / page_median if page_median > 0 else 1.0
                ),
                "is_bold": float(run.is_bold),
                "is_italic": float(run.is_italic),
                "font_name_hash": hash(run.font_name) % 1000,  # Simple font encoding
            }
            features.append(feat)

        return features

    def compute_spatial_features(
        self, text_runs: List[TextRun]
    ) -> List[Dict[str, float]]:
        """Compute spatial and positional features"""
        features = []

        for i, run in enumerate(text_runs):
            # Normalize coordinates
            x_center = (run.x0 + run.x1) / 2
            y_center = (run.y0 + run.y1) / 2

            feat = {
                # Position features
                "x_position_norm": x_center / run.page_width,
                "y_position_norm": y_center / run.page_height,
                "left_margin": run.x0 / run.page_width,
                "width_ratio": (run.x1 - run.x0) / run.page_width,
                "height_ratio": (run.y1 - run.y0) / run.page_height,
                # Context features
                "distance_from_top": y_center / run.page_height,
                "distance_from_left": x_center / run.page_width,
            }

            # Distance to previous/next elements
            if i > 0:
                prev_run = text_runs[i - 1]
                if prev_run.page_num == run.page_num:
                    feat["distance_to_prev"] = (
                        abs(run.y0 - prev_run.y1) / run.page_height
                    )
                else:
                    feat["distance_to_prev"] = 1.0
            else:
                feat["distance_to_prev"] = 1.0

            if i < len(text_runs) - 1:
                next_run = text_runs[i + 1]
                if next_run.page_num == run.page_num:
                    feat["distance_to_next"] = (
                        abs(next_run.y0 - run.y1) / run.page_height
                    )
                else:
                    feat["distance_to_next"] = 1.0
            else:
                feat["distance_to_next"] = 1.0

            features.append(feat)

        return features

    def compute_structural_features(
        self, text_runs: List[TextRun]
    ) -> List[Dict[str, float]]:
        """Compute structural and content-based features"""
        features = []

        try:
            stop_words = set(stopwords.words("english"))
        except:
            stop_words = set()

        for run in text_runs:
            text = run.text.strip()
            try:
                words = word_tokenize(text.lower())
            except:
                words = text.lower().split()

            feat = {
                # Text structure features
                "text_length": len(text),
                "word_count": len(words),
                "char_count": len(text),
                "uppercase_ratio": (
                    sum(1 for c in text if c.isupper()) / len(text) if text else 0
                ),
                "digit_ratio": (
                    sum(1 for c in text if c.isdigit()) / len(text) if text else 0
                ),
                "stopword_ratio": (
                    sum(1 for w in words if w in stop_words) / len(words)
                    if words
                    else 0
                ),
                # Numbering patterns
                "has_numbering": float(bool(re.match(r"^\d+\.", text.strip()))),
                "has_sub_numbering": float(bool(re.match(r"^\d+\.\d+", text.strip()))),
                "has_bullet": float(text.strip().startswith(("•", "-", "*"))),
                # Capitalization patterns
                "is_all_caps": float(text.isupper()),
                "is_title_case": float(text.istitle()),
                "starts_with_cap": float(text and text[0].isupper()),
                # Punctuation
                "ends_with_period": float(text.endswith(".")),
                "ends_with_colon": float(text.endswith(":")),
            }
            features.append(feat)

        return features

    def extract_features(
        self, text_runs: List[TextRun]
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """Extract all features and combine them"""
        if not text_runs:
            return np.array([]), [], []

        # Get different feature sets
        typo_features = self.compute_typography_features(text_runs)
        spatial_features = self.compute_spatial_features(text_runs)
        structural_features = self.compute_structural_features(text_runs)

        # Extract text for TF-IDF
        texts = [run.text for run in text_runs]

        # Combine non-text features
        all_features = []
        feature_names = []

        for i in range(len(text_runs)):
            combined = {}
            combined.update(typo_features[i])
            combined.update(spatial_features[i])
            combined.update(structural_features[i])
            all_features.append(combined)

        # Convert to DataFrame for easier handling
        df = pd.DataFrame(all_features)
        feature_names = list(df.columns)
        feature_matrix = df.values

        return feature_matrix, feature_names, texts

    @classmethod
    def load_model(cls, filepath: str) -> "PDFStructureClassifier":
        """Load a trained classifier from a file"""
        try:
            # Try loading with joblib first
            classifier = joblib.load(filepath)
            logger.info(f"Model loaded successfully from {filepath}")
            return classifier
        except Exception as e:
            logger.warning(f"Error loading model with joblib: {str(e)}")
            # Fallback to pickle
            try:
                with open(filepath, "rb") as f:
                    classifier = pickle.load(f)
                logger.info(f"Model loaded successfully from {filepath} using pickle")
                return classifier
            except Exception as e2:
                logger.error(f"Error loading model with pickle: {str(e2)}")
                raise e2

    def predict_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Load PDF and make predictions"""
        # Extract text runs from the PDF
        text_runs = self.extract_text_runs(pdf_path)
        if not text_runs:
            return {"error": "No text runs extracted from PDF"}

        # Extract features
        features, _, texts = self.extract_features(text_runs)
        if len(features) == 0:
            return {"error": "No features extracted from PDF"}

        # Apply same preprocessing as training
        features_selected = self.feature_selector.transform(features)
        features_scaled = self.scaler.transform(features_selected)

        # Make predictions
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)

        # Convert predictions back to labels
        predicted_labels = self.label_encoder.inverse_transform(predictions)

        # Combine results
        results = []
        for i, (text_run, label, prob) in enumerate(
            zip(text_runs, predicted_labels, probabilities)
        ):
            results.append(
                {
                    "text": text_run.text,
                    "predicted_label": label,
                    "confidence": float(np.max(prob)),
                    "page_num": text_run.page_num,
                    "position_x0": text_run.x0,
                    "position_y0": text_run.y0,
                    "position_x1": text_run.x1,
                    "position_y1": text_run.y1,
                    "font_size": text_run.font_size,
                    "font_name": text_run.font_name,
                    "is_bold": text_run.is_bold,
                    "is_italic": text_run.is_italic,
                }
            )

        # Create DataFrame for easier manipulation
        df = pd.DataFrame(results)

        # Remove duplicates based on text content (keeping first occurrence)
        df_deduplicated = df.drop_duplicates(subset=["text"], keep="first")

        logger.info(f"Original entries: {len(df)}")
        logger.info(f"After removing duplicates: {len(df_deduplicated)}")

        # Convert back to list of dictionaries for return value
        deduplicated_results = df_deduplicated.to_dict("records")

        return {
            "predictions": deduplicated_results,
            "total_elements": len(deduplicated_results),
            "original_elements": len(results),
            "duplicates_removed": len(results) - len(deduplicated_results),
            "label_distribution": dict(Counter(df_deduplicated["predicted_label"])),
        }


def convert_predictions_to_json(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Convert flat prediction dicts into structured JSON with title and outline."""
    title = None
    outline = []

    for pred in predictions:
        label = pred["predicted_label"]
        text = pred["text"]
        page = pred["page_num"]

        if label.lower() == "title" and not title:
            title = text
        elif label.upper() in {"H1", "H2", "H3"}:
            outline.append({"level": label.upper(), "text": text, "page": page})

    return {"title": title or "Unknown Title", "outline": outline}

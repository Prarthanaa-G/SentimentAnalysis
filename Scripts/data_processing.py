import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from helper_functions import log_info, log_error

PIPELINE_PATH = os.getenv("PIPELINE_PATH", "Artifacts/text_pipeline.pkl")
LABEL_ENCODER_PATH = os.getenv("LABEL_ENCODER_PATH", "Artifacts/label_encoder.pkl")

def create_text_pipeline():
    try:
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(stop_words='english', max_features=10000))
        ])
        log_info("Text processing pipeline created successfully.")
        return pipeline
    except Exception as e:
        log_error(f"Error creating text pipeline: {str(e)}")
        return None

def save_pipeline(pipeline):
    try:
        with open(PIPELINE_PATH, 'wb') as file:
            pickle.dump(pipeline, file)
        log_info(f"Pipeline saved at {PIPELINE_PATH}")
    except Exception as e:
        log_error(f"Failed to save pipeline: {str(e)}")

def encode_response_variable(y):
    try:
        mapping = {'negative': -1, 'neutral': 0, 'positive': 1}
        y_encoded = y.map(mapping) if hasattr(y, 'map') else [mapping[val] for val in y]
        with open(LABEL_ENCODER_PATH, 'wb') as f:
            pickle.dump(mapping, f)
        log_info(f"Custom label mapping saved at {LABEL_ENCODER_PATH}")
        return y_encoded
    except Exception as e:
        log_error(f"Failed to encode target variable: {str(e)}")
        return None

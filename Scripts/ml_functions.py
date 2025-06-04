import os
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from helper_functions import log_info, log_error

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'Artifacts')

LABEL_ENCODER_PATH = os.path.join(ARTIFACTS_DIR, "label_encoder.pkl")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "logistic_regression_model.pkl")

def load_model():
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        log_info("Model loaded successfully.")
        return model
    except FileNotFoundError as e:
        log_error(f"Model file not found: {e}")
        raise

def load_label_encoder():
    try:
        with open(LABEL_ENCODER_PATH, 'rb') as f:
            label_encoder = pickle.load(f)
        return label_encoder
    except FileNotFoundError as e:
        log_error(f"Label encoder file not found: {e}")
        raise

def training_pipeline(X_train, y_train):
    from sklearn.linear_model import LogisticRegression
    try:
        model = LogisticRegression(
        max_iter=5000,
        penalty='l2',
        class_weight='balanced'
    )
        model.fit(X_train, y_train)
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        log_info(f"Model trained and saved at {MODEL_PATH}")
        return model
    except Exception as e:
        log_error(f"Error during training: {e}")
        raise


def prediction_pipeline(X_val):
    try:
        model = load_model()
        label_encoder = load_label_encoder()
        preds_encoded = model.predict(X_val)
        preds_decoded = label_encoder.inverse_transform(preds_encoded)
        return preds_decoded
    except Exception as e:
        log_error(f"Prediction failed: {e}")
        raise

def evaluation_metrics(X_val, y_val_encoded):
    try:
        model = load_model()
        label_map = load_label_encoder()  # This is a dict

        preds_encoded = model.predict(X_val)

        # Manually invert the mapping
        reverse_map = {v: k for k, v in label_map.items()}
        preds_decoded = [reverse_map[p] for p in preds_encoded]
        true_labels_decoded = [reverse_map[y] for y in y_val_encoded]

        conf_matrix = confusion_matrix(true_labels_decoded, preds_decoded, labels=list(label_map.keys()))
        acc_score = accuracy_score(true_labels_decoded, preds_decoded)
        class_report = classification_report(true_labels_decoded, preds_decoded)
        print(conf_matrix, acc_score, class_report)
        log_info("Evaluation metrics computed successfully.")
        return conf_matrix, acc_score, class_report
    except Exception as e:
        log_error(f"Evaluation failed: {e}")
        raise
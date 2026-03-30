import os
import logging
import hashlib
import json
import joblib
import torch

logger = logging.getLogger(__name__)

SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_models")
os.makedirs(SAVE_DIR, exist_ok=True)


def _model_key(ticker: str, model_name: str, period: str, params: dict) -> str:
    """Generate a unique filename key for a trained model."""
    param_str = json.dumps(params, sort_keys=True) if params else ""
    raw = f"{ticker}:{model_name}:{period}:{param_str}"
    h = hashlib.md5(raw.encode()).hexdigest()[:10]
    return f"{ticker}_{model_name}_{h}"


def save_model(model, ticker: str, model_name: str, period: str, params: dict = None):
    key = _model_key(ticker, model_name, period, params or {})
    path = os.path.join(SAVE_DIR, key)

    try:
        if hasattr(model, "state_dict"):
            # PyTorch model
            torch.save({
                "state_dict": model.state_dict(),
                "config": {
                    "hidden_layer_size": model.hidden_layer_size,
                    "sequence_length": model.sequence_length,
                    "num_layers": model.num_layers,
                    "epochs": model.epochs,
                    "lr": model.lr,
                    "_input_size": model._input_size,
                },
                "scaler": model.scaler,
                "target_scaler": model.target_scaler,
                "_last_X_scaled": getattr(model, "_last_X_scaled", None),
            }, path + ".pt")
            logger.info("Saved PyTorch model to %s.pt", key)
        else:
            joblib.dump(model, path + ".joblib")
            logger.info("Saved model to %s.joblib", key)
        return True
    except Exception as e:
        logger.error("Failed to save model %s: %s", key, e)
        return False


def load_model(model_class, ticker: str, model_name: str, period: str, params: dict = None):
    key = _model_key(ticker, model_name, period, params or {})

    pt_path = os.path.join(SAVE_DIR, key + ".pt")
    joblib_path = os.path.join(SAVE_DIR, key + ".joblib")

    try:
        if os.path.exists(pt_path):
            checkpoint = torch.load(pt_path, weights_only=False)
            config = checkpoint["config"]
            model = model_class(
                hidden_layer_size=config["hidden_layer_size"],
                sequence_length=config["sequence_length"],
                num_layers=config["num_layers"],
                epochs=config["epochs"],
                lr=config["lr"],
            )
            if config["_input_size"]:
                model._build(config["_input_size"])
            model.load_state_dict(checkpoint["state_dict"])
            model.scaler = checkpoint["scaler"]
            model.target_scaler = checkpoint["target_scaler"]
            model._last_X_scaled = checkpoint.get("_last_X_scaled")
            logger.info("Loaded PyTorch model from %s.pt", key)
            return model
        elif os.path.exists(joblib_path):
            model = joblib.load(joblib_path)
            logger.info("Loaded model from %s.joblib", key)
            return model
    except Exception as e:
        logger.error("Failed to load model %s: %s", key, e)

    return None


def list_saved_models():
    files = os.listdir(SAVE_DIR)
    return [f.rsplit(".", 1)[0] for f in files if f.endswith((".joblib", ".pt"))]


def clear_saved_models():
    count = 0
    for f in os.listdir(SAVE_DIR):
        os.remove(os.path.join(SAVE_DIR, f))
        count += 1
    logger.info("Cleared %d saved models", count)
    return count

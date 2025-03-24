from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
import pandas as pd
from catboost import CatBoostClassifier
from pathlib import Path
import time
import logging
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Set Paths for Model and Data ===
ROOT_DIR = Path(__file__).resolve().parent.parent  # Correct project root
MODEL_PATH = ROOT_DIR / "models" / "cb_model.cbm"
DATA_PATH = ROOT_DIR / "data"  # Parquet files are inside "data/" in root

# === Load CatBoost Model ===
try:
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    logger.info("CatBoost model loaded successfully!")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise


# === Load Parquet Data Efficiently ===
def load_parquet_data(folder_path: Path):
    """Recursively load all Parquet files from partitioned folders."""
    try:
        # Search for all .parquet files in subdirectories
        files = list(folder_path.rglob("*.parquet"))  # Recursively find Parquet files

        if not files:
            logger.error(
                f"No Parquet files found in partitioned folders under {folder_path}"
            )
            return pd.DataFrame()  # Return empty DataFrame to prevent crashes

        logger.info(f"Found {len(files)} Parquet files in partitioned folders.")

        # Load all Parquet files and concatenate them into a single DataFrame
        df_list = [pd.read_parquet(file) for file in files]
        df = pd.concat(df_list, ignore_index=True)

        logger.info(
            f"Successfully loaded {len(df)} rows from {len(files)} partitioned Parquet files."
        )
        return df
    except Exception as e:
        logger.error(f"Error loading Parquet data: {e}")
        return pd.DataFrame()


# Load dataset at startup (cache it for faster inference)
data = load_parquet_data(DATA_PATH)


# === FastAPI App ===
def create_app(use_gzip: bool = True):
    app = FastAPI()
    # https://fastapi.tiangolo.com/advanced/middleware/#trustedhostmiddleware
    if use_gzip:
        app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=5)
        logger.info("GZIP Middleware Enabled.")
    else:
        logger.info("GZIP Middleware Disabled.")

    return app


# Enable/Disable GZIP Middleware dynamically
USE_GZIP = False  # Change to False to disable GZIP
app = create_app(use_gzip=USE_GZIP)


# === API Models ===
class PredictionRequest(BaseModel):
    num_rows: int = 1000  # Number of rows to use for inference


# === API Endpoint ===
@app.post("/predict")
async def predict(request: PredictionRequest):
    """Perform inference on the Parquet data."""
    global data

    if data.empty:
        logger.warning("⚠️ No data available for inference.")
        return {"error": "No data available for inference."}

    num_rows = min(request.num_rows, len(data))
    data_sample = data.head(num_rows)

    # Profiling inference
    start_time = time.perf_counter()
    predictions = model.predict(data_sample)
    end_time = time.perf_counter()

    latency = end_time - start_time  # More precise timing

    logger.info(f"🔍 Inference completed in {latency:.4f} seconds.")

    return {
        "num_rows": num_rows,
        "predictions": predictions.tolist(),
        "inference_time": latency,
    }


# === Run Server ===
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

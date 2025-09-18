import uvicorn
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager


class TransactionFeatures(BaseModel):
    amount: float
    time_of_day: int
    mismatch: int
    frequency: int


class PredictionResponse(BaseModel):
    probability: float
    is_fraudulent: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the PyTorch model at startup and attach it to app.state."""
    try:
        model = torch.jit.load("fraud_prevention_model.pt")
        model.eval()
        app.state.model = model
        print("PyTorch model loaded successfully.")
    except FileNotFoundError:
        print("Error: Model file 'fraud_prevention_model.pt' not found.")
        app.state.model = None
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        app.state.model = None

    # Placeholder mean and std; in a real scenario, load these from saved artifacts created during training
    app.state.mean = torch.tensor([0.0, 0.0, 0.0, 0.0])
    app.state.std = torch.tensor([1.0, 1.0, 1.0, 1.0])

    yield

    app.state.model = None


app = FastAPI(
    title="Fraud Prevention API",
    description="A simple API to predict fraudulent transactions using a PyTorch model.",
    version="1.0.0",
    lifespan=lifespan
)


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: TransactionFeatures):
    """
    Accepts transaction features and returns a fraud prediction.
    """
    model = app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    tensor = torch.tensor([[features.amount, features.time_of_day, features.mismatch, features.frequency]], dtype=torch.float32)
    # Normalisation
    n_tensor = (tensor - app.state.mean) / app.state.std
    probability = torch.sigmoid(model(n_tensor)).item()

    return PredictionResponse(probability=probability, is_fraudulent=(probability > 0.5))


if __name__ == "__main__":
    # This block allows you to run the server directly from the command line using `python part2.py`.
    # The server will be available at http://127.0.0.1:8000
    uvicorn.run(app, host="127.0.0.1", port=8000)

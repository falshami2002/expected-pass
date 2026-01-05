from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from model import load_xpass_model
from inference import predict_xpass
from schemas import PredictRequest, PredictResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_xpass_model("xpass_best_weights.weights.h5")
    yield


app = FastAPI(
    title="xPass API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if req.passer_index == req.receiver_index:
        return PredictResponse(xpass=0.0)

    xpass = predict_xpass(
        app.state.model,
        [p.model_dump() for p in req.players],
        req.ball.model_dump(),
        passer_index=req.passer_index,
        receiver_index=req.receiver_index,
    )
    return PredictResponse(xpass=xpass)

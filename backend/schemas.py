from pydantic import BaseModel, Field
from typing import List, Literal


class PlayerPlacement(BaseModel):
    x: float = Field(..., ge=0.0, le=1.0)
    y: float = Field(..., ge=0.0, le=1.0)
    team: Literal["attack", "defense"]


class BallPlacement(BaseModel):
    x: float = Field(..., ge=0.0, le=1.0)
    y: float = Field(..., ge=0.0, le=1.0)


class PredictRequest(BaseModel):
    players: List[PlayerPlacement] = Field(..., min_length=22, max_length=22)
    ball: BallPlacement
    passer_index: int = Field(..., ge=0, le=21)
    receiver_index: int = Field(..., ge=0, le=21)


class PredictResponse(BaseModel):
    xpass: float

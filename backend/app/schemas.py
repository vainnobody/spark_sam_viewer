from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class CameraPayload(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    aspect: float = Field(gt=0)
    fovY: float = Field(gt=0)
    near: float = Field(gt=0)
    far: float = Field(gt=0)
    position: list[float] = Field(min_length=3, max_length=3)
    quaternion: list[float] = Field(min_length=4, max_length=4)
    viewMatrix: list[list[float]] = Field(min_length=4, max_length=4)
    projectionMatrix: list[list[float]] = Field(min_length=4, max_length=4)

    @field_validator("viewMatrix", "projectionMatrix")
    @classmethod
    def validate_matrix(cls, value: list[list[float]]) -> list[list[float]]:
        if any(len(row) != 4 for row in value):
            raise ValueError("Matrix rows must all have length 4.")
        return value


class PromptPoint(BaseModel):
    world: list[float] = Field(min_length=3, max_length=3)
    label: Literal[1, -1]


class PreviewRequest(BaseModel):
    camera: CameraPayload
    points: list[PromptPoint]


class CommitRequest(BaseModel):
    op: Literal["union", "invert", "reset"]


class SessionCreated(BaseModel):
    sessionId: str
    numSplats: int
    bbox: dict[str, list[float]]
    warnings: list[str]


class PreviewResponse(BaseModel):
    previewImage: str
    previewCount: int
    previewMaskBitset: str
    visibleCount: int
    visibleMaskBitset: str


class CommitResponse(BaseModel):
    visibleCount: int
    visibleMaskBitset: str

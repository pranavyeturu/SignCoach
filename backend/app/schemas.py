from pydantic import BaseModel


class FrameRequest(BaseModel):
    image: str


class DatasetSummary(BaseModel):
    path: str
    classes: list[str]
    image_count: int


class CalibrationSampleRequest(BaseModel):
    label: str
    landmarks: list[dict[str, float]]


class PhraseAnalyzeRequest(BaseModel):
    phrase_id: str
    frames: list[list[dict[str, float]]]

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_serializer, model_validator


class BBox(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    left: int = Field(ge=0)
    top: int = Field(ge=0)
    width: int = Field(ge=0)
    height: int = Field(ge=0)


class Detection(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    bbox: BBox
    confidence: float = Field(ge=0.0, le=1.0)
    class_id: int = Field(ge=0)
    class_name: str | None = None
    track_id: int | None = None


class PersonCountResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    task: Literal["person_counting"] = "person_counting"
    count: int = Field(ge=0)
    detections: list[Detection] = Field(default_factory=list)
    raw_prediction_count: int = Field(ge=0)
    executed_at: int = Field(ge=0)
    latency_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_detection_counts(self) -> "PersonCountResponse":
        if self.count != len(self.detections):
            raise ValueError(
                f"count ({self.count}) must match detections length ({len(self.detections)})"
            )
        if self.raw_prediction_count < self.count:
            raise ValueError(
                "raw_prediction_count must be greater than or equal to count"
            )
        return self


class PersonDetectionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    label: Literal["person"] = "person"
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: list[int]
    annotated_image_uri: str | None = None
    executed_at: int = Field(ge=0)
    latency_ms: int = Field(ge=0)

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, value: list[int]) -> list[int]:
        if len(value) != 4:
            raise ValueError("bbox must contain [left, top, width, height]")
        if any(v < 0 for v in value):
            raise ValueError("bbox values must be >= 0")
        return value

    @model_serializer(mode="wrap")
    def serialize_without_nulls(self, handler):
        data = handler(self)
        if data.get("annotated_image_uri") is None:
            data.pop("annotated_image_uri", None)
        return data


class FaceDetection(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    bbox: list[int]
    confidence: float = Field(ge=0.0, le=1.0)
    landmarks: list[list[int]] | None = None
    age: int | None = Field(default=None, ge=0)
    gender: Literal["M", "F"] | None = None

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, value: list[int]) -> list[int]:
        if len(value) != 4:
            raise ValueError("bbox must contain [x1, y1, x2, y2]")
        if any(v < 0 for v in value):
            raise ValueError("bbox values must be >= 0")
        return value

    @field_validator("landmarks")
    @classmethod
    def validate_landmarks(cls, value: list[list[int]] | None) -> list[list[int]] | None:
        if value is None:
            return None
        for pair in value:
            if len(pair) != 2:
                raise ValueError("each landmark must contain exactly [x, y]")
            if any(point < 0 for point in pair):
                raise ValueError("landmark coordinates must be >= 0")
        return value

    @model_serializer(mode="wrap")
    def serialize_without_nulls(self, handler):
        data = handler(self)
        if data.get("landmarks") is None:
            data.pop("landmarks", None)
        if data.get("age") is None:
            data.pop("age", None)
        if data.get("gender") is None:
            data.pop("gender", None)
        return data


class FaceDetectionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    faces: list[FaceDetection] = Field(default_factory=list)
    face_count: int = Field(ge=0)
    annotated_image_uri: str | None = None
    executed_at: int = Field(ge=0)
    latency_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_face_count(self) -> "FaceDetectionResponse":
        if self.face_count != len(self.faces):
            raise ValueError(
                f"face_count ({self.face_count}) must match faces length ({len(self.faces)})"
            )
        return self

    @model_serializer(mode="wrap")
    def serialize_without_nulls(self, handler):
        data = handler(self)
        if data.get("annotated_image_uri") is None:
            data.pop("annotated_image_uri", None)
        return data


class FaceRecognitionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    recognized: bool
    person_id: str | None = None
    name: str | None = None
    category: str | None = None
    similarity: float | None = Field(default=None, ge=0.0, le=1.0)
    face_bbox: list[int] | None = None
    message: str | None = None
    latency_ms: int = Field(ge=0)

    @field_validator("face_bbox")
    @classmethod
    def validate_face_bbox(cls, value: list[int] | None) -> list[int] | None:
        if value is None:
            return None
        if len(value) != 4:
            raise ValueError("face_bbox must contain [x1, y1, x2, y2]")
        if any(v < 0 for v in value):
            raise ValueError("face_bbox values must be >= 0")
        return value

    @model_validator(mode="after")
    def validate_recognition_payload(self) -> "FaceRecognitionResponse":
        if self.recognized and (
            self.person_id is None or self.name is None or self.similarity is None
        ):
            raise ValueError(
                "person_id, name, and similarity are required when recognized is true"
            )
        return self

    @model_serializer(mode="wrap")
    def serialize_without_nulls(self, handler):
        data = handler(self)
        for key in ("person_id", "name", "category", "similarity", "face_bbox", "message"):
            if data.get(key) is None:
                data.pop(key, None)
        return data


class SceneDescriptionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    task: Literal["scene_description"] = "scene_description"
    caption: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    executed_at: int = Field(ge=0)
    latency_ms: int = Field(ge=0)


class TranscriptSegment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    start: float = Field(ge=0.0)
    end: float = Field(ge=0.0)
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_span(self) -> "TranscriptSegment":
        if self.end < self.start:
            raise ValueError(f"segment end ({self.end}) must be >= start ({self.start})")
        return self

    @model_serializer(mode="wrap")
    def serialize_without_nulls(self, handler):
        data = handler(self)
        for key in ("avg_logprob", "no_speech_prob"):
            if data.get(key) is None:
                data.pop(key, None)
        return data


class AudioTranscriptionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    task: Literal["audio_transcription", "audio_translation"]
    text: str
    segments: list[TranscriptSegment] = Field(default_factory=list)
    language: str = Field(min_length=1)
    language_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    duration_seconds: float = Field(ge=0.0)
    model: str = Field(min_length=1)
    translated_to_english: bool = False
    executed_at: int = Field(ge=0)
    latency_ms: int = Field(ge=0)

    @model_serializer(mode="wrap")
    def serialize_without_nulls(self, handler):
        data = handler(self)
        if data.get("language_confidence") is None:
            data.pop("language_confidence", None)
        return data


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatCompletionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    task: Literal["chat_completion"] = "chat_completion"
    message: ChatMessage
    model: str = Field(min_length=1)
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter", "unknown"] = "stop"
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    executed_at: int = Field(ge=0)
    latency_ms: int = Field(ge=0)

    @model_serializer(mode="wrap")
    def serialize_without_nulls(self, handler):
        data = handler(self)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if data.get(key) is None:
                data.pop(key, None)
        return data


class SpeechSynthesisResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    task: Literal["speech_synthesis"] = "speech_synthesis"
    audio_uri: str = Field(min_length=1)
    duration_seconds: float = Field(ge=0.0)
    sample_rate: int = Field(gt=0)
    voice: str = Field(min_length=1)
    text_length: int = Field(ge=0)
    executed_at: int = Field(ge=0)
    latency_ms: int = Field(ge=0)

    @field_validator("audio_uri")
    @classmethod
    def validate_audio_uri(cls, value: str) -> str:
        if not value.startswith("opennvr://audio/"):
            raise ValueError("audio_uri must start with opennvr://audio/")
        return value

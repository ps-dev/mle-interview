import enum
import uuid
from typing import List

from pydantic import BaseModel, Field


class UserType(enum.Enum):
    B2B = "B2B"
    B2C = "B2C"


class UserInterestsRequest(BaseModel):
    user_handle: str = Field(min_length=5)


class Interest(BaseModel):
    id: uuid.UUID
    label: str
    probability: float


class UserInterestsResponse(BaseModel):
    user_handle: uuid.UUID
    interests: List[Interest]
    name: str
    type: UserType


class UserFeatureStoreRecord(BaseModel):
    id: uuid.UUID
    name: str
    type: UserType
    content_titles: str


class InterestsModelFeatures(BaseModel):
    user_handle: uuid.UUID
    user_type: UserType
    content_titles: str


class ModelUserInterest(BaseModel):
    probability: float
    id: uuid.UUID

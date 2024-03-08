import uuid
from typing import List

import requests

from .types import InterestsModelFeatures, ModelUserInterest


def get_interests_from_model(
    inputs: InterestsModelFeatures, top_k: int
) -> List[ModelUserInterest]:
    url = "http://localhost:8501/v1/models/interests-model:predict"

    payload = {
        "inputs": {
            "USER_HANDLE": [str(inputs.user_handle)],
            "USER_TYPE": [str(inputs.user_type)],
            "CONTENT_TITLES_JOINED": ["".join(inputs.content_titles)],
            "top_k": top_k,
        }
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()

    data = response.json()

    probabilities = data["outputs"]["probabilities"][0]
    interest_ids = data["outputs"]["interests_ids"][0]

    model_user_interests = []

    for prob, interest_id in zip(probabilities, interest_ids):
        model_user_interests.append(
            ModelUserInterest(probability=prob, id=uuid.UUID(interest_id))
        )

    return model_user_interests

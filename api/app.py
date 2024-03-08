import uuid
from typing import cast

from flask import Flask, request
from flask_pydantic import validate

from .data import FeatureStore, UserFeatureStoreRecord
from .integrations import InterestsModelFeatures, get_interests_from_model
from .types import UserInterestsResponse

app = Flask(__name__)

app.feature_store = FeatureStore()


@app.route("/interests/<user_handle>", methods=["GET"])
@validate()
def get_interests(user_handle: uuid.UUID) -> UserInterestsResponse:
    features = app.feature_store.get_user_features(user_handle)
    top_k = request.args.get("top_k", default=10, type=int)

    features = cast(UserFeatureStoreRecord, features)

    user_features = InterestsModelFeatures(
        user_handle=features.id,
        user_type=features.type,
        content_titles=features.content_titles,
    )

    top_interests = get_interests_from_model(user_features, top_k=top_k)
    ids = [interest.id for interest in top_interests]

    labels = app.feature_store.get_interests_labels(ids)

    return UserInterestsResponse(
        user_handle=user_handle,
        name=features.name,
        type=features.type,
        interests=labels,
    )


@app.route("/")
def hello_world():
    return {"status": "healthy"}

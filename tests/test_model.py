import random
import uuid

import numpy as np
import tensorflow as tf
from faker import Faker

from model.model import UserInterestsModel


class TestModel:
    def setup_method(self):
        faker = Faker()

        self.content_titles = [faker.text() for _ in range(0, 10)]
        self.users = [str(uuid.uuid4()) for _ in range(0, 10)]
        self.interests_vocab = list(set([faker.color_name() for _ in range(0, 10)]))

        user_types = ["B2B", "B2C"]

        hyper_params = {
            "embedding_output_dims": 124,
            "hidden_units": [128, 64],
            "learning_rate": 0.01,
        }

        self.model = UserInterestsModel(
            content_titles_vocab=self.content_titles,
            user_handles_vocab=self.users,
            interests_vocab=self.interests_vocab,
            user_types=user_types,
            hyper_params=hyper_params,
        )

    def test_probability_scores(self):
        input_record = {
            "USER_HANDLE": tf.constant([self.users[0]]),
            "USER_TYPE": tf.constant(["B2B"]),
            "CONTENT_TITLES_JOINED": tf.constant([self.content_titles[0]]),
        }
        top_k = tf.constant(1, dtype=tf.int64)

        result = self.model.serving_predict(input_record, top_k)
        probabilities = result["probabilities"].numpy().flatten()

        assert all(0 <= p <= 1 for p in probabilities), (
            "Probabilities must be between 0 and 1"
        )

    def test_model_build(self):
        interests_input = [self.interests_vocab[0], self.interests_vocab[3]]

        encoded_interests = self.model.target_interests_lookup(
            tf.constant([interests_input])
        )

        assert encoded_interests.shape == (
            1,
            len(self.interests_vocab),
        ), "Incorrect shape size"

        assert np.allclose(
            encoded_interests,
            np.array(
                [[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32
            ),
        )

from typing import Any, Dict, Tuple

import pandas as pd
import tensorflow as tf


def create_training_example(
    input: Dict[str, Any],
) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    user_input = {
        "CONTENT_TITLES_JOINED": input["CONTENT_TITLES_JOINED"],
        "USER_HANDLE": input["USER_HANDLE"],
        "USER_TYPE": input["USER_TYPE"],
    }

    target_interests = input["USER_INTERESTS"]

    return (user_input, target_interests)


def get_tf_dataset(dataset: pd.DataFrame, batch_size) -> tf.data.Dataset:
    user_features = dataset[
        ["CONTENT_TITLES_JOINED", "USER_HANDLE", "USER_TYPE"]
    ].to_dict(orient="list")

    user_features["USER_INTERESTS"] = tf.ragged.constant(
        dataset["USER_INTERESTS"].to_list()
    )

    tf_dataset = tf.data.Dataset.from_tensor_slices(user_features)

    tf_dataset = tf_dataset.map(
        create_training_example, num_parallel_calls=tf.data.AUTOTUNE
    )

    return (
        tf_dataset.apply(tf.data.experimental.dense_to_ragged_batch(batch_size))
        .shuffle(buffer_size=1_024)
        .prefetch(tf.data.AUTOTUNE)
    )

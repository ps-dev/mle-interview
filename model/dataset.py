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


def get_tf_dataset(dataset: pd.DataFrame, batch_size: str) -> tf.data.Dataset:
    """Creates a TensorFlow Dataset from a pandas DataFrame containing user data.

    Args: dataset (pd.DataFrame): The input pandas DataFrame with user data. batch_size (str): The batch size for batching the dataset.

    Returns: tf.data.Dataset: A TensorFlow Dataset with user features and interests.

    The function extracts user features and interests from the input DataFrame, maps them to create training examples,
    batches the examples into ragged tensors, shuffles the dataset, and prefetches data for efficient processing
    before returning the transformed TensorFlow Dataset.
    """
    user_features = dataset[
        ["CONTENT_TITLES_JOINED", "USER_HANDLE", "USER_TYPE"]
    ].to_dict(orient="list")

    user_features["USER_INTERESTS"] = tf.ragged.constant(
        dataset["USER_INTERESTS"].to_list()
    )

    tf_dataset = tf.data.Dataset.from_tensor_slices(user_features)

    return (
        tf_dataset.map(create_training_example, num_parallel_calls=tf.data.AUTOTUNE)
        .apply(tf.data.experimental.dense_to_ragged_batch(batch_size))
        .shuffle(buffer_size=1_024)
        .prefetch(tf.data.AUTOTUNE)
    )

import itertools
from ast import literal_eval

import pandas as pd
import tensorflow as tf

from .dataset import get_tf_dataset
from .model import ProductRankerModel

dataset = pd.read_csv("/workspaces/ml-interview/model/user_interactions.csv.gz", compression='gzip')

dataset["USER_INTERESTS"] = dataset["USER_INTERESTS"].map(lambda x: literal_eval(x))

content_titles = dataset["CONTENT_TITLES_JOINED"]
user_handles_vocab = dataset["USER_HANDLE"].unique()
user_types = dataset["USER_TYPE"].unique()

interests_vocab = list(set(itertools.chain.from_iterable(dataset["USER_INTERESTS"])))

training_dataset = get_tf_dataset(dataset, 64)

hyper_params = {
    "embedding_output_dims": 124,
    "hidden_units": [128, 64],
    "learning_rate": 0.01,
}

model = ProductRankerModel(
    interests_vocab=interests_vocab,
    user_handles_vocab=user_handles_vocab,
    content_titles_vocab=content_titles,
    user_types=user_types,
    hyper_params=hyper_params,
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=hyper_params["learning_rate"]),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[
        tf.keras.metrics.BinaryCrossentropy(from_logits=True),
        tf.keras.metrics.TopKCategoricalAccuracy(k=10),
    ],
)

callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir="./training_logs"),
]

model.fit(
    training_dataset.take(100).cache(),
    epochs=2,
    callbacks=callbacks,
)

tf.saved_model.save(
    model,
    "serving/interests-model/1",
    signatures={"serving_default": model.serving_predict},
)

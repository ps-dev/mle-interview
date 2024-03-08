from typing import Any, Dict, List, Optional, Union

import tensorflow as tf
from tensorflow import keras

SERVING_TOPK_INPUT_SIG = (
    {
        "USER_HANDLE": tf.TensorSpec(
            shape=(None,), dtype=tf.string, name="USER_HANDLE"
        ),
        "USER_TYPE": tf.TensorSpec(shape=(None,), dtype=tf.string, name="USER_TYPE"),
        "CONTENT_TITLES_JOINED": tf.TensorSpec(
            shape=(None,), dtype=tf.string, name="CONTENT_TITLES_JOINED"
        ),
    },
    tf.TensorSpec(shape=(None), dtype=tf.int64, name="top_k"),
)


class UserInterestsModel(tf.keras.Model):
    """
    A model that predicts user interests based on their content interaction and user attributes.
    It utilizes embeddings and dense layers to generate a vector representation of user interests.

    Attributes:
        content_titles_vocab (List[str]): Vocabulary of content titles.
        user_handles_vocab (List[str]): Vocabulary of user handles.
        interests_vocab (List[str]): Vocabulary of interests.
        user_types (List[str]): List of user types.
        hyper_params (Dict[str, Any]): Hyperparameters for the model.
        mask_token (Optional[str]): Token used for masking, if any.
    """

    def __init__(
        self,
        content_titles_vocab: List[str],
        user_handles_vocab: List[str],
        interests_vocab: List[str],
        user_types: List[str],
        hyper_params: Dict[str, Any],
        mask_token: Optional[str] = None,
    ):
        super(UserInterestsModel, self).__init__()

        self.hyper_params = hyper_params

        self.content_title_tokenizer = keras.layers.TextVectorization()
        self.content_title_tokenizer.adapt(content_titles_vocab)

        self.content_title_embeddings = keras.layers.Embedding(
            input_dim=self.content_title_tokenizer.vocabulary_size(),
            output_dim=hyper_params["embedding_output_dims"],
            name="content_title_embeddings",
        )

        self.user_type_lookup = keras.layers.StringLookup(
            vocabulary=user_types,
            num_oov_indices=1,
            mask_token=mask_token,
            name="user_type_lookup",
        )
        self.user_type = keras.layers.CategoryEncoding(
            num_tokens=self.user_type_lookup.vocabulary_size(),
            output_mode="one_hot",
            name="user_type",
        )

        self.user_handle_lookup = keras.layers.StringLookup(
            vocabulary=user_handles_vocab,
            num_oov_indices=1,
            mask_token=mask_token,
            name="user_handle_lookup",
        )
        self.user_embeddings = keras.layers.Embedding(
            input_dim=self.user_handle_lookup.vocabulary_size(),
            output_dim=hyper_params["embedding_output_dims"],
            name="user_embeddings",
        )

        self.target_interests_lookup = keras.layers.StringLookup(
            vocabulary=interests_vocab,
            num_oov_indices=0,
            mask_token=mask_token,
            output_mode="one_hot",
            name="target_interests_lookup",
        )

        self.index_to_interest_lookup = keras.layers.StringLookup(
            vocabulary=interests_vocab,
            num_oov_indices=0,
            mask_token=mask_token,
            output_mode="int",
            invert=True,
            name="index_to_interest_lookup",
        )

        self.hidden_layers = keras.Sequential()

        for layer in hyper_params["hidden_units"]:
            self.hidden_layers.add(keras.layers.Dense(layer, activation="relu"))

        self.normalization_layer = keras.layers.LayerNormalization(axis=-1)

        self.output_layer = tf.keras.layers.Dense(
            self.index_to_interest_lookup.vocabulary_size(),
            activation=None,
            use_bias=False,
        )

    def get_config(self):
        return {"hyper_params": self.hyper_params}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, input: Dict[str, tf.Tensor]) -> tf.Tensor:
        """
        Forward pass for the model.

        Args:
            input (Dict[str, tf.Tensor]): A dictionary containing input tensors.

        Returns:
            tf.Tensor: The output tensor of the model.
        """
        query_tokenize = self.content_title_tokenizer(input["CONTENT_TITLES_JOINED"])
        query_embedding = self.content_title_embeddings(query_tokenize)

        query_mean = tf.math.reduce_mean(query_embedding, axis=1)

        channel_encoding = self.user_type(
            self.user_type_lookup(tf.expand_dims(input["USER_TYPE"], axis=1))
        )

        visitor_embedding = self.user_embeddings(
            self.user_handle_lookup(input["USER_HANDLE"])
        )

        user_features = tf.concat(
            [query_mean, channel_encoding, visitor_embedding], axis=-1
        )

        user_vector = self.hidden_layers(user_features)
        user_vector_norm = self.normalization_layer(user_vector)

        return self.output_layer(user_vector_norm)

    def train_step(self, dataset: tf.data.Dataset) -> Dict[str, Union[float, int, str]]:
        """
        Custom training step for the model.

        Args:
            dataset (tf.data.Dataset): The dataset to train on.

        Returns:
            Dict[str, Union[float, int, str]]: A dictionary containing metric results.
        """
        (inputs, targets) = dataset

        target_tensor = self.target_interests_lookup(targets)

        with tf.GradientTape() as tape:
            output = self.call(inputs)
            batch_loss = self.loss(target_tensor, output)

        grads = tape.gradient(batch_loss, self.trainable_weights)

        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.compiled_metrics.update_state(target_tensor, output)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, dataset: tf.data.Dataset) -> Dict[str, Union[float, int, str]]:
        (inputs, targets) = dataset
        output = self.call(inputs)
        target_tensor = self.target_interests_lookup(targets)

        self.compiled_metrics.update_state(target_tensor, output)

        return {m.name: m.result() for m in self.metrics}

    @tf.function(input_signature=SERVING_TOPK_INPUT_SIG)
    def serving_predict(
        self, input_record: Dict[str, tf.Tensor], top_k: tf.Tensor = 5
    ) -> Dict[str, tf.Tensor]:
        """
        Generates predictions for serving with an option to return top K results.

        Args:
            input_record (Dict[str, tf.Tensor]): Input data for prediction.
            top_k (tf.Tensor): Tensor indicating the number of top results to return.

        Returns:
            Dict[str, tf.Tensor]: A dictionary with keys 'interests_ids' and 'probabilities',
                                   containing the top K interests and their corresponding probabilities.
        """

        if top_k.shape.rank == 1:
            top_k = tf.squeeze(top_k)

        output = self.call(input_record)

        logits, indices = tf.math.top_k(output, tf.cast(top_k, tf.int32))

        top_k_product_ids = self.index_to_interest_lookup(tf.squeeze(indices))

        return {
            "interests_ids": tf.expand_dims(top_k_product_ids, 0),
            "probabilities": logits,
        }

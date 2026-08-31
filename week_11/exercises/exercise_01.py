"""
DATASCI 207 - Module 11 Exercise: Wire a Wider Model

Six inputs (incl. an engineered, bucketed family_size), Functional API,
hand-predicted parameter count, one-hot vs embedding comparison.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

np.random.seed(42)
tf.random.set_seed(42)

df = sns.load_dataset("titanic")[
    ["survived", "age", "fare", "sex", "pclass", "embarked", "sibsp", "parch"]
].copy()
df["age"] = df["age"].fillna(df["age"].median())
df["embarked"] = df["embarked"].fillna("S")
df["family_size"] = df["sibsp"] + df["parch"] + 1

idx = np.random.permutation(len(df))
cut = int(0.8 * len(df))
train, dev = df.iloc[idx[:cut]].copy(), df.iloc[idx[cut:]].copy()
y_train = train["survived"].values.astype("float32")
y_dev = dev["survived"].values.astype("float32")
print(len(train), "train /", len(dev), "dev")
# =============================================================================
# TODO 1: ENCODING LAYERS
# =============================================================================
# TODO 1: Build the encoding layers.
# - norm_age, norm_fare: Normalization layers, adapted on the TRAINING data only
# - lookup_sex: StringLookup(vocabulary=["male", "female"], output_mode="one_hot")
# - lookup_emb: StringLookup(vocabulary=["S", "C", "Q"], output_mode="one_hot")
# - buck_family: Discretization(bin_boundaries=[1.5, 2.5, 4.5], output_mode="one_hot")
#   (buckets: alone / pair / small family / large family)
# - lookup_cls: IntegerLookup(vocabulary=[1, 2, 3], output_mode="one_hot")

norm_age = ...
norm_fare = ...
lookup_sex = ...
lookup_emb = ...
buck_family = ...
lookup_cls = ...
# =============================================================================
# TODO 2: WIRE THE MODEL
# =============================================================================
# TODO 2: Wire the six-input model with the Functional API.
# Named inputs: age, fare (float), sex, embarked (string), pclass, family_size (int64 / float).
# Route each through its encoder, Flatten the one-hot outputs, Concatenate,
# then Dense(16, relu) -> Dense(1, sigmoid, name="survived").
# Build `model = keras.Model(inputs=[...], outputs=out)`.

model = ...
# =============================================================================
# TODO 3: PREDICT THE BILL, THEN PAY IT
# =============================================================================
# TODO 3: Predict the parameter count BEFORE running this cell.
# Work it out on paper: concat width -> (width + 1) * 16 + 17.
# Then compile, train 30 epochs (verbose=0), and evaluate on dev.

paper_count = ...   # your hand computation, an integer

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
# model.fit(...)
# print evaluate(...) and the sum of trainable_weights shapes; it should equal paper_count
# (count_params() also includes Normalization's frozen mean/variance - trainable only!)
# =============================================================================
# TODO 4: EMBEDDING SWAP
# =============================================================================
# TODO 4: Swap embarked's one-hot for an Embedding(input_dim=4, output_dim=4)
# (StringLookup WITHOUT output_mode, then Embedding, then Flatten).
# Rebuild, retrain, and answer in a comment:
#   - how did count_params() change, and where did the change come from?
#   - did dev accuracy move enough to justify the parameters?

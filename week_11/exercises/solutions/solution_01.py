"""
DATASCI 207 - Module 11 Solution: Wire a Wider Model
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


def feed(frame):
    return {
        "age": frame[["age"]].values.astype("float32"),
        "fare": frame[["fare"]].values.astype("float32"),
        "sex": frame["sex"].values,
        "embarked": frame["embarked"].values,
        "pclass": frame["pclass"].values,
        "family_size": frame[["family_size"]].values.astype("float32"),
    }


# ============================================================================
# TODO 1: ENCODING LAYERS
# ============================================================================

norm_age = layers.Normalization(); norm_age.adapt(feed(train)["age"])
norm_fare = layers.Normalization(); norm_fare.adapt(feed(train)["fare"])
lookup_sex = layers.StringLookup(vocabulary=["male", "female"], output_mode="one_hot")
lookup_emb = layers.StringLookup(vocabulary=["S", "C", "Q"], output_mode="one_hot")
buck_family = layers.Discretization(bin_boundaries=[1.5, 2.5, 4.5], output_mode="one_hot")
lookup_cls = layers.IntegerLookup(vocabulary=[1, 2, 3], output_mode="one_hot")

# ============================================================================
# TODO 2: WIRE THE MODEL
# ============================================================================

in_age = keras.Input(shape=(1,), name="age")
in_fare = keras.Input(shape=(1,), name="fare")
in_sex = keras.Input(shape=(1,), dtype="string", name="sex")
in_emb = keras.Input(shape=(1,), dtype="string", name="embarked")
in_cls = keras.Input(shape=(1,), dtype="int64", name="pclass")
in_fam = keras.Input(shape=(1,), name="family_size")

paths = [
    norm_age(in_age),                       # width 1
    norm_fare(in_fare),                     # width 1
    layers.Flatten()(lookup_sex(in_sex)),   # width 3 (vocab 2 + OOV)
    layers.Flatten()(lookup_emb(in_emb)),   # width 4 (vocab 3 + OOV)
    layers.Flatten()(lookup_cls(in_cls)),   # width 4 (vocab 3 + OOV)
    layers.Flatten()(buck_family(in_fam)),  # width 4 (3 boundaries -> 4 buckets)
]
trunk = layers.Dense(16, activation="relu")(layers.Concatenate()(paths))
out = layers.Dense(1, activation="sigmoid", name="survived")(trunk)
model = keras.Model([in_age, in_fare, in_sex, in_emb, in_cls, in_fam], out)

# ============================================================================
# TODO 3: PREDICT THE BILL, THEN PAY IT
# ============================================================================
# Concat width = 1 + 1 + 3 + 4 + 4 + 4 = 17
# Trunk: (17 + 1) * 16 = 288.  Head: 16 + 1 = 17.  Total = 305.
# (Lookup one-hot layers reserve an OOV slot, so widths are vocab + 1 -
#  the machine's count is the ground truth your paper answer must match.)

paper_count = 305

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(feed(train), y_train, epochs=30, verbose=0)
print("dev:", dict(zip(model.metrics_names, model.evaluate(feed(dev), y_dev, verbose=0))))
trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
print("trainable params:", trainable, "paper:", paper_count)
assert trainable == paper_count

# ============================================================================
# TODO 4: EMBEDDING SWAP
# ============================================================================

lk_emb = layers.StringLookup(vocabulary=["S", "C", "Q"])
emb_emb = layers.Embedding(input_dim=4, output_dim=4)

paths[3] = layers.Flatten()(emb_emb(lk_emb(in_emb)))
trunk = layers.Dense(16, activation="relu")(layers.Concatenate()(paths))
out = layers.Dense(1, activation="sigmoid", name="survived")(trunk)
model2 = keras.Model([in_age, in_fare, in_sex, in_emb, in_cls, in_fam], out)
model2.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model2.fit(feed(train), y_train, epochs=30, verbose=0)
print("dev (embedding):", dict(zip(model2.metrics_names, model2.evaluate(feed(dev), y_dev, verbose=0))))
trainable2 = int(sum(np.prod(w.shape) for w in model2.trainable_weights))
print("trainable params:", trainable2)

# The embedding adds 4 x 4 = 16 parameters (vocab + OOV, dim 4) and keeps
# concat width at 17 (4-wide either way), so the trunk is unchanged:
# 305 + 16 = 321. With a 3-category column the dev accuracy barely moves -
# the honest conclusion is that one-hot was already enough here, and
# embeddings earn their parameters only when the vocabulary grows.

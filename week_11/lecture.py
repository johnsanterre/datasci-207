"""
DATASCI 207: Applied Machine Learning
Module 11: Network Architecture Design

This module covers:
- Baselines before networks
- Sequential vs Functional API
- Bucketing numeric features (Discretization)
- Multiple named inputs; one-hot vs embeddings for categoricals
- Multi-output models with per-head losses

Using TensorFlow/Keras on the Titanic dataset (via seaborn).
"""

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

np.random.seed(42)
tf.random.set_seed(42)

df = sns.load_dataset("titanic")[["survived", "age", "fare", "sex", "pclass"]].copy()
print(df.shape)
print(df.isna().sum())
df.head()

# =============================================================================
# SETUP
# =============================================================================

df["age"] = df["age"].fillna(df["age"].median())

idx = np.random.permutation(len(df))
cut = int(0.8 * len(df))
train, dev = df.iloc[idx[:cut]].copy(), df.iloc[idx[cut:]].copy()

y_train = train["survived"].values.astype("float32")
y_dev = dev["survived"].values.astype("float32")
print(len(train), "train /", len(dev), "dev")
print("survival rate (train): %.3f" % y_train.mean())

# =============================================================================
# PART 1: BASELINES
# =============================================================================

maj = np.zeros_like(y_dev)
rule = (dev["sex"] == "female").values.astype("float32")

print("majority-class accuracy: %.3f" % (maj == y_dev).mean())
print("one-rule (sex) accuracy: %.3f" % (rule == y_dev).mean())

# =============================================================================
# PART 2: SEQUENTIAL VS FUNCTIONAL
# =============================================================================

x_age_train = train[["age"]].values.astype("float32")
x_age_dev = dev[["age"]].values.astype("float32")

norm_age = layers.Normalization()
norm_age.adapt(x_age_train)

seq = keras.Sequential([
    keras.Input(shape=(1,), name="age"),
    norm_age,
    layers.Dense(1, activation="sigmoid"),
])

inp = keras.Input(shape=(1,), name="age")
out = layers.Dense(1, activation="sigmoid")(norm_age(inp))
fun = keras.Model(inputs=inp, outputs=out)

for m in (seq, fun):
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    m.fit(x_age_train, y_train, epochs=20, verbose=0)

print("sequential:", dict(zip(seq.metrics_names, seq.evaluate(x_age_dev, y_dev, verbose=0))))
print("functional:", dict(zip(fun.metrics_names, fun.evaluate(x_age_dev, y_dev, verbose=0))))
fun.summary()

# =============================================================================
# PART 3: BUCKETING A NUMBER (DISCRETIZATION)
# =============================================================================

buckets = layers.Discretization(bin_boundaries=[12.0, 25.0, 40.0, 60.0],
                                output_mode="one_hot")

inp = keras.Input(shape=(1,), name="age")
out = layers.Dense(1, activation="sigmoid")(buckets(inp))
bucketed = keras.Model(inp, out)
bucketed.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
bucketed.fit(x_age_train, y_train, epochs=20, verbose=0)
print("bucketed age:", dict(zip(bucketed.metrics_names, bucketed.evaluate(x_age_dev, y_dev, verbose=0))))
bucketed.summary()

# =============================================================================
# PART 4: MULTIPLE NAMED INPUTS
# =============================================================================

def feed(frame):
    return {
        "age": frame[["age"]].values.astype("float32"),
        "fare": frame[["fare"]].values.astype("float32"),
        "sex": frame["sex"].values,
        "pclass": frame["pclass"].values,
    }

norm_age = layers.Normalization(); norm_age.adapt(feed(train)["age"])
norm_fare = layers.Normalization(); norm_fare.adapt(feed(train)["fare"])
lookup_sex = layers.StringLookup(vocabulary=["male", "female"], output_mode="one_hot")
lookup_cls = layers.IntegerLookup(vocabulary=[1, 2, 3], output_mode="one_hot")

in_age = keras.Input(shape=(1,), name="age")
in_fare = keras.Input(shape=(1,), name="fare")
in_sex = keras.Input(shape=(1,), dtype="string", name="sex")
in_cls = keras.Input(shape=(1,), dtype="int64", name="pclass")

paths = [
    norm_age(in_age),
    norm_fare(in_fare),
    layers.Flatten()(lookup_sex(in_sex)),
    layers.Flatten()(lookup_cls(in_cls)),
]
trunk = layers.Dense(16, activation="relu")(layers.Concatenate()(paths))
out = layers.Dense(1, activation="sigmoid", name="survived")(trunk)

multi = keras.Model([in_age, in_fare, in_sex, in_cls], out)
multi.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
def tparams(m):
    return int(sum(np.prod(w.shape) for w in m.trainable_weights))

multi.fit(feed(train), y_train, epochs=30, verbose=0)
print("multi-input:", dict(zip(multi.metrics_names, multi.evaluate(feed(dev), y_dev, verbose=0))))
print("trainable parameters:", tparams(multi))
# The graph you wired, drawn. (Falls back to summary() if pydot is absent.)
try:
    keras.utils.plot_model(multi, show_shapes=True, dpi=64)
except Exception as e:
    print("plot_model unavailable here (%s) - summary instead:" % type(e).__name__)
    multi.summary()

# =============================================================================
# PART 5: EMBEDDINGS FOR CATEGORICAL COLUMNS
# =============================================================================

lk_sex = layers.StringLookup(vocabulary=["male", "female"])
lk_cls = layers.IntegerLookup(vocabulary=[1, 2, 3])

in_age = keras.Input(shape=(1,), name="age")
in_fare = keras.Input(shape=(1,), name="fare")
in_sex = keras.Input(shape=(1,), dtype="string", name="sex")
in_cls = keras.Input(shape=(1,), dtype="int64", name="pclass")

emb_sex = layers.Embedding(input_dim=3, output_dim=4)   # vocab + OOV token
emb_cls = layers.Embedding(input_dim=4, output_dim=4)

paths = [
    norm_age(in_age),
    norm_fare(in_fare),
    layers.Flatten()(emb_sex(lk_sex(in_sex))),
    layers.Flatten()(emb_cls(lk_cls(in_cls))),
]
trunk = layers.Dense(16, activation="relu")(layers.Concatenate()(paths))
out = layers.Dense(1, activation="sigmoid", name="survived")(trunk)

embedded = keras.Model([in_age, in_fare, in_sex, in_cls], out)
embedded.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
embedded.fit(feed(train), y_train, epochs=30, verbose=0)
print("embedded:", dict(zip(embedded.metrics_names, embedded.evaluate(feed(dev), y_dev, verbose=0))))
print("trainable parameters:", tparams(embedded))

# What did pclass learn? Three 4-d vectors (plus the OOV row).
w = emb_cls.get_weights()[0]
for i, cls_name in enumerate(["OOV", "1st", "2nd", "3rd"]):
    print(cls_name, np.round(w[i], 3))
print("distance 1st-2nd: %.3f" % np.linalg.norm(w[1] - w[2]))
print("distance 1st-3rd: %.3f" % np.linalg.norm(w[1] - w[3]))

# =============================================================================
# PART 6: MULTIPLE OUTPUTS
# =============================================================================

q1, q2 = train["fare"].quantile([1/3, 2/3]).values
def fare_bracket(frame):
    f = frame["fare"].values
    return np.digitize(f, [q1, q2]).astype("int32")   # 0, 1, 2

yb_train, yb_dev = fare_bracket(train), fare_bracket(dev)

in_age = keras.Input(shape=(1,), name="age")
in_sex = keras.Input(shape=(1,), dtype="string", name="sex")
in_cls = keras.Input(shape=(1,), dtype="int64", name="pclass")

paths = [
    norm_age(in_age),
    layers.Flatten()(lookup_sex(in_sex)),
    layers.Flatten()(lookup_cls(in_cls)),
]
trunk = layers.Dense(16, activation="relu")(layers.Concatenate()(paths))
out_surv = layers.Dense(1, activation="sigmoid", name="survived")(trunk)
out_fare = layers.Dense(3, activation="softmax", name="fare_bracket")(trunk)

two_headed = keras.Model([in_age, in_sex, in_cls], [out_surv, out_fare])
two_headed.compile(
    optimizer="adam",
    loss={"survived": "binary_crossentropy",
          "fare_bracket": "sparse_categorical_crossentropy"},
    loss_weights={"survived": 1.0, "fare_bracket": 0.5},
    metrics={"survived": ["accuracy"], "fare_bracket": ["accuracy"]},
)
feed2_train = {k: v for k, v in feed(train).items() if k != "fare"}
feed2_dev = {k: v for k, v in feed(dev).items() if k != "fare"}
two_headed.fit(feed2_train, {"survived": y_train, "fare_bracket": yb_train},
               epochs=30, verbose=0)
res = two_headed.evaluate(feed2_dev, {"survived": y_dev, "fare_bracket": yb_dev}, verbose=0)
print(dict(zip(two_headed.metrics_names, np.round(res, 3))))
print("trainable parameters:", tparams(two_headed))

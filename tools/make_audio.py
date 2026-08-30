#!/usr/bin/env python3
"""Orientation audio per module: Kokoro narration -> audio/week_XX.m4a.

Distinct from the video track: this is the two-minute spoken orientation a
student hears before starting the module — what it is, why it matters, how to
walk the stages, and the one standing pitfall.

    /Users/john/Dropbox/_/tts/venv/bin/python make_audio.py            # all 13
    /Users/john/Dropbox/_/tts/venv/bin/python make_audio.py week_04    # one
"""
import os, subprocess, sys
import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = "/tmp/ds207-audio"
os.makedirs(WORK, exist_ok=True)
os.makedirs(f"{ROOT}/audio", exist_ok=True)

WALK = ("Here is how to walk the module. Watch the overview first — it is the map. "
        "Read the through-line, then open the lecture notebook in Colab and run every "
        "cell; change things and rerun, because the notebook is the lecture. Attempt "
        "the exercise before looking at the solution, take the knowledge check, and "
        "use the glossary and readings as reference. ")

SCRIPTS = {
"week_01": "Welcome to module one, and to the course. This module installs the mental "
 "model everything else builds on: a model is a function — fitted from recorded "
 "examples rather than written by hand. You will meet the four words the course "
 "returns to every week: model, parameters, loss, and generalization. The heart of "
 "the module is the train-test split. Evaluating a model on the data it was fit to "
 "rewards memorization, and memorization fails on the first unseen example — so we "
 "hold data back, and judge only on what the model never studied. In the notebook "
 "you will overfit a model on purpose, so you can recognize the failure on sight "
 "for the rest of the course. " + WALK +
 "The pitfall this week: rushing past the vocabulary. Every later module assumes "
 "these four words are reflexes, not memories. Give this one real attention — it is "
 "the cheapest module to learn and the most expensive to skip.",

"week_02": "Module two answers the question module one left open: how are the "
 "parameters actually chosen? The move is to define wrongness as a number — the "
 "loss, mean squared error for regression — and then learning stops being mystical. "
 "It is minimization. Gradient descent is the minimizer: the gradient points uphill, "
 "you step downhill, scaled by a learning rate, and repeat. You will derive the "
 "update rule by hand this week — it is two lines of calculus, and seeing that "
 "there is no magic in it is the point. The dials matter: a learning rate too high "
 "diverges, too low crawls; batch descent is stable and slow, stochastic is noisy "
 "and fast, and mini-batches are what everyone ships. " + WALK +
 "The pitfall: treating gradient descent as this module's topic only. It is the "
 "training loop for everything ahead — logistic regression, neural networks, "
 "transformers. Learn it here, own it everywhere.",

"week_03": "Module three is about the handoff between raw data and the numbers a "
 "model actually sees — feature engineering. It opens with regression in matrix "
 "form: rows are examples, columns are features, and reading shapes fluently is a "
 "skill you will use every week from now on. Then the craft: scaling features so "
 "gradient descent stops zigzagging, handling missing values while knowing which "
 "assumption each choice encodes, and one-hot encoding categories so the model "
 "does not invent an ordering between red, green, and blue. Feature crosses and "
 "bucketing close the module — two ways to add expressive power while staying "
 "linear. " + WALK +
 "The pitfall: reaching for a fancier algorithm when a model disappoints. Most of "
 "the time the fix is in the features, and the habit of looking there first will "
 "decide more of your real projects than any model choice.",

"week_04": "Module four crosses from predicting numbers to predicting categories. "
 "The sigmoid squashes a linear score into a probability; the decision boundary "
 "underneath stays a straight line; and the threshold that turns probability into "
 "decision is a dial you own — medical screening and spam filtering want different "
 "settings of the same model. Training needs cross-entropy loss, because squared "
 "error pushed through a sigmoid goes non-convex and its gradients die exactly "
 "when the model is most confidently wrong. Same descent loop as last week, new "
 "loss, new reading of the output. " + WALK +
 "The pitfall: forgetting the threshold is an application decision, not a "
 "mathematical one. When you move it in the notebook, watch both error rates move "
 "in opposite directions — that trade-off is the subject of module five, and it "
 "never goes away.",

"week_05": "Module five extends classification past two classes and then teaches "
 "you to measure honestly. Softmax turns k scores into a probability distribution; "
 "categorical cross-entropy asks how much probability landed on the true class. "
 "The measurement half matters more. Accuracy is one number and on imbalanced data "
 "it lies — a model that answers healthy every time scores ninety-nine percent "
 "while missing every case that matters. The confusion matrix is the honest "
 "ledger, and from it come precision and recall, which trade off through the "
 "threshold you met last week. F1 collapses them into one number, at a cost. "
 + WALK +
 "The pitfall — and the discipline this module installs: never report a single "
 "metric for a classifier that matters. Print the confusion matrix first. It shows "
 "which way the model is wrong, and the two directions almost never cost the same.",

"week_06": "Module six is where models learn to bend. It opens with XOR — four "
 "points no straight line can separate — and that failure forces the whole design: "
 "hidden layers computing intermediate features, nonlinear activations keeping the "
 "stack from collapsing back into one line, and backpropagation walking the chain "
 "rule backwards to hand every weight its gradient. You will build a network in "
 "raw NumPy — forward pass, backward pass, training loop — and teach it XOR, then "
 "rebuild the same network in Keras where it becomes three lines. Scratch first, "
 "framework second, because frameworks buy convenience, not understanding. " + WALK +
 "The pitfall: letting the framework arrive before the understanding. If you can "
 "explain why removing the activation function makes the network exactly as weak "
 "as a straight line, this module has done its job.",

"week_07": "Module seven breaks the pattern: two model families that never compute "
 "a gradient. K-nearest neighbors stores the data and answers with the vote of the "
 "closest examples — barely training at all, and surprisingly strong. Decision "
 "trees ask learned questions, chosen by information gain, and give you a model "
 "you can read aloud — plus a model that memorizes noise if grown deep. The "
 "ensembles fix that: random forests average many noisy trees into something "
 "stable, and gradient boosting builds small trees in sequence, each correcting "
 "the errors of the sum so far. " + WALK +
 "The pitfall is scope: on tables of features, tuned boosting is usually the "
 "strongest baseline you can field — reach for it before a neural network on "
 "tabular data. Networks earn their keep on images, text, and audio, which is "
 "where the course goes next.",

"week_08": "Module eight takes the labels away. Two questions survive: can we "
 "group the examples, and can we compress the dimensions? K-means answers the "
 "first — assign points to the nearest centroid, move centroids to the mean, "
 "repeat — and you will implement the loop from scratch and watch it converge. "
 "The elbow method and silhouette score give evidence for your choice of k, but "
 "only evidence. PCA answers the second: keep the directions of greatest "
 "variance, and hundreds of correlated columns become a few axes. Hierarchical "
 "clustering closes the module by deferring the choice of k entirely. " + WALK +
 "The pitfall is certainty. With no labels there is no ground truth — when the "
 "algorithm reports four segments, that is a defensible reading, not a fact. "
 "Treat every unsupervised result as a hypothesis to test, not a discovery to "
 "announce.",

"week_09": "Module nine turns text into numbers. The counting era first: "
 "bag-of-words gives every vocabulary word a column; TF-IDF weights down the "
 "words that appear everywhere; n-grams recover short phrases. All of it works, "
 "and all of it is blind — excellent and outstanding share no dimensions at all. "
 "Embeddings fix that: short dense vectors learned from context, where similarity "
 "becomes geometry and cosine similarity measures what counting cannot see. You "
 "will build every piece from scratch, through document embeddings. " + WALK +
 "Keep the destination in view as you work: this token-and-vector pipeline, "
 "scaled a million-fold, is the front end of every transformer — module thirteen "
 "finishes this road. The pitfall: skipping the from-scratch builds because "
 "libraries exist. The libraries are one import; the understanding of what they "
 "compute is this module.",

"week_10": "Module ten is vision. Flattening an image throws away the fact that "
 "pixel neighbors are neighbors, and dense layers drown in parameters. The "
 "convolution fixes both: a small filter scans every position, so a pattern "
 "learned once is found anywhere, with dozens of weights instead of millions. "
 "Pooling buys tolerance to small shifts, and stacked layers specialize — edges, "
 "textures, parts, objects. Do the output-size arithmetic by hand once; every "
 "shape bug you will ever hit in Keras traces back to it. The two working tricks "
 "close the module: transfer learning from pretrained networks, and data "
 "augmentation. Between them, serious vision work gets done on small datasets. "
 + WALK +
 "The pitfall: treating transfer learning as cheating. It is the professional "
 "default — millions of images already taught the network edges and textures, "
 "and your data teaches it your task.",

"week_11": "Module eleven is the defense module. Regularization writes a penalty "
 "for complexity into the loss itself: L2 shrinks weights smoothly, L1 drives "
 "some exactly to zero and selects features as a side effect; dropout and early "
 "stopping do the same job for networks. Then honest tuning: the test set stays "
 "sealed until the end, cross-validation rotates a validation fold through the "
 "training data, random search usually beats grids, and pipelines fit your "
 "preprocessing inside each fold so the silent leak that inflates scores becomes "
 "structurally impossible. " + WALK +
 "The pitfall is quiet and common: any decision made by peeking at test "
 "performance poisons the estimate. This module is the difference between a model "
 "that demos well and one you can defend — and it is the material technical "
 "interviews draw on most.",

"week_12": "Module twelve is fairness, and it starts with a deliberately "
 "uncomfortable demonstration: a hiring classifier trained on biased history "
 "reproduces the bias with clean code and a good accuracy score. Nothing "
 "malfunctioned. You will measure fairness with tools you already own — "
 "confusion matrices computed per group. Demographic parity asks whether groups "
 "are selected at equal rates; equalized odds asks whether qualified candidates "
 "face the same error rates. Then the impossibility theorem: when base rates "
 "differ, reasonable fairness criteria conflict — as arithmetic, not as an "
 "engineering shortfall. Choosing a definition is a values decision, made "
 "explicitly or made by default. " + WALK +
 "The posture to carry out of this module: audit by group, never only in "
 "aggregate; name the fairness definition you chose and why; and treat accuracy "
 "as the beginning of an evaluation, never the end.",

"week_13": "Module thirteen ends the course at the architecture running modern "
 "AI. Attention is weighted looking: each position scores its query against every "
 "key, softmaxes the scores, and takes the weighted sum of values — so a word "
 "forty tokens back is one step away, not forty. Causal masking keeps next-token "
 "training honest, multi-head attention runs several patterns in parallel, and "
 "positional encodings inject the order that attention alone ignores. Assembled "
 "with residuals and layer norm, that is the encoder block — and stacked blocks "
 "are, to a first approximation, the whole architecture. " + WALK +
 "Leave with the course's closing note: the frontier of AI is matrix multiplies, "
 "softmax, and gradient descent. After thirteen modules, you have implemented "
 "every one of those pieces from scratch. Congratulations — now go build "
 "something with it.",
}

def main():
    from kokoro import KPipeline
    pipe = KPipeline(lang_code="a")
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for slug, text in SCRIPTS.items():
        if only and slug != only:
            continue
        chunks = [a for _, _, a in pipe(text, voice="af_heart")]
        audio = np.concatenate(chunks)
        wav = f"{WORK}/{slug}.wav"
        sf.write(wav, audio, 24000)
        out = f"{ROOT}/audio/{slug}.m4a"
        subprocess.run(["ffmpeg", "-y", "-i", wav, "-c:a", "aac", "-b:a", "80k", out],
                       capture_output=True, check=True)
        print(f"{out}  ({len(audio)/24000/60:.1f} min)", flush=True)
    print("AUDIO DONE", flush=True)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Rebuild each week_XX/index.html as a scroll-through module sequence.

Extracts the existing intro, learning objectives, key concepts, prerequisites
and footer from the current page, then re-emits the page as staged segments:
watch -> listen -> read -> code -> practice -> check -> go deeper -> done,
with localStorage progress and a completion event.

The per-week "through-line" read segments live in THROUGH below — they are the
teaching prose and the reason this script is kept in the repo. Idempotent:
regenerating from an already-built page works because extraction reads the
data-* blocks it emits.
"""
import os, re, sys, html

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

THROUGH = {
1: """
<p>Machine learning starts with a claim modest enough to miss: a prediction is just a
function. Something takes inputs — square footage, a pixel grid, last month's sales —
and returns an output. Programming writes that function by hand. Machine learning
fits it from recorded input–output pairs, which is why the field obsesses over data
before it obsesses over algorithms.</p>
<p>The lecture builds this up in six moves. Functions first, as pure Python. Then data
as (x, y) pairs, then a <em>model</em>: a function with free parameters, whose values are
chosen to fit the pairs rather than typed in by a person. Fitting immediately raises
the question the rest of the course keeps answering — how good is the fit? — so
evaluation enters in the same breath: a loss number, not a feeling.</p>
<p>The last two parts contain the week's real lesson. If you evaluate a model on the
data it was fit to, you reward memorization, and memorization fails on the first
example it hasn't seen. The train/test split is the field's defense: hold data back,
fit on one part, judge on the other. The gap between training error and test error
has a name — overfitting — and you will watch it happen on purpose in the notebook.</p>
<p>Hold on to the vocabulary this week installs: model, parameters, loss,
generalization. Every module after this one is a variation on those four words.</p>
""",
2: """
<p>Module 1 said a model is a function with free parameters. This module answers the
obvious next question: chosen <em>how</em>? For linear regression the model is a weighted
sum — predict y as wx + b — and "learning" means finding the w and b that make the
predictions least wrong on the training data.</p>
<p>"Least wrong" needs a definition, and that is the loss function. Mean squared
error averages the squared gaps between predictions and truth. Squaring does two
jobs: errors in both directions count, and large misses count disproportionately.
With a loss defined, learning stops being mystical — it is minimization, full stop.</p>
<p>Gradient descent is the minimizer. The gradient of the loss with respect to each
parameter says which direction increases the loss; step the opposite way, scaled by
a learning rate, and repeat. The lecture derives the update rule by hand so you see
there is no magic in it — two lines of calculus, applied over and over.</p>
<p>The practical content is in the failure modes. A learning rate too large
overshoots and diverges; too small crawls. Batch descent uses every example per
step and is stable but slow; stochastic descent uses one example, is noisy but
fast; mini-batches split the difference and are what everyone actually uses. You
will watch all three walk the same loss surface in the notebook.</p>
<p>What generalizes beyond regression: nearly everything ahead — logistic
regression, neural networks, transformers — trains by exactly this loop. Different
model, different loss, same descent.</p>
""",
3: """
<p>Models see only the numbers you hand them. This module is about the gap between
raw data and those numbers — the unglamorous work that decides more real-world
projects than the choice of algorithm does.</p>
<p>It opens by generalizing regression to many inputs, in matrix form: X becomes a
table, w a vector, prediction a product. That notation is not just compact — it is
how every library expects your data, so learn to read shapes: rows are examples,
columns are features.</p>
<p>Then the craft. Features on wildly different scales (income in dollars, age in
years) tilt the loss surface so gradient descent zigzags; standardization levels
it. Missing values must become numbers somehow — dropped, imputed, or flagged —
and each choice encodes an assumption you should be able to defend. Categorical
values become one-hot vectors, because handing "red=1, green=2, blue=3" to a linear
model invents an ordering that does not exist.</p>
<p>The last two parts add expressive power without leaving the linear world. A
feature cross multiplies two features so the model can respond to their
combination — location <em>and</em> size, not location plus size. Bucketing turns a
continuous value into ranges, letting a straight-line model fit a curve in
steps.</p>
<p>The habit to build this week: when a model underperforms, look at the features
before reaching for a fancier algorithm. Most of the fix is usually here.</p>
""",
4: """
<p>Regression predicts a number. Classification predicts a category, and the first
honest question is why not just run regression on labels 0 and 1. The lecture
starts there, shows where it breaks — unbounded outputs, no probabilistic reading —
and lets that failure motivate the fix.</p>
<p>The fix is the sigmoid: squash the linear score into (0, 1) and read the result
as a probability. Logistic regression is exactly linear regression pushed through
that squash. The geometry stays linear underneath — the set of points where the
probability equals one half is a straight line (a hyperplane, in general), and
that is the decision boundary.</p>
<p>A probability is not yet a decision. The threshold converts one into the other,
and it is a dial, not a law of nature: lower it and you catch more positives at
the cost of more false alarms. Where to set it depends on which error costs more —
a question about the application, not the mathematics. Medical screening and spam
filtering want different thresholds for the same model.</p>
<p>Training needs a loss that fits probabilities, and squared error is the wrong
one — it goes non-convex through the sigmoid and its gradients vanish exactly when
the model is most confidently wrong. Cross-entropy loss punishes confident
wrongness hard, keeps the surface clean for gradient descent, and trains with the
same update loop you derived last week.</p>
<p>Sigmoid, boundary, threshold, cross-entropy: four pieces, one pipeline, and the
template for every classifier in the rest of the course.</p>
""",
5: """
<p>Two classes was a warm-up; the world mostly is not binary. Softmax generalizes
the sigmoid to k classes: exponentiate k scores, normalize, and read off a
probability distribution. One clean idea and the same training loop — with
categorical cross-entropy asking one thing only: how much probability did you put
on the true class?</p>
<p>The heavier half of the module is measurement. Accuracy is one number, and on
imbalanced data it is a liar — a model that answers "healthy" every time scores
99% on a dataset that is 99% healthy while missing every case that matters.</p>
<p>The confusion matrix is the honest ledger: what was predicted against what was
true, per class. From it come precision (of everything flagged, how much was
real?) and recall (of everything real, how much was flagged?). The two trade off
against each other through the threshold you met last week; F1 collapses the pair
into one number when ranking models, at the cost of hiding which side is weak.</p>
<p>The discipline this module installs: never report one metric for a classifier
that matters. Print the confusion matrix first — it tells you not just how often
the model is wrong, but <em>which way</em> it is wrong, and the two errors almost never
cost the same.</p>
""",
6: """
<p>Everything so far draws straight lines through feature space. XOR — output 1
when exactly one input is 1 — is the classic four-point dataset no straight line
can split, and it opens this module because it forces the conclusion: to go
further, models need to bend.</p>
<p>The bend comes from stacking. A hidden layer computes several linear functions
of the input, passes each through a nonlinearity, and hands the results to the
next layer as learned features. Without the nonlinearity the stack collapses back
into one linear map — this is why activation functions exist, not decoration.
ReLU's blunt max(0, x) has largely won over the smoother sigmoid and tanh because
its gradient doesn't die as networks deepen.</p>
<p>Forward propagation is just those layers evaluated in order, and the lecture
builds it from scratch in NumPy so the "network" demystifies into a handful of
matrix multiplies. Backpropagation answers the training question: the chain rule,
applied layer by layer from the loss backwards, gives every weight its gradient —
and then the update is the same descent step you have run since module 2.</p>
<p>The module closes by rebuilding the same network in Keras, where the from-scratch
version becomes three lines. Frameworks buy convenience, not understanding —
which is why the scratch version comes first. Watch a network learn XOR, the
problem a line could not touch, and module 1's vocabulary — model, parameters,
loss, generalization — now covers deep learning too.</p>
""",
7: """
<p>This module breaks the pattern on purpose. Not every model is a parametric
function trained by descent — two other families matter, and one of them wins most
tabular-data competitions to this day.</p>
<p>K-nearest neighbors barely counts as training: store the data, and at
prediction time answer with the majority vote of the k closest stored examples.
It is simple, surprisingly strong, and a standing lesson in how far "similar
inputs get similar outputs" goes — plus a warning about what "close" even means
in many dimensions.</p>
<p>Decision trees ask learned questions — is income over 50k? is age under 30? —
and route each example down branches to an answer. The learning is in choosing
questions: entropy measures the impurity of a group, and the tree greedily picks
the split with the largest information gain. The result is a model you can read
aloud, and one that will happily memorize noise if grown deep enough.</p>
<p>Ensembles fix the tree's variance with statistics. A random forest trains many
trees on bootstrapped samples with random feature subsets and averages them —
individually noisy, collectively stable. Gradient boosting goes the other way:
build small trees in sequence, each fitting the errors of the sum so far. Forests
tame variance; boosting attacks bias; both routinely beat a single carefully
tuned tree.</p>
<p>The practical takeaway is scope: on tables of features, tuned boosting is often
the strongest baseline you can field, and it should be the first serious model
you reach for there — neural networks earn their keep on images, text, and audio,
which the coming modules turn to.</p>
""",
8: """
<p>Every module so far had labels. This one takes them away and asks what structure
the inputs carry on their own. Two answers: group the examples (clustering), or
compress the dimensions (dimensionality reduction).</p>
<p>K-means is clustering at its most direct: place k centroids, assign each point
to its nearest centroid, move each centroid to the mean of its points, repeat
until nothing moves. You will implement the loop from scratch and watch it
converge in a handful of iterations. Its catch is honest and unavoidable — you
chose k. The elbow method (where does adding a cluster stop paying?) and the
silhouette score (how much closer is each point to its own cluster than the next?)
are evidence for that choice, not proofs.</p>
<p>PCA compresses instead of grouping: find the orthogonal directions along which
the data varies most and keep the top few. Built from scratch, it is an
eigendecomposition of the covariance matrix; used well, it turns hundreds of
correlated columns into a handful of axes that preserve most of the variance —
for visualization, denoising, or as features for the supervised models you
already know.</p>
<p>Hierarchical clustering closes the module by removing the need to pick k up
front: merge the closest pair of clusters repeatedly and read the dendrogram at
whatever depth serves the question.</p>
<p>The caveat that governs all of it: with no labels there is no ground truth, so
"the algorithm found four segments" means four was a defensible reading of the
evidence — not a fact about the world. Unsupervised results are hypotheses.</p>
""",
9: """
<p>Models eat numbers, and text is not numbers. Every technique in this module is
an answer to one question: how do you turn language into vectors without losing
what it means?</p>
<p>Bag-of-words is the bluntest answer — one column per vocabulary word, count the
occurrences, ignore the order. TF-IDF sharpens it by weighting words up when they
are frequent in a document and down when they are frequent everywhere, so "the"
stops mattering and "gradient" starts to. N-grams recover short-range order by
counting word pairs and triples as their own features. All three are built from
scratch in the notebook, and all three produce long, sparse, brittle vectors:
"excellent" and "outstanding" share no dimensions at all.</p>
<p>Embeddings are the modern answer. Give each word a short dense vector, learned
from the company it keeps, and similarity becomes geometry: synonyms land near
each other, and cosine similarity — the angle between vectors — measures
relatedness the counting methods cannot see. Averaging word vectors gives a
serviceable embedding for a whole document, enough for search and classification
demos in the notebook.</p>
<p>This is the module where language modeling from the AI-literacy world and
classical ML meet: the token-and-vector pipeline you build here by hand is,
scaled a million-fold, the front end of every transformer — which is exactly
where module 13 ends the course.</p>
""",
10: """
<p>Flatten a 200×200 image into a vector and a dense network needs forty thousand
weights per neuron in the first layer — and it forgets that pixel neighbors are
neighbors. Convolutions fix both at once, and they are the reason computer vision
works.</p>
<p>A convolution slides a small filter — say 3×3 — across the image, computing a
dot product at each position. The same nine weights scan every location, so a
pattern learned once is found anywhere (translation invariance), and parameters
drop from millions to dozens. You will implement the sliding window from scratch
and watch handmade filters light up on edges before any learning happens.</p>
<p>Pooling downsamples the resulting feature maps — max pooling keeps the
strongest response in each window — buying tolerance to small shifts and cutting
computation. Stacking convolution, nonlinearity, and pooling gives the CNN: early
layers detect edges, middle layers combine them into textures and parts, late
layers into objects. The output-size arithmetic (filter size, stride, padding)
is unglamorous and worth doing by hand once, because every shape bug in Keras
traces back to it.</p>
<p>The two closing ideas are the ones practitioners lean on daily. Transfer
learning: a network trained on millions of images has already learned edges and
textures, so start from it and retrain the top for your task with a fraction of
the data. Augmentation: flip, crop, and shift your training images so the model
sees more variety than you collected. Between them, serious vision work gets done
on small datasets.</p>
""",
11: """
<p>Module 1 named the enemy — overfitting — and every module since has walked past
it. This one arms you. The theme throughout: constrain the model on purpose, and
choose its settings by evidence instead of taste.</p>
<p>Regularization adds a penalty for large weights directly into the loss, so the
model pays for complexity. L2 shrinks weights smoothly toward zero; L1 drives
some weights exactly to zero, which performs feature selection as a side effect —
the sparsity difference between the two is derived and then demonstrated. Dropout
and early stopping apply the same philosophy to networks: randomly silence units
during training, or quit while validation error is still falling.</p>
<p>Choosing the penalty strength — or any hyperparameter — with the test set is
self-deception; that data must stay untouched until the end. Cross-validation is
the honest instrument: split the training data k ways, rotate which fold
validates, average. Grid search sweeps combinations exhaustively; random search
usually finds as good a configuration faster, because most hyperparameters barely
matter and random sampling explores the ones that do.</p>
<p>Pipelines close the module as the guardrail that makes all of it trustworthy:
chain preprocessing and model into one object so scaling is fit inside each
training fold — never on data the fold will validate on. That subtle leak
inflates scores silently, and pipelines make it structurally impossible.</p>
<p>This module is the difference between a model that demos well and one you can
defend. It is also, not coincidentally, the module interviews draw on most.</p>
""",
12: """
<p>A model trained on history learns history's patterns — including the ones a
deployment should not repeat. This module makes that concrete with a simulated
hiring classifier: trained on biased outcomes, it reproduces the bias with clean
code and good accuracy. Nothing malfunctioned. That is the point.</p>
<p>Fairness has competing definitions, and the module builds the two central ones.
Demographic parity asks whether groups are selected at equal rates. Equalized
odds asks whether error rates — true and false positive rates — match across
groups, so qualified candidates face the same odds regardless of group. Both are
computed, per group, from the confusion matrices you learned in module 5; the
fairness toolkit is the measurement toolkit, pointed at people.</p>
<p>Then the uncomfortable mathematics: the impossibility theorem. When base rates
differ between groups, several reasonable fairness criteria cannot all hold at
once — not as an engineering shortfall but as arithmetic. Choosing which fairness
definition to satisfy is therefore a decision about values, made explicitly or
made by default. The module also demonstrates post-processing mitigation:
adjusting thresholds per group to equalize the chosen metric, with visible costs
elsewhere.</p>
<p>The professional posture this module asks of you: audit by group, not in
aggregate; say which fairness definition you chose and why; and treat "the model
is accurate" as the beginning of the evaluation, not the end. This is also where
this course's practicum instinct lives — the systems you build will touch people
who never saw the training data.</p>
""",
13: """
<p>Module 9 left text as averaged word vectors — order ignored, context lost. The
transformer is the architecture that fixed this, and it runs essentially every
system currently called AI. This module builds its core mechanism small enough
to read.</p>
<p>Attention is weighted looking. For each position, compute a query; against
every position, compute keys and values; score query against keys, softmax the
scores into weights, and take the weighted sum of values. Each word assembles its
representation by attending to the words that matter for it — "it" can look at
whatever "it" refers to, forty tokens back, in one step. The scaling by the
square root of the key dimension is the small detail that keeps softmax gradients
alive, and you will see why in code.</p>
<p>Three refinements complete the block. Causal masking zeroes attention to future
positions, which is what makes next-token generation honest at training time —
and connects directly to the prediction machine you may have met elsewhere:
this is how those models are trained. Multi-head attention runs several attention
patterns in parallel — one head tracking syntax, another coreference. Positional
encodings inject order, because attention alone treats a sentence as a bag.</p>
<p>Assembled — attention, residuals, layer norm, feedforward — these form the
encoder block, and stacking such blocks is, to a first approximation, the whole
architecture. The course ends on a deliberate note: the exotic frontier of AI is
matrix multiplies, softmax, and gradient descent — every piece of which you have
now implemented from scratch.</p>
""",
}

# per-week one-line orientation used under the audio player
LISTEN_LINE = {
1: "What a model actually is, why held-out data is non-negotiable, and how to walk this module.",
2: "Loss, gradients, and the training loop the whole course reuses — a spoken orientation before you start.",
3: "Why features decide projects, and the order to take the six parts in.",
4: "From regression to classification: what changes, what stays, and where the threshold decision lives.",
5: "Beyond accuracy: how to read a classifier honestly, and what to focus on this week.",
6: "Why networks need to bend, and how the from-scratch build connects to Keras.",
7: "Neighbors, trees, and ensembles — where each wins, and why boosting is the tabular baseline.",
8: "Learning without labels: what clustering can and cannot tell you.",
9: "Text into vectors, counting into embeddings — the road that ends at transformers.",
10: "Convolutions, pooling, and the two tricks that make small-data vision work.",
11: "Regularization and honest tuning — the module that makes models defensible.",
12: "Fairness as measurement, the impossibility trade-off, and the posture to carry into deployment.",
13: "Attention, step by step — the mechanism under modern AI, built small enough to read.",
}

def extract(s, n):
    d = {}
    d['title'] = re.search(r'<title>(.*?)</title>', s).group(1)
    d['h1'] = re.search(r'<h1>(.*?)</h1>', s).group(1)
    m = re.search(r'<h1>.*?</h1>\s*<p>(.*?)</p>', s, re.S)
    d['intro'] = re.sub(r'\s+', ' ', m.group(1)).strip()
    m = re.search(r'<div class="learning-objectives">(.*?)</div>', s, re.S)
    d['objectives'] = m.group(1).strip() if m else None
    m = (re.search(r'<h2>Key Concepts</h2>\s*(<ul>.*?</ul>)', s, re.S) or
         re.search(r'<h2[^>]*>Key concepts</h2>\s*(<ul>.*?</ul>)', s, re.S))
    if not m:
        raise SystemExit(f'week {n}: Key Concepts block not found — refusing to rebuild')
    d['concepts'] = m.group(1).strip()
    m = (re.search(r'<h2>Central Concepts from Prerequisites</h2>\s*(<ul>.*?</ul>)', s, re.S) or
         re.search(r'<h2[^>]*>From the prerequisites</h2>\s*(<ul>.*?</ul>)', s, re.S))
    d['prereq'] = m.group(1).strip() if m else None
    m = re.search(r'<footer>(.*?)</footer>', s, re.S)
    d['footer'] = m.group(1).strip()
    return d

def build(n):
    wk = f'week_{n:02d}'
    p = os.path.join(ROOT, wk, 'index.html')
    s = open(p).read()
    d = extract(s, n)
    gh = f'https://github.com/johnsanterre/datasci-207/blob/main/{wk}'
    co = f'https://colab.research.google.com/github/johnsanterre/datasci-207/blob/main/{wk}'
    key = f'datasci207-{wk}'
    modnum = int(re.search(r'Module (\d+)', d['h1']).group(1))

    page = f'''<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="{d['title']}">
    <title>{d['title']}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="../assets/styles.css">
<script data-goatcounter="https://johnsanterre.goatcounter.com/count" async src="//gc.zgo.at/count.js"></script>
</head>

<body>
    <header class="nav-header">
        <div class="container">
            <a href="../index.html" class="site-title">DATASCI 207: Applied Machine Learning</a>
        </div>
    </header>

    <main>
        <div class="container">
            <nav class="breadcrumb">
                <a href="../index.html">Home</a>
                <span class="separator">/</span>
                <span>Module {modnum}</span>
            </nav>

            <h1>{d['h1']}</h1>

            <p>{d['intro']}</p>

            <div class="learning-objectives">
{d['objectives']}
            </div>

            <div class="progresswrap">
              <div class="progressbar"><div id="pbar"></div></div>
              <div class="plabel"><span id="pdone">0</span> of 8 stages complete</div>
            </div>

            <section class="seg" data-seg="watch">
              <div class="stype">1 &middot; Watch</div>
              <h2>The module in two minutes</h2>
              <video controls preload="metadata" src="../video/{wk}.mp4"></video>
              <p class="medianote">An animated overview of the module&rsquo;s through-line.
              Watch it before the lecture &mdash; it is the map, not the territory.</p>
              <button class="mark" data-for="watch">Mark complete</button>
            </section>

            <section class="seg" data-seg="listen">
              <div class="stype">2 &middot; Listen</div>
              <h2>Orientation audio</h2>
              <p>{LISTEN_LINE[n]}</p>
              <audio controls preload="none" src="../audio/{wk}.m4a"></audio>
              <button class="mark" data-for="listen">Mark complete</button>
            </section>

            <section class="seg" data-seg="read">
              <div class="stype">3 &middot; Read</div>
              <h2>The through-line</h2>
              <div class="through">
{THROUGH[n].strip()}
              </div>
              <h2 style="font-size:1.05rem;margin-top:1.4rem">Key concepts</h2>
              {d['concepts']}
{f'''              <h2 style="font-size:1.05rem;margin-top:1.2rem">From the prerequisites</h2>
              {d['prereq']}''' if d['prereq'] else ''}
              <button class="mark" data-for="read">Mark complete</button>
            </section>

            <section class="seg" data-seg="code">
              <div class="stype">4 &middot; Code</div>
              <h2>The lecture notebook</h2>
              <p>Six parts, from scratch before any framework. Run every cell; change
              things and rerun &mdash; the notebook is the lecture.</p>
              <nav class="module-nav">
                <a href="{co}/lecture.ipynb">Open Lecture in Colab</a>
                <a href="{gh}/lecture.py">View Code on GitHub</a>
                <a href="README.pdf">Module Summary (PDF)</a>
              </nav>
              <p class="medianote">In Colab: File &rarr; Save a copy in Drive keeps your work.</p>
              <button class="mark" data-for="code">Mark complete</button>
            </section>

            <section class="seg" data-seg="practice">
              <div class="stype">5 &middot; Practice</div>
              <h2>Exercise</h2>
              <p>Attempt the exercise before opening the solution &mdash; the attempt is
              where the learning happens.</p>
              <nav class="module-nav">
                <a href="{co}/exercises/exercise_01.ipynb">Open Exercise in Colab</a>
                <a href="exercises/index.html">Exercise page (with solution link)</a>
              </nav>
              <button class="mark" data-for="practice">Mark complete</button>
            </section>

            <section class="seg" data-seg="check">
              <div class="stype">6 &middot; Check</div>
              <h2>Knowledge check</h2>
              <p>The quiz is self-graded and open-book. If a question surprises you,
              the corresponding lecture part is the fastest fix.</p>
              <nav class="module-nav">
                <a href="quiz.html">Take the Knowledge Check</a>
              </nav>
              <button class="mark" data-for="check">Mark complete</button>
            </section>

            <section class="seg" data-seg="deeper">
              <div class="stype">7 &middot; Go deeper</div>
              <h2>Reference and discussion</h2>
              <nav class="module-nav">
                <a href="glossary.html">Glossary</a>
                <a href="readings.html">Additional Readings</a>
                <a href="discussion.html">Discussion Topics</a>
              </nav>
              <button class="mark" data-for="deeper">Mark complete</button>
            </section>

            <section class="seg" data-seg="done">
              <div class="stype">8 &middot; Done</div>
              <h2 id="doneHead">Finish the module</h2>
              <p id="doneMsg">Complete the stages above, then claim the finish here.</p>
              <button class="mark" id="finishBtn" data-for="done" disabled>Complete Module {modnum}</button>
            </section>

            <footer>
{d['footer']}
            </footer>
        </div>
    </main>

<script>
(function(){{
  const KEY='{key}';
  const SEGS=['watch','listen','read','code','practice','check','deeper','done'];
  let state={{}};
  try{{state=JSON.parse(localStorage.getItem(KEY)||'{{}}')}}catch(e){{state={{}}}}
  function save(){{try{{localStorage.setItem(KEY,JSON.stringify(state))}}catch(e){{}}}}
  function segEl(id){{return document.querySelector('.seg[data-seg="'+id+'"]')}}
  function render(){{
    let n=0;
    for(const s of SEGS){{
      const done=!!state[s]; if(done)n++;
      const el=segEl(s); el.classList.toggle('done',done);
      const b=el.querySelector('button.mark[data-for="'+s+'"]');
      if(b&&s!=='done'){{b.textContent=done?'\\u2713 Complete \\u2014 undo':'Mark complete'}}
    }}
    document.getElementById('pbar').style.width=(n/SEGS.length*100)+'%';
    document.getElementById('pdone').textContent=n;
    const others=SEGS.slice(0,-1).every(s=>state[s]);
    const fin=document.getElementById('finishBtn');
    if(state.done){{fin.disabled=false;fin.textContent='\\u2713 Module {modnum} complete \\u2014 undo';
      document.getElementById('doneHead').textContent='Module {modnum} complete';
      document.getElementById('doneMsg').textContent='Done. The next module is linked in the footer.';}}
    else{{fin.disabled=!others;fin.textContent='Complete Module {modnum}';
      document.getElementById('doneHead').textContent='Finish the module';
      document.getElementById('doneMsg').textContent=others?
        'Every stage above is done \\u2014 claim it.':'Complete the stages above, then claim the finish here.'}}
  }}
  document.querySelectorAll('button.mark').forEach(b=>{{
    b.addEventListener('click',()=>{{
      const s=b.dataset.for;
      state[s]=!state[s];save();render();
      if(s==='done'&&state.done&&!state._counted){{
        state._counted=true;save();
        if(window.goatcounter&&window.goatcounter.count){{
          window.goatcounter.count({{path:'complete/'+KEY,event:true}});}}}}
    }})
  }});
  render();
}})();
</script>
</body>

</html>
'''
    open(p, 'w').write(page)
    print(f'{wk}: rebuilt ({len(THROUGH[n].split())} words of through-line)')

if __name__ == '__main__':
    for n in range(1, 14):
        build(n)

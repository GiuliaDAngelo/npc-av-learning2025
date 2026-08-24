# Background: from learned scanpaths to sparse relational object models

*Written 2026-08-15. Context for the planned pivot of this project: instead of using RL
only to learn a good scanpath, we want exploration (rotating the object + saccading) to
build a sparse, graph-like model of the object — salient features as nodes, the actions
that connect them as edges. Like knowing a mug "has a handle, an open mouth, is roughly
cylindrical" without a pixel-perfect mesh.*

---

## 1. Why change anything? The problem with the current memory

The current working memory is an exponential moving average of superposed VSA bindings:
every saccade produces `feature ⊛ position ⊛ rotation`, and all of them get averaged
into one vector. Averaging is commutative — after 200 saccades the memory is a *bag*.
Which features sit next to each other, and what rotation carries you from one to
another, is exactly the information that gets destroyed. Empirically we saw the
consequence: memories of different object classes ended up *more* similar than memories
of the same class.

A graph keeps what the average throws away: **nodes** = distinctive features (what),
**edges** = the movement that took you between them (how). That relational skeleton is
the "sparse model of geometry" we're after — not structure-from-motion, but topology.

---

## 2. The concepts, one at a time

### 2.1 Aspect graphs — objects as views + transitions

Old idea from vision science: represent a 3D object not as a mesh but as a graph of its
*characteristic views*, with edges for the "visual events" where one view changes into
another as you move around it. This is the closest classical ancestor of what we want
to build, except our nodes are salient local features rather than whole views.

- Koenderink & van Doorn (1979), *The internal representation of solid shape with
  respect to vision*, Biological Cybernetics.
- Bajcsy, Aloimonos & Tsotsos (2018), *Revisiting active perception*, Autonomous
  Robots — a modern review of why perception should be an active, exploratory process.

### 2.2 Sensorimotor contingencies — objects as action→sensation rules

The philosophical backbone: what you *know* about an object is the set of regularities
between your actions and the resulting sensations ("if I rotate it this way, the handle
comes into view"). An object model, in this view, literally *is* the graph of
action-conditioned predictions. This justifies putting the rotation command on the
graph edges.

- O'Regan & Noë (2001), *A sensorimotor account of vision and visual consciousness*,
  Behavioral and Brain Sciences.

### 2.3 Affordances — features are context-dependent

Gibson's point that we perceive objects in terms of what we can *do* with them (a log
is a seat, an obstacle, or a lever depending on current needs). Design consequence for
us: store features richly and let context act as a **query at retrieval time**, rather
than baking one task's viewpoint into what gets stored. VSA is well-suited to this —
probing a bound structure with a context vector is its native operation. We are *not*
building this now; we're just not precluding it.

- Gibson (1979), *The Ecological Approach to Visual Perception*.

### 2.4 Cognitive maps beyond physical space — the hippocampal connection

The claim "the machinery that navigates space also navigates the feature-space of an
object" is not just a metaphor; there is direct evidence:

- **Grid cells** in entorhinal cortex fire on a hexagonal lattice over physical space —
  Hafting et al. (2005), *Microstructure of a spatial map in the entorhinal cortex*,
  Nature.
- **The same cells map visual exploration.** Monkey entorhinal grid cells are tuned to
  *gaze position* while the animal saccades over images — Killian, Jutras & Buffalo
  (2012), *A map of visual space in the primate entorhinal cortex*, Nature. Human
  analogs: Nau et al. (2018) and Julian et al. (2018), both Nature Neuroscience. This
  is literally our setting: saccades traversing a map.
- **They also map abstract spaces**: sound frequency in rats (Aronov, Nevers & Tank
  2017, Nature) and 2D "concept spaces" in humans (Constantinescu, O'Reilly & Behrens
  2016, Science).
- **Framework paper**: Behrens et al. (2018), *What is a cognitive map? Organizing
  knowledge for flexible behavior*, Neuron.

### 2.5 Toroidal topology — the geometry question Matt raised

Gardner et al. (2022, Nature: *Toroidal topology of population activity in grid
cells*) showed that the joint activity of a grid-cell module lies on a **torus** — and
that this torus is preserved across different environments and even sleep. The
analogous result for head-direction cells is a **ring** (Chaudhuri et al. 2019, Nature
Neuroscience). Lesson: the brain represents a variable on a manifold whose *topology
matches the variable* — heading is a circle, so the code is a ring; periodic 2D phase
is a torus, so the code is a torus.

Why this matters for us: object orientation lives on a compact manifold too (rotations;
for full 3D pose it's the quaternion sphere with antipodes identified). If you force a
representation of a periodic variable into a flat Gaussian cloud, somewhere the space
must "tear" — two nearly-identical poses (359° and 0°) end up far apart. **Loop closure
— recognizing you've rotated back to a feature you saw before — fails exactly at that
tear.** So the pose part of our representation should live on a periodic (toroidal)
manifold by construction.

### 2.6 SSPs already are toroidal (we own this machinery)

Spatial Semantic Pointers — the `sspspace` encodings already used in this repo —
represent a continuous value `p` by *fractional binding*: take a fixed random unitary
vector `X` and raise it to the power `p`, which in the Fourier domain multiplies each
component's **phase** by `p`. A vector of phases *is* a point on a torus. So SSPs are a
toroidal code, and moreover:

- SSP neurons develop hexagonal grid-cell firing patterns — Komer, Stewart, Voelker &
  Eliasmith (2019), *A neural representation of continuous space using fractional
  binding*, CogSci; Dumont & Eliasmith (2020), *Accurate representation for spatial
  cognition using grid cells*, CogSci.
- **Fractional binding is path integration**: `X^(p+Δ) = X^p ⊛ X^Δ`. Updating the pose
  code by a movement is a single exact binding — no learned network needed.
- General VSA background: Plate (1995), *Holographic reduced representations*, IEEE
  Trans. Neural Networks; Eliasmith (2013), *How to Build a Brain* (the Semantic
  Pointer Architecture).

So "hippocampus represents location toroidally" and "encode pose with SSPs" are nearly
the same statement. The answer to the SIGReg-vs-torus tension (below) is: don't ask the
learned encoder to represent pose at all.

### 2.7 The Tolman-Eichenbaum Machine — the closest computational model

TEM (Whittington et al. 2020, Cell: *The Tolman-Eichenbaum Machine*) models the
hippocampal system as a **factorization**: a *structural* code (grid-like, describing
"where am I in the abstract space and how do actions move me" — reused across
environments) and a *sensory* code ("what is here"), bound together by fast Hebbian
memory, trained by predicting the next observation given the action. It explains many
cell types and, crucially, generalizes structure to new environments. Our plan is
TEM-shaped: SSP pose code = structural factor, learned feature encoder = sensory
factor, per-object graph = the fast binding memory.

- Whittington et al. (2020), Cell. Follow-up connecting this to transformers:
  Whittington, Warren & Behrens (2022), ICLR.

### 2.8 JEPA, LeJEPA, SIGReg — the learned encoder

- **JEPA** (joint-embedding predictive architecture): learn representations by
  predicting the *embedding* of one view from another, rather than reconstructing
  pixels — LeCun (2022), *A Path Towards Autonomous Machine Intelligence* (position
  paper); Assran et al. (2023), *I-JEPA*, CVPR.
- **LeJEPA** (Balestriero & LeCun, 2025): replaces the usual anti-collapse heuristics
  (stop-grad, EMA teachers, whitening) with **SIGReg** — a statistical test pushing the
  embedding distribution toward an isotropic Gaussian, which they prove is the optimal
  *task-agnostic* embedding distribution. This is what `midlevelfeatures/mmnist`
  implements (`minimal_s4nd.py`: Epps–Pulley test on random 1D projections).
- The tension Matt spotted: SIGReg's isotropic Gaussian is the right prior *when you
  don't know the downstream task*. But pose isn't task-agnostic — it has known
  topology. Resolution: **factorize**. The learned encoder handles appearance
  ("what"), where Gaussian is a sensible prior; SSPs handle pose ("where"), toroidal by
  construction and exempt from SIGReg.

### 2.9 Why not frozen DINO

DINO/DINOv2's self-supervision builds in invariance to its augmentations — multi-crop
scale changes, color jitter, blur (Caron et al. 2021, ICCV; Oquab et al. 2023). Those
are the right invariances for RGB internet photos, not necessarily for our patches. The
deeper point: for graph-building we don't want *any* hand-picked invariance. An encoder
trained with an **action-conditioned** predictor — predict the next fixation's
embedding given the current one *and the rotation command* — keeps exactly the
information that makes the outcome of your own actions predictable, and discards the
rest. Invariances are learned from the interaction statistics instead of imported.

Related work on action-conditioned / factorized representation learning: Kipf, van der
Pol & Welling (2020), *Contrastive Learning of Structured World Models*, ICLR;
Marchetti et al. (2023), *Equivariant Representation Learning via Class-Pose
Decomposition*, AISTATS.

### 2.10 Exploration as the RL objective

Once the representation is a graph, the natural reward is **information gain**: reward
the agent for discovering a new node or closing a loop, i.e. "rotate the object so as
to complete its model efficiently" — like an infant turning over a novel toy. This
replaces the contrastive reward (which wasn't separating classes anyway) and needs no
class labels during exploration. Discrimination becomes a *readout* of the finished
graph, not the training signal.

- Pathak et al. (2017), *Curiosity-driven exploration by self-supervised prediction*,
  ICML — the canonical prediction-error-as-reward paper.
- The CRIB dataset itself: Stojanov et al. (2019), *Incremental Object Learning from
  Contiguous Views*, CVPR (PDF is in `CRIB_Data_Generator/`).

### 2.11 The RL algorithm families — DQN vs. policy gradient

Two ways to turn rewards into a policy, and this project has now run both:

**Value-based (DQN).** The network learns `Q(s, a)`, a *prediction of future reward*
for each action in each state; the policy is implicit — act greedily on the argmax.
Sample-efficient when it works (experience replay reuses every transition), but the
argmax is brittle under sparse/noisy rewards: a lucky early estimate wins the argmax,
the agent repeats that action, and never gathers the data to correct the estimate.
This is exactly the **constant-action collapse** our v2/v5 explorers hit. Exploration
is bolted on (ε-greedy), not part of what's learned.

**Policy gradient (PG / REINFORCE).** The network outputs action *probabilities*
directly; training nudges probability mass toward actions followed by
better-than-expected reward (`∇ log π(a|s) · advantage`). No value intermediary. Two
standard variance fixes matter in practice: subtract a **baseline** (running-average
reward) so only the *advantage* teaches, and add an **entropy bonus** so the policy
is explicitly rewarded for staying stochastic — it cannot collapse onto one action
unless the reward evidence genuinely justifies it. Our 2D active-vision run
(`digits_active.py`) used entropy-regularized REINFORCE and was the first RL run in
this project that trained stably with no collapse (the arena just had no headroom —
see memory Update 20). This is the learner to port to the 3D explorer.

- Sutton & Barto (2018), *Reinforcement Learning: An Introduction*, 2nd ed., chs. 6 &
  13 — the canonical textbook treatment of both families; ch. 13 is policy gradient.
- Mnih et al. (2015), *Human-level control through deep reinforcement learning*,
  Nature — the DQN paper (Atari).
- Williams (1992), *Simple statistical gradient-following algorithms for
  connectionist reinforcement learning*, Machine Learning — REINFORCE itself.
- Mnih et al. (2016), *Asynchronous methods for deep reinforcement learning* (A3C),
  ICML — where the entropy-bonus recipe became standard practice (§4).
- Schulman et al. (2017), *Proximal Policy Optimization* (PPO), arXiv — the modern
  default PG method; read if REINFORCE's variance ever becomes the bottleneck.
- Mnih et al. (2014), *Recurrent models of visual attention* (RAM), NeurIPS — PG
  applied to exactly our problem: learning where to saccade with a glimpse network.

---

## 3. The proposed architecture in one paragraph

Each fixation produces a **content embedding** `z_what` (learned encoder on the RGB
patch, SIGReg-regularized, trained across all objects — the *slow* system) and a
**pose code** `z_where` (SSP of accumulated rotation, toroidal, updated by exact
fractional binding with the motor command — path integration, no learning). A
per-object **graph** (the *fast* system) is built online: create a node when `z_what`
is novel, merge when it matches an existing node (loop closure), and label the edge
between successively visited nodes with the relative rotation. The JEPA predictor
learns only content dynamics: "given this feature and this rotation, what feature
appears next." The RL agent's reward becomes graph growth / loop closure rate.
Recognition compares graphs (or VSA-embeddings of graphs), not averaged vectors.

## 4. Experiment sequence (and status)

1. **Instrument the pipeline** to log raw per-saccade tuples `(patch, coords,
   rotation, embedding)` instead of only the EMA. *(done — see `record_explorations.py`)*
2. **Drift experiment (go/no-go gate)**: track the *same physical feature* across known
   rotations (exact correspondence from the mesh + camera projection); measure how fast
   its embedding drifts vs. how separated different features are. If a merge threshold
   exists between "same feature, new view" and "different feature", graph-building is
   viable. Run for RGB patches and event patches. *(script: `experiments/drift_dino.py`)*
3. **Encoder bake-off**: frozen DINO vs. view-augmentation LeJEPA (mmnist recipe) vs.
   action-conditioned JEPA, same metric. Train on the recorded corpus (GPU 1 is free).
4. **Offline graph construction** from recorded episodes; evaluate node purity, loop
   closure, and same-class vs cross-class graph similarity (direct comparison against
   the EMA memory's failure).
5. **Swap the RL reward** to information gain; only after 1–4 look good.

## 5. Suggested reading order

1. Gardner et al. 2022 (the torus result Matt referenced — short, stunning)
2. Killian, Jutras & Buffalo 2012 (grid cells for saccades — our setting exactly)
3. Behrens et al. 2018 (the big-picture framework, very readable)
4. Whittington et al. 2020, TEM (the computational blueprint; read the intro + Fig 1–2
   first pass)
5. Komer et al. 2019 + Dumont & Eliasmith 2020 (SSPs ↔ grid cells; short CogSci papers)
6. Balestriero & LeCun 2025, LeJEPA (you know this one; reread §SIGReg with the
   topology question in mind)
7. O'Regan & Noë 2001 (skim for the framing; long)
8. Koenderink & van Doorn 1979 (historical, for the aspect-graph idea)
9. Sutton & Barto 2018, chs. 6 & 13 (DQN vs policy gradient — why our explorer
   collapsed and what fixed it; then Mnih 2014 RAM for PG-on-saccades)

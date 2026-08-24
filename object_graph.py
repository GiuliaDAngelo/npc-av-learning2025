# Sparse relational object model built from an exploration episode.
#
# A stream of fixations (embedding, image coord, rotation quaternion) becomes a graph:
#   nodes = distinctive features (created when an embedding matches no existing node,
#           merged into an existing node when it does -- loop closure)
#   edges = observed transitions between features, labelled with the relative rotation
#           that carried the observer from one to the other
#
# The merge threshold comes from the drift experiment (experiments/drift_dino.py):
# same-feature similarity under rotation stays above the different-feature 95th
# percentile (~0.875 for RGB/DINOv2) out to ~21 deg, so thresholds in ~0.88-0.92 are
# the operating band. See notes/BACKGROUND.md.
import numpy as np


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def rel_quat(q_from, q_to):
    """Rotation carrying pose q_from to q_to, sign-canonicalized (w >= 0)."""
    q = quat_mul(q_to, quat_conj(q_from))
    return q if q[0] >= 0 else -q


def rot_angle(q):
    """Rotation angle (radians) of a unit quaternion."""
    return 2.0 * np.arccos(np.clip(abs(q[0]), 0.0, 1.0))


class ObjectGraph:
    """Online graph construction from a fixation stream."""

    def __init__(self, merge_threshold=0.90):
        self.tau = merge_threshold
        self.protos = None          # (N, D) running-mean node prototypes, L2-normalized
        self.counts = []            # fixations absorbed per node
        self.members = []           # fixation indices per node (for exemplar lookup)
        self.node_quats = []        # per node: object orientations at member fixations
        self.node_scales = []       # per node: fixation scale (reserved for hierarchy)
        self.edges = {}             # (i, j) -> list of relative quaternions ('rotate' kind)
        self.trail = []             # node id per fixation, in order
        self.loop_closures = 0      # matched a node other than the previous one

    @property
    def n_nodes(self):
        return 0 if self.protos is None else len(self.protos)

    def add_fixation(self, emb, quat, fix_idx, scale=0):
        e = emb / (np.linalg.norm(emb) + 1e-12)
        quat = np.asarray(quat, dtype=float)
        if self.protos is None:
            self.protos = e[None]
            self.counts, self.members = [1], [[fix_idx]]
            self.node_quats, self.node_scales = [[quat.copy()]], [scale]
            node = 0
        else:
            sims = self.protos @ e
            best = int(np.argmax(sims))
            if sims[best] >= self.tau:
                node = best
                c = self.counts[node]
                p = (self.protos[node] * c + e) / (c + 1)
                self.protos[node] = p / (np.linalg.norm(p) + 1e-12)
                self.counts[node] += 1
                self.members[node].append(fix_idx)
                self.node_quats[node].append(quat.copy())
                if self.trail and node != self.trail[-1]:
                    self.loop_closures += 1
            else:
                self.protos = np.vstack([self.protos, e[None]])
                self.counts.append(1)
                self.members.append([fix_idx])
                self.node_quats.append([quat.copy()])
                self.node_scales.append(scale)
                node = self.n_nodes - 1

        if self.trail:
            prev_node = self.trail[-1]
            if prev_node != node:
                self.edges.setdefault((prev_node, node), []).append(
                    rel_quat(self._last_quat, quat))
        self.trail.append(node)
        self._last_quat = np.asarray(quat, dtype=float)
        return node

    @classmethod
    def from_episode(cls, embeddings, quats, merge_threshold=0.90):
        g = cls(merge_threshold)
        for i, (e, q) in enumerate(zip(embeddings, quats)):
            g.add_fixation(e, q, i)
        return g

    def mean_node_quats(self):
        """(N, 4) mean object orientation per node -- the node's 'home' on the viewing
        sphere. Member quats are sign-aligned before averaging (q and -q are the same
        rotation)."""
        out = np.zeros((self.n_nodes, 4))
        for i, qs in enumerate(self.node_quats):
            ref = qs[0]
            acc = np.zeros(4)
            for q in qs:
                acc += q if np.dot(q, ref) >= 0 else -q
            out[i] = acc / (np.linalg.norm(acc) + 1e-12)
        return out

    def stats(self):
        n_fix = len(self.trail)
        return dict(
            n_nodes=self.n_nodes,
            n_fixations=n_fix,
            n_edges=len(self.edges),
            loop_closures=self.loop_closures,
            revisit_rate=1.0 - self.n_nodes / max(n_fix, 1),
        )


def match_nodes(g1, g2):
    """Greedy one-to-one matching of node prototypes by cosine similarity.
    Returns list of (i, j, sim), highest-similarity pairs first."""
    if g1.n_nodes == 0 or g2.n_nodes == 0:
        return []
    S = g1.protos @ g2.protos.T
    pairs = []
    used1, used2 = set(), set()
    order = np.dstack(np.unravel_index(np.argsort(-S, axis=None), S.shape))[0]
    for i, j in order:
        if i not in used1 and j not in used2:
            pairs.append((int(i), int(j), float(S[i, j])))
            used1.add(int(i))
            used2.add(int(j))
    return pairs


def structural_similarity(g1, g2, pairs, sigma_deg=25.0):
    """Object-centric structure agreement over matched nodes.

    A node's home orientation on the viewing sphere locates its feature ON the object;
    the pairwise angular separations between node homes are invariant to both the
    exploration trajectory and the instance's global mesh frame (relative rotations
    cancel a constant frame offset). Two instances of a category share structure iff
    their matched features show the same separation pattern (a mug's handle sits the
    same angular distance from its rim, however either mug was explored).

    Returns a count-weighted mean of exp(-|dtheta| / sigma) over all matched node
    PAIRS (dense; unlike traversal edges, which encode the exploration path).
    """
    if len(pairs) < 3:
        return 0.0
    i1 = [i for i, _, _ in pairs]
    i2 = [j for _, j, _ in pairs]
    q1 = g1.mean_node_quats()[i1]
    q2 = g2.mean_node_quats()[i2]
    # pairwise separation: theta = 2*arccos(|q_a . q_b|)
    th1 = 2 * np.arccos(np.clip(np.abs(q1 @ q1.T), 0, 1))
    th2 = 2 * np.arccos(np.clip(np.abs(q2 @ q2.T), 0, 1))
    w1 = np.array(g1.counts)[i1] / sum(g1.counts)
    w2 = np.array(g2.counts)[i2] / sum(g2.counts)
    w = np.minimum(w1[:, None], w2[None, :]) * np.minimum(w1[None, :], w2[:, None])
    mask = ~np.eye(len(pairs), dtype=bool)
    agree = np.exp(-np.degrees(np.abs(th1 - th2)) / sigma_deg)
    return float((w * agree)[mask].sum() / (w[mask].sum() + 1e-12))


def spectral_match_score(g1, g2, k=3, min_sim=0.2, sigma_deg=25.0):
    """Joint content+structure matching (spectral graph matching, Leordeanu-Hebert).

    Greedy content matching fails across instances precisely when content is ambiguous
    (two mugs' parts look alike); here candidate matches vote for each other when their
    pairwise viewing-sphere separations AGREE, and the principal eigenvector of the
    compatibility matrix finds the largest geometrically-consistent assignment.
    Returns mean pairwise compatibility of the selected matches, scaled by coverage.
    """
    if g1.n_nodes < 3 or g2.n_nodes < 3:
        return 0.0
    S = g1.protos @ g2.protos.T
    cand = []
    for i in range(S.shape[0]):
        for j in np.argsort(-S[i])[:k]:
            if S[i, j] > min_sim:
                cand.append((i, int(j), float(S[i, j])))
    if len(cand) < 4:
        return 0.0
    a = np.array([c[0] for c in cand])
    b = np.array([c[1] for c in cand])
    s = np.clip(np.array([c[2] for c in cand]), 0, None)
    q1, q2 = g1.mean_node_quats(), g2.mean_node_quats()
    th1 = 2 * np.arccos(np.clip(np.abs(q1[a] @ q1[a].T), 0, 1))
    th2 = 2 * np.arccos(np.clip(np.abs(q2[b] @ q2[b].T), 0, 1))
    W = (np.exp(-np.degrees(np.abs(th1 - th2)) / sigma_deg)
         * np.sqrt(s[:, None] * s[None, :]))
    W[(a[:, None] == a[None, :]) | (b[:, None] == b[None, :])] = 0.0

    x = np.full(len(cand), 1.0 / len(cand))
    for _ in range(30):
        x = W @ x
        x /= (np.linalg.norm(x) + 1e-12)
    sel, used_a, used_b = [], set(), set()
    for m in np.argsort(-x):
        if x[m] <= 1e-9:
            break
        if a[m] in used_a or b[m] in used_b:
            continue
        sel.append(m)
        used_a.add(int(a[m]))
        used_b.add(int(b[m]))
    if len(sel) < 3:
        return 0.0
    sub = W[np.ix_(sel, sel)]
    consistency = sub.sum() / (len(sel) * (len(sel) - 1))
    coverage = len(sel) / min(g1.n_nodes, g2.n_nodes)
    return float(consistency * coverage)


def graph_similarity(g1, g2):
    """Similarity of two object graphs.

    node score: visit-count-weighted mean similarity of matched node prototypes.
    edge score: for matched node pairs, agreement of the relative-rotation
        magnitudes on edges both graphs possess (1 = identical angles).
    """
    pairs = match_nodes(g1, g2)
    if not pairs:
        return dict(node_score=0.0, edge_score=0.0, edge_overlap=0, struct_score=0.0)
    w1 = np.array(g1.counts) / sum(g1.counts)
    w2 = np.array(g2.counts) / sum(g2.counts)
    weights = np.array([min(w1[i], w2[j]) for i, j, _ in pairs])
    sims = np.array([s for _, _, s in pairs])
    node_score = float((weights * sims).sum() / (weights.sum() + 1e-12))

    m12 = {i: j for i, j, _ in pairs}
    diffs = []
    for (a, b), qs1 in g1.edges.items():
        if a in m12 and b in m12:
            qs2 = g2.edges.get((m12[a], m12[b]))
            if qs2:
                a1 = np.mean([rot_angle(q) for q in qs1])
                a2 = np.mean([rot_angle(q) for q in qs2])
                diffs.append(abs(a1 - a2))
    edge_score = float(np.exp(-np.mean(diffs) / 0.2)) if diffs else 0.0
    return dict(node_score=node_score, edge_score=edge_score, edge_overlap=len(diffs),
                struct_score=structural_similarity(g1, g2, pairs))

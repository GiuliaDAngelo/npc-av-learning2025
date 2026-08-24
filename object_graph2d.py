"""2D object graphs: the viewing-sphere machinery flattened onto the image plane.

A digit is explored by a saccading fovea exactly as a 3D object is explored by
rotation: salient fixations yield glimpse embeddings (nodes, merged by content)
and each node's 'home' is its mean fixation position in center-of-mass-centered
coordinates (the 2D analog of the viewing-sphere home orientation). Structure is
the pattern of pairwise home OFFSETS (full vectors, not distances -- digits have a
canonical orientation, and 6 vs 9 differ only by it).

Mirrors object_graph.py so results transfer between the worlds.
"""
import numpy as np


class ObjectGraph2D:
    """Online graph construction from a 2D fixation stream."""

    def __init__(self, merge_threshold=0.85):
        self.tau = merge_threshold
        self.protos = None          # (N, D) running-mean node prototypes, L2-normalized
        self.counts = []
        self.members = []
        self.node_pos = []          # per node: fixation positions (CoM-centered px)
        self.edges = {}             # (i, j) -> list of saccade offset vectors
        self.trail = []
        self.loop_closures = 0

    @property
    def n_nodes(self):
        return 0 if self.protos is None else len(self.protos)

    def add_fixation(self, emb, pos, fix_idx):
        e = emb / (np.linalg.norm(emb) + 1e-12)
        pos = np.asarray(pos, dtype=float)
        if self.protos is None:
            self.protos = e[None]
            self.counts, self.members = [1], [[fix_idx]]
            self.node_pos = [[pos.copy()]]
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
                self.node_pos[node].append(pos.copy())
                if self.trail and node != self.trail[-1]:
                    self.loop_closures += 1
            else:
                self.protos = np.vstack([self.protos, e[None]])
                self.counts.append(1)
                self.members.append([fix_idx])
                self.node_pos.append([pos.copy()])
                node = self.n_nodes - 1

        if self.trail:
            prev = self.trail[-1]
            if prev != node:
                self.edges.setdefault((prev, node), []).append(pos - self._last_pos)
        self.trail.append(node)
        self._last_pos = pos
        return node

    @classmethod
    def from_episode(cls, embeddings, positions, merge_threshold=0.85):
        g = cls(merge_threshold)
        for i, (e, p) in enumerate(zip(embeddings, positions)):
            g.add_fixation(e, p, i)
        return g

    def mean_node_pos(self):
        """(N, 2) mean fixation position per node -- the node's 'home' on the digit."""
        return np.array([np.mean(ps, axis=0) for ps in self.node_pos])


def match_nodes(g1, g2):
    """Greedy one-to-one matching of node prototypes by cosine similarity."""
    if g1.n_nodes == 0 or g2.n_nodes == 0:
        return []
    S = g1.protos @ g2.protos.T
    pairs, used1, used2 = [], set(), set()
    order = np.dstack(np.unravel_index(np.argsort(-S, axis=None), S.shape))[0]
    for i, j in order:
        if i not in used1 and j not in used2:
            pairs.append((int(i), int(j), float(S[i, j])))
            used1.add(int(i))
            used2.add(int(j))
    return pairs


def node_score(g1, g2):
    """Visit-count-weighted mean content similarity of greedily matched nodes."""
    pairs = match_nodes(g1, g2)
    if not pairs:
        return 0.0
    c1, c2 = np.array(g1.counts), np.array(g2.counts)
    w = np.array([min(c1[i] / c1.sum(), c2[j] / c2.sum()) for i, j, _ in pairs])
    s = np.array([s for _, _, s in pairs])
    return float((w * s).sum() / (w.sum() + 1e-12))


def spectral_match_score(g1, g2, k=3, min_sim=0.2, sigma_px=12.0):
    """Joint content+structure matching (Leordeanu-Hebert), 2D-offset kernel.

    Candidate content matches vote for each other when their pairwise home OFFSET
    vectors agree (translation-invariant, orientation-preserving); the principal
    eigenvector of the compatibility matrix finds the largest geometrically
    consistent assignment. Returns mean pairwise compatibility x coverage.
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
    p1, p2 = g1.mean_node_pos(), g2.mean_node_pos()
    off1 = p1[a][:, None, :] - p1[a][None, :, :]     # (M, M, 2)
    off2 = p2[b][:, None, :] - p2[b][None, :, :]
    d = np.linalg.norm(off1 - off2, axis=-1)
    W = np.exp(-d / sigma_px) * np.sqrt(s[:, None] * s[None, :])
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


def spectral_align(g_query, g_mem, k=3, min_sim=0.2, sigma_px=12.0):
    """Selected node correspondences + translation aligning g_mem onto g_query.

    Same construction as spectral_match_score but returns the assignment:
    [(i_query, j_mem)] and the mean home offset (query - mem) over matches --
    everything imagination needs to paste memory nodes onto the query canvas.
    """
    if g_query.n_nodes < 2 or g_mem.n_nodes < 2:
        return [], np.zeros(2)
    S = g_query.protos @ g_mem.protos.T
    cand = []
    for i in range(S.shape[0]):
        for j in np.argsort(-S[i])[:k]:
            if S[i, j] > min_sim:
                cand.append((i, int(j), float(S[i, j])))
    if len(cand) < 2:
        return [], np.zeros(2)
    a = np.array([c[0] for c in cand])
    b = np.array([c[1] for c in cand])
    s = np.clip(np.array([c[2] for c in cand]), 0, None)
    p1, p2 = g_query.mean_node_pos(), g_mem.mean_node_pos()
    off1 = p1[a][:, None, :] - p1[a][None, :, :]
    off2 = p2[b][:, None, :] - p2[b][None, :, :]
    d = np.linalg.norm(off1 - off2, axis=-1)
    W = np.exp(-d / sigma_px) * np.sqrt(s[:, None] * s[None, :])
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
    pairs = [(int(a[m]), int(b[m])) for m in sel]
    if not pairs:
        return [], np.zeros(2)
    shift = np.mean([p1[i] - p2[j] for i, j in pairs], axis=0)
    return pairs, shift

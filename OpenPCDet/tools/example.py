
import json
import numpy as np
from typing import Dict, List, Tuple


# ----------------------------
# Core: Jensen–Shannon Divergence
# ----------------------------
def _normalize_probs(counts: np.ndarray, alpha: float = 1e-6) -> np.ndarray:
    """
    counts: (..., K) nonnegative
    returns: (..., K) probabilities with smoothing
    """
    counts = counts.astype(np.float64)
    counts = counts + alpha
    s = counts.sum(axis=-1, keepdims=True)
    # In case something is weird (shouldn't happen after alpha), guard:
    s = np.where(s == 0, 1.0, s)
    return counts / s


def jsd(p: np.ndarray, q: np.ndarray, base: float = 2.0) -> float:
    """
    JSD between two 1D probability vectors p and q.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)

    if base == 2.0:
        log = np.log2
    elif base == np.e:
        log = np.log
    else:
        log = lambda x: np.log(x) / np.log(base)

    # KL(p||m) and KL(q||m)
    kl_pm = np.sum(p * (log(p) - log(m)))
    kl_qm = np.sum(q * (log(q) - log(m)))
    return 0.5 * (kl_pm + kl_qm)


def jsd_vectorized(P: np.ndarray, Q: np.ndarray, base: float = 2.0) -> np.ndarray:
    """
    Vectorized JSD between batches:
      P: (N, K), Q: (M, K)
    Returns:
      D: (N, M), where D[i,j] = JSD(P[i], Q[j])
    Note: use in chunks to avoid huge memory use.
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)

    M = 0.5 * (P[:, None, :] + Q[None, :, :])  # (N, M, K)

    if base == 2.0:
        log = np.log2
    elif base == np.e:
        log = np.log
    else:
        log = lambda x: np.log(x) / np.log(base)

    kl_pm = np.sum(P[:, None, :] * (log(P[:, None, :]) - log(M)), axis=-1)  # (N, M)
    kl_qm = np.sum(Q[None, :, :] * (log(Q[None, :, :]) - log(M)), axis=-1)  # (N, M)
    return 0.5 * (kl_pm + kl_qm)


# ----------------------------
# Build matrices from your dict
# ----------------------------
def build_count_matrix(
    frame_counts: Dict[str, Dict[str, int]],
    class_order: List[str] = None
) -> Tuple[List[str], List[str], np.ndarray]:
    """
    Returns:
      frame_ids: list of frame_id in consistent order
      classes: list of classes in consistent order
      X: (num_frames, num_classes) count matrix
    """
    frame_ids = list(frame_counts.keys())
    # Determine class order
    if class_order is None:
        # union all class keys to be safe
        class_set = set()
        for d in frame_counts.values():
            class_set.update(d.keys())
        classes = sorted(class_set)
    else:
        classes = list(class_order)

    K = len(classes)
    X = np.zeros((len(frame_ids), K), dtype=np.float64)

    class_to_idx = {c: i for i, c in enumerate(classes)}
    for i, fid in enumerate(frame_ids):
        d = frame_counts[fid]
        for c, v in d.items():
            if c in class_to_idx:
                X[i, class_to_idx[c]] = float(v)

    return frame_ids, classes, X


# ----------------------------
# Metrics you likely want
# ----------------------------
def aggregated_jsd_full_vs_subset(
    frame_counts: Dict[str, Dict[str, int]],
    subset_ids: List[str],
    alpha: float = 1e-6,
    base: float = 2.0
) -> float:
    """
    Aggregates counts across frames, then computes JSD(P_full, P_subset).
    Lower = subset matches full distribution better (coverage).
    """
    frame_ids, classes, X = build_count_matrix(frame_counts)
    subset_set = set(subset_ids)

    mask = np.array([fid in subset_set for fid in frame_ids], dtype=bool)
    if mask.sum() == 0:
        raise ValueError("subset_ids has no overlap with frame_counts keys.")

    full_total = X.sum(axis=0)
    subset_total = X[mask].sum(axis=0)

    P_full = _normalize_probs(full_total[None, :], alpha=alpha)[0]
    P_sub  = _normalize_probs(subset_total[None, :], alpha=alpha)[0]

    return jsd(P_full, P_sub, base=base)


def per_frame_coverage_jsd(
    frame_counts: Dict[str, Dict[str, int]],
    subset_ids: List[str],
    alpha: float = 1e-6,
    base: float = 2.0,
    chunk_full: int = 256,
    chunk_sub: int = 2048
) -> dict:
    """
    Per-frame coverage metric:
      For each full frame i, compute min_{j in subset} JSD(p_i, p_j).
    Returns summary stats: mean, p95, max, and the per-frame mins if you want.

    Lower mean/p95/max => subset better covers the full set in label-composition space.
    """
    frame_ids, classes, X = build_count_matrix(frame_counts)
    subset_set = set(subset_ids)

    full_idx = np.arange(len(frame_ids))
    sub_idx = np.array([i for i, fid in enumerate(frame_ids) if fid in subset_set], dtype=int)

    if len(sub_idx) == 0:
        raise ValueError("subset_ids has no overlap with frame_counts keys.")

    P_full = _normalize_probs(X[full_idx], alpha=alpha)
    P_sub  = _normalize_probs(X[sub_idx], alpha=alpha)

    mins = np.full((P_full.shape[0],), np.inf, dtype=np.float64)

    # Chunk over full and subset to control memory
    for i0 in range(0, P_full.shape[0], chunk_full):
        i1 = min(i0 + chunk_full, P_full.shape[0])
        Pf = P_full[i0:i1]  # (cf, K)
        cur_min = np.full((Pf.shape[0],), np.inf, dtype=np.float64)

        for j0 in range(0, P_sub.shape[0], chunk_sub):
            j1 = min(j0 + chunk_sub, P_sub.shape[0])
            Ps = P_sub[j0:j1]  # (cs, K)

            D = jsd_vectorized(Pf, Ps, base=base)  # (cf, cs)
            cur_min = np.minimum(cur_min, D.min(axis=1))

        mins[i0:i1] = cur_min

    return {
        "mean_min_jsd": float(np.mean(mins)),
        "p95_min_jsd": float(np.percentile(mins, 95)),
        "max_min_jsd": float(np.max(mins)),
        "per_frame_min_jsd": mins,   # you can drop this if too large
        "num_full_frames": int(P_full.shape[0]),
        "num_subset_frames": int(P_sub.shape[0]),
    }


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # 1) Load your dict from a JSON file (if it's saved that way)
    # with open("frame_counts.json", "r") as f:
    #     frame_counts = json.load(f)

    # Or if you already have it in memory:
    frame_counts = {
        "frameA.bin": {"car": 2, "truck": 3, "construction_vehicle": 0, "bus": 0, "trailer": 0,
                       "barrier": 0, "motorcycle": 0, "bicycle": 0, "pedestrian": 2, "traffic_cone": 3},
        "frameB.bin": {"car": 10, "truck": 0, "construction_vehicle": 0, "bus": 1, "trailer": 0,
                       "barrier": 2, "motorcycle": 0, "bicycle": 0, "pedestrian": 0, "traffic_cone": 0},
        "frameC.bin": {"car": 0, "truck": 0, "construction_vehicle": 0, "bus": 0, "trailer": 0,
                       "barrier": 0, "motorcycle": 0, "bicycle": 2, "pedestrian": 5, "traffic_cone": 1},
    }

    # with open('output/sta_per_frame.txt', 'r') as f:
    with open('waymo_output/sta_per_frame.txt', 'r') as f:
        frame_counts = json.load(f)

    subset_ids = ["frameA.bin", "frameC.bin"]

    # with open('output/select_frame_id_list.txt', 'r') as f:
    with open('waymo_output/select_frame_id_list.txt', 'r') as f:
        all_subset_dict = json.load(f)
        # subset_ids = all_subset_dict['random_0.1']
        sample_rate_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        # sample_method_list = ['random', 'loss_scene_even', 'divergence_scene_even', 'scene_even_euclidean', 'heatmap_scene_tree', 'scene_even_heatmap']
        sample_method_list = ['random', 'loss', 'divergence', 'euclidean', 'heatmap_scene_tree', 'scene_even_heatmap']

        for rate in sample_rate_list:
            for method in sample_method_list:
                key = "{}_{}".format(method, rate)
                print(key)
                # Aggregated coverage: subset vs full
                subset_ids = all_subset_dict[key]
                jsd_val = aggregated_jsd_full_vs_subset(frame_counts, subset_ids, alpha=1e-6, base=2.0)
                print("Aggregated JSD(full || subset) =", jsd_val)

                # Per-frame coverage: how well subset represents each full frame
                cov = per_frame_coverage_jsd(frame_counts, subset_ids, alpha=1e-6, base=2.0)
                print("Per-frame coverage stats:", {k: v for k, v in cov.items() if k != "per_frame_min_jsd"})

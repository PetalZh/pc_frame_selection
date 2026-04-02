import json, os
import numpy as np
from typing import List, Tuple, Optional, Iterable, Callable

# ==============================
# Configurable file parser(s)
# ==============================

EPS = 1e-12

import numpy as np
from typing import Iterable, Tuple, List, Optional

EPS = 1e-12

def _collect_ref_masses(
    ref_items: Iterable[Tuple[str, str]],
    C: int, H: int, W: int
) -> List[List[float]]:
    """
    Returns a list of length C; each entry is a list of per-frame masses for that class.
    Mass = sum of nonnegative heat values over HxW.
    """
    masses_by_class: List[List[float]] = [[] for _ in range(C)]
    for _, path in ref_items:
        Hc = LOAD_HEATMAPS(path, C, H, W)
        nonneg = np.maximum(Hc, 0.0)
        class_masses = nonneg.reshape(C, -1).sum(axis=1)
        for c in range(C):
            masses_by_class[c].append(float(class_masses[c]))
    return masses_by_class


def _otsu_threshold(values: np.ndarray, nbins: int = 128) -> float:
    """
    Otsu threshold on log-mass. Returns a positive mass threshold (not log).
    If degenerate (all same / too few), returns NaN.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v) & (v >= 0)]
    if v.size < 3:
        return np.nan

    # Work in log domain for better separation near zero
    logs = np.log(v + EPS)
    vmin, vmax = logs.min(), logs.max()
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return np.nan

    hist, edges = np.histogram(logs, bins=nbins, range=(vmin, vmax))
    hist = hist.astype(np.float64)
    prob = hist / max(hist.sum(), EPS)
    bins = 0.5 * (edges[:-1] + edges[1:])  # bin centers in log domain

    omega = np.cumsum(prob)
    mu_k = np.cumsum(prob * bins)
    mu_T = mu_k[-1]

    # Between-class variance for each threshold index
    sigma_b2 = (mu_T * omega - mu_k) ** 2 / (omega * (1.0 - omega) + EPS)

    # Ignore thresholds that put all mass on one side
    sigma_b2[(omega <= 0.0) | (omega >= 1.0)] = -np.inf

    k = int(np.argmax(sigma_b2))
    if not np.isfinite(sigma_b2[k]) or k <= 0 or k >= nbins - 2:
        return np.nan  # likely unimodal / degenerate

    log_tau = edges[k + 1]  # threshold edge to the right of k
    tau = float(np.exp(log_tau) - EPS)
    return max(tau, 0.0)


def estimate_tau_from_reference(
    ref_items: Iterable[Tuple[str, str]],
    C: int, H: int, W: int,
    method: str = "otsu",       # "otsu" or "quantile"
    quantile_q: float = 0.10,   # used if method="quantile" or otsu fails
    nbins: int = 128
) -> np.ndarray:
    """
    Compute a per-class τ using only the reference frames.
    - method="otsu": Otsu on log-mass; fallback to quantile if degenerate.
    - method="quantile": τ_c = quantile_q percentile of class-c masses.

    Returns: tau_ref with shape (C,)
    """
    masses_by_class = _collect_ref_masses(ref_items, C, H, W)
    tau_ref = np.zeros(C, dtype=np.float64)

    for c in range(C):
        vals = np.asarray(masses_by_class[c], dtype=np.float64)
        vals = vals[np.isfinite(vals) & (vals >= 0)]
        if vals.size == 0:
            tau_ref[c] = 0.0
            continue

        if method == "otsu":
            tau = _otsu_threshold(vals, nbins=nbins)
            if np.isnan(tau):
                # fallback to quantile if Otsu is not informative
                tau = float(np.quantile(vals, quantile_q))
        elif method == "quantile":
            tau = float(np.quantile(vals, quantile_q))
        else:
            raise ValueError(f"Unknown method: {method}")

        tau_ref[c] = max(tau, 0.0)

    return tau_ref



def load_heatmaps_from_txt_json(
    path: str, C: int, H: int, W: int
) -> np.ndarray:
    """
    Expects JSON file with shape:
      {
        "heatmaps": [
          [[... HxW ...]],   # class 0
          [[... HxW ...]],   # class 1
          ...
        ]
      }
    Returns np.ndarray (C, H, W) float64.
    """
    with open(path, "r") as f:
        data = json.load(f)
    arr = np.asarray(data, dtype=np.float64)
    assert arr.shape == (C, H, W), f"JSON heatmaps shape mismatch in {path}: {arr.shape} != {(C,H,W)}"
    return arr


def load_heatmaps_from_txt_blocks(
    path: str, C: int, H: int, W: int, sep: str = '---'
) -> np.ndarray:
    """
    Expects a text file with C blocks, each block is H lines of W floats.
    Blocks separated by a line with only `sep`, e.g.:

      # class 0
      <H lines each with W floats>
      ---
      # class 1
      <H lines each with W floats>
      ---
      ...

    Returns np.ndarray (C, H, W) float64.
    """
    heatmaps = []
    with open(path, "r") as f:
        block_lines = []
        for line in f:
            line = line.strip()
            if line == sep:
                if block_lines:
                    mat = _lines_to_matrix(block_lines, H, W)
                    heatmaps.append(mat)
                    block_lines = []
            elif line and not line.startswith("#"):
                block_lines.append(line)

        # last block (no trailing sep)
        if block_lines:
            mat = _lines_to_matrix(block_lines, H, W)
            heatmaps.append(mat)

    arr = np.stack(heatmaps, axis=0).astype(np.float64)
    assert arr.shape == (C, H, W), f"Block heatmaps shape mismatch in {path}: {arr.shape} != {(C,H,W)}"
    return arr


def _lines_to_matrix(lines: List[str], H: int, W: int) -> np.ndarray:
    assert len(lines) == H, f"Expected {H} lines, got {len(lines)}"
    mat = np.zeros((H, W), dtype=np.float64)
    for i, ln in enumerate(lines):
        vals = ln.split()
        assert len(vals) == W, f"Line {i} expected {W} floats, got {len(vals)}"
        mat[i] = np.fromstring(ln, sep=' ', dtype=np.float64)
    return mat


# Choose ONE parser for your data:
# LOAD_HEATMAPS: Callable[[str, int, int, int], np.ndarray] = load_heatmaps_from_txt_blocks
# If your files are JSON, switch to:
LOAD_HEATMAPS = load_heatmaps_from_txt_json


# ==============================
# Divergence machinery (streaming-friendly)
# ==============================

def _normalize_heatmap(H: np.ndarray, eps: float = EPS) -> np.ndarray:
    H = np.maximum(H, 0.0)
    s = H.sum()
    if s < eps:
        return np.ones_like(H, dtype=np.float64) / H.size
    return (H / s).astype(np.float64)


def _sample_points_from_heatmap(H: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    H_norm = _normalize_heatmap(H)
    h, w = H_norm.shape
    probs = H_norm.ravel()
    probs = np.maximum(probs, 0.0)
    probs = probs / probs.sum()

    idx = rng.choice(h * w, size=n_samples, replace=True, p=probs)
    ys, xs = np.divmod(idx, w)
    xs = xs + rng.uniform(0.0, 1.0, size=n_samples)
    ys = ys + rng.uniform(0.0, 1.0, size=n_samples)
    return np.stack([xs, ys], axis=1).astype(np.float64)


def _sliced_wasserstein_distance_2d(
    P: np.ndarray,
    Q: np.ndarray,
    n_projections: int = 64,
    rng: Optional[np.random.Generator] = None,
) -> float:
    if rng is None:
        rng = np.random.default_rng(0)

    P0 = P - P.mean(axis=0, keepdims=True)
    Q0 = Q - Q.mean(axis=0, keepdims=True)

    n = min(len(P0), len(Q0))
    if len(P0) != n:
        P0 = P0[rng.choice(len(P0), size=n, replace=False)]
    if len(Q0) != n:
        Q0 = Q0[rng.choice(len(Q0), size=n, replace=False)]

    thetas = rng.uniform(0.0, 2.0 * np.pi, size=n_projections)
    dirs = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)

    dists = []
    for v in dirs:
        p_proj = P0 @ v
        q_proj = Q0 @ v
        p_proj.sort()
        q_proj.sort()
        dists.append(np.mean(np.abs(p_proj - q_proj)))
    return float(np.mean(dists))


# ==============================
# Streaming reference builder
# ==============================

def build_reference_mixtures_from_files(
    ref_items: Iterable[Tuple[str, str]],  # (frame_id, path) for the 500 ref frames
    C: int, H: int, W: int,
    n_samples_per_class: int = 4096,
    per_frame_samples: int = 256,   # how many samples to draw from each class per frame file
    seed: int = 0,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Iterates over reference files one-by-one. For each file and each class,
    draws a small number of samples and accumulates into per-class buffers.
    At the end, downsamples to n_samples_per_class per class.
    """
    rng = np.random.default_rng(seed)
    buffers: List[List[np.ndarray]] = [[] for _ in range(C)]

    # Mass accumulators
    mass_sum = np.zeros(C, dtype=np.float64)
    frame_count = 0

    for _, path in ref_items:
        # frame_H = LOAD_HEATMAPS(path, C, H, W)  # loads only this frame
        # for c in range(C):
        #     pts = _sample_points_from_heatmap(frame_H[c], per_frame_samples, rng)
        #     buffers[c].append(pts)
        frame_H = LOAD_HEATMAPS(path, C, H, W)  # (C,H,W)
        # accumulate class masses (clip negatives to zero)
        nonneg = np.maximum(frame_H, 0.0)
        mass_sum += nonneg.reshape(C, -1).sum(axis=1)
        frame_count += 1

        # collect per-frame samples for mixtures
        for c in range(C):
            pts = _sample_points_from_heatmap(frame_H[c], per_frame_samples, rng)
            buffers[c].append(pts)

    mixtures: List[np.ndarray] = []
    for c in range(C):
        if not buffers[c]:
            mixtures.append(np.zeros((0, 2), dtype=np.float64))
            continue
        cloud = np.concatenate(buffers[c], axis=0)
        if len(cloud) > n_samples_per_class:
            idx = rng.choice(len(cloud), size=n_samples_per_class, replace=False)
            cloud = cloud[idx]
        mixtures.append(cloud)

    if frame_count == 0:
        ref_mass_mean = np.zeros(C, dtype=np.float64)
    else:
        ref_mass_mean = mass_sum / float(frame_count)
    return mixtures, ref_mass_mean


def score_frame_file_against_reference(
    path: str,
    C: int, H: int, W: int,
    ref_mixtures: List[np.ndarray],
    ref_mass_mean: np.ndarray,
    n_samples_per_class: int = 2048,
    n_projections: int = 64,
    class_weights: Optional[np.ndarray] = None,
    alpha: float = 0.5,
    tau_cand: Optional[np.ndarray] = None,
    seed: int = 0,
) -> float:
    rng = np.random.default_rng(seed)
    frame_H = LOAD_HEATMAPS(path, C, H, W)  # load one candidate frame

    if class_weights is None:
        class_weights = np.ones(C, dtype=np.float64) / C
    else:
        class_weights = np.asarray(class_weights, dtype=np.float64)
        s = max(class_weights.sum(), 1e-12)
        class_weights = class_weights / s
        # class_weights = class_weights / max(class_weights.sum(), 1e-12)

    per_class_scores = np.zeros(C, dtype=np.float64)
    
    for c in range(C):
        # candidate mass (clip negatives)
        m_cand = float(np.maximum(frame_H[c], 0.0).sum())
        m_ref = float(ref_mass_mean[c])

        # mass penalty in [0,1]
        mpen = _bounded_mass_penalty(m_cand, m_ref, eps=1e-12)

        # geometric difference via sliced-Wasserstein
        ref_pts = ref_mixtures[c]
        if (m_cand <= tau_cand[c]) or (len(ref_pts) == 0):
            # No geometric reference: rely purely on mass penalty.
            sw = 0.0
        else:
            cand_pts = _sample_points_from_heatmap(frame_H[c], n_samples_per_class, rng)
            sw = _sliced_wasserstein_distance_2d(
                cand_pts, ref_pts, n_projections=n_projections, rng=rng
            )

        per_class_scores[c] = sw + alpha * mpen

    return float(np.dot(per_class_scores, class_weights))

def _bounded_mass_penalty(m_cand: float, m_ref: float, eps: float = 1e-12) -> float:
    """
    Symmetric, bounded (0..1) relative difference.
    Large when one is near-zero and the other is not, zero when equal.
    """
    return float(abs(m_cand - m_ref) / (m_cand + m_ref + eps))

def rank_candidate_files_by_divergence_with_mass(
    ref_items: Iterable[Tuple[str, str]],
    cand_items: Iterable[Tuple[str, str]],
    C: int, H: int, W: int,
    top_k: int,
    n_ref_samples_per_class: int = 4096,
    per_ref_frame_samples: int = 256,
    n_cand_samples_per_class: int = 2048,
    n_projections: int = 64,
    class_weights: Optional[np.ndarray] = None,
    alpha: float = 0.5,
    tau_ref: Optional[np.ndarray] = None,   # <-- allow auto
    seed: int = 0,
) -> Tuple[List[str], np.ndarray]:
    """
    Same API as before but includes mass penalty via `alpha`.
    Returns (top_ids, top_scores) with scores sorted descending.
    """
    # 1) Build reference mixtures AND per-class mean masses
    ref_mixtures, ref_mass_mean = build_reference_mixtures_from_files(
        ref_items, C, H, W,
        n_samples_per_class=n_ref_samples_per_class,
        per_frame_samples=per_ref_frame_samples,
        seed=seed,
    )

    # 2) Score candidates
    cand_items_list = list(cand_items)
    scores = np.zeros(len(cand_items_list), dtype=np.float64)
    for i, (_cid, cpath) in enumerate(cand_items_list):
        scores[i] = score_frame_file_against_reference(
            cpath, C, H, W,
            ref_mixtures=ref_mixtures,
            ref_mass_mean=ref_mass_mean,
            n_samples_per_class=n_cand_samples_per_class,
            n_projections=n_projections,
            class_weights=class_weights,
            alpha=alpha,
            tau_cand=tau_ref,
            seed=seed + i,
        )

    # 3) Select top-K (descending)
    if top_k >= len(scores):
        order = np.argsort(-scores)
        return [cand_items_list[i][0] for i in order], scores[order]

    top_idx = np.argpartition(-scores, top_k)[:top_k]
    top_sorted = top_idx[np.argsort(-scores[top_idx])]
    top_ids = [cand_items_list[i][0] for i in top_sorted]
    return top_ids, scores[top_sorted]

def generateKITTIRank():
    all_idx = []
    all_items = os.listdir('/data/kitti_heatmap')
    for item in all_items:
        if item.endswith('.txt'):
            item_id = item.split('.')[0]
            all_idx.append(item_id)
    ref_frame_ids = all_idx[:500]  # First 500 for reference
    cand_frame_ids = all_idx[500:]  # Next 3000 for candidates
    ref_items  = [(fid, f"/data/kitti_heatmap/{fid}.txt")  for fid in ref_frame_ids]
    cand_items = [(fid, f"/data/kitti_heatmap/{fid}.txt") for fid in cand_frame_ids]

    # Heatmap geometry and number of classes
    C, H, W = 3, 400, 352

    # Selection hyperparams
    top_k = 3000
    class_weights = np.array([1,10,10])  # or e.g., np.array([1,1,2,3])

    top_ids, top_scores = rank_candidate_files_by_divergence_with_mass(
        ref_items, cand_items,
        C, H, W,
        top_k=top_k,
        n_ref_samples_per_class=4096,
        per_ref_frame_samples=256,
        n_cand_samples_per_class=4096,
        n_projections=128,
        class_weights=class_weights,
        alpha=0.5,  # <-- tune this
        seed=123,
    )

    print("Top-K candidate frame_ids:", top_ids[:20])
    print("Top-K scores:", top_scores[:20])
    # top_scores.tolist()  # Convert to list for JSON serialization
    # top_scores.tolist()
    output_dict = {'score_list': top_scores.tolist(), 'id_list': top_ids}
    file_name = 'kitti_heatmap_EMD_1_10_10.txt'
    with open(file_name, 'w') as f: 
        json.dump(output_dict, f)


def generateNuscenesRank():
    all_idx = []
    all_items = os.listdir('/data/nuscenes_heatmap2')
    for item in all_items:
        if item.endswith('.txt'):
            item_id = item.split('.')[0]
            all_idx.append(item_id)
    print('number of frames:', len(all_idx))
    
    with open('nus_pretrain_file_list.txt', 'r') as f:
        pretrain_idx = json.load(f)
    ref_frame_ids = []
    for id in pretrain_idx:
        id = id.replace('samples/LIDAR_TOP/', '')
        id = id.split('.')[0]
        if id in all_idx:
            ref_frame_ids.append(id)
    # ref_frame_ids = all_idx[:num_pretrain]  # First 500 for reference
    # cand_frame_ids have all items not in ref_frame_ids
    cand_frame_ids = [fid for fid in all_idx if fid not in ref_frame_ids]
    ref_items  = [(fid, f"/data/nuscenes_heatmap2/{fid}.pcd.txt")  for fid in ref_frame_ids]
    cand_items = [(fid, f"/data/nuscenes_heatmap2/{fid}.pcd.txt") for fid in cand_frame_ids]

    # Heatmap geometry and number of classes
    C, H, W = 10, 128, 128

    tau_ref = estimate_tau_from_reference(
        ref_items, C, H, W, method='otsu', quantile_q=0.5
    )

    # print(tau_ref)

    # Selection hyperparams
    top_k = 24000
    class_weights = np.array([1,1,1,1,1,1,1,1,1,1])  # or e.g., np.array([1,1,2,3])

    top_ids, top_scores = rank_candidate_files_by_divergence_with_mass(
        ref_items, cand_items,
        C, H, W,
        top_k=top_k,
        n_ref_samples_per_class=4096,
        per_ref_frame_samples=256,
        n_cand_samples_per_class=4096,
        n_projections=128,
        class_weights=class_weights,
        alpha=0.5,  # <-- tune this
        tau_ref=tau_ref,
        seed=123,
    )

    print("Top-K candidate frame_ids:", top_ids[:20])
    print("Top-K scores:", top_scores[:20])
    # top_scores.tolist()  # Convert to list for JSON serialization
    # top_scores.tolist()
    output_dict = {'score_list': top_scores.tolist(), 'id_list': top_ids}
    file_name = 'nuscenes_heatmap_EMD_4.txt'
    with open(file_name, 'w') as f: 
        json.dump(output_dict, f)

        
# def test():
#     all_items = os.listdir('/data/nuscenes_heatmap')
#     for item in all_items:
#         path = os.path.join('/data/nuscenes_heatmap', item)
#         with open(path, 'r') as f:
#             contents = json.load(f)
#         heatmap_list = []
#         for content in contents:
#             if len(content) == 2:
#                 for heatmap in content:
#                     heatmap_list.append(heatmap)
#             else:
#                 heatmap_list.append(content)
#         output_path = os.path.join('/data/nuscenes_heatmap2', item)
#         with open(output_path, 'w') as f: 
#             json.dump(heatmap_list, f)
        
        
# ==============================
# Example wiring
# ==============================
if __name__ == "__main__":
    # Example metadata lists. Replace with your actual enumerations of (frame_id, file_path).
    # For instance, if your files are named like "<frame_id>.txt":
    
    # generateKITTIRank()
    # test()
    generateNuscenesRank()


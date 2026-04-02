import os, json
import numpy as np
from sklearn.neighbors import BallTree
import time
import glob, random
from typing import Optional, Dict

# --- reuse your LOAD_HEATMAPS ---

def weighted_quantiles(values: np.ndarray, weights: np.ndarray, qs: np.ndarray):
    # values (N,), weights (N,) >=0, qs in [0,1]
    order = np.argsort(values)
    v = values[order]; w = weights[order]
    cw = np.cumsum(w); cw = cw / max(cw[-1], 1e-12)
    return np.interp(qs, cw, v)

def build_sw_signature(
    Hmap: np.ndarray,     # (C,H,W), nonneg
    P: int = 32,          # projections
    Q: int = 32,          # quantiles per projection
    class_weights=None,
    seed: int = 123
) -> np.ndarray:
    Hmap = np.asarray(Hmap, dtype=np.float64)
    
    Hmap = np.squeeze(Hmap)
    C, H, W = Hmap.shape
    if class_weights is None:
        # print('number of channel: {}'.format(C))
        # class_weights = np.ones(C, dtype=np.float64)
        class_weights = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    cw = class_weights / max(class_weights.sum(), 1e-12)

    # Pixel centers
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    xs = (xs + 0.5).ravel()
    ys = (ys + 0.5).ravel()

    rng = np.random.default_rng(seed)
    thetas = rng.uniform(0.0, 2.0*np.pi, size=P)
    dirs = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)
    qs = (np.arange(Q) + 0.5) / Q

    sig_parts = []
    for c in range(C):
        # print(Hmap[c]+100)
        Hmap[c]+=15
        # print(Hmap[c])
        Hc = np.maximum(Hmap[c], 0.0)
        w  = Hc.ravel().astype(np.float64)
        s  = w.sum()
        print(s)
        if s <= 0:
            sig_parts.append(np.zeros(P*Q, dtype=np.float32))
            continue
        w /= s
        class_sig = []
        for vx, vy in dirs:
            proj = xs*vx + ys*vy    # (H*W,)
            qv = weighted_quantiles(proj, w, qs)  # (Q,)
            class_sig.append(qv.astype(np.float32))
        class_sig = np.concatenate(class_sig, axis=0)  # (P*Q,)
        # weight this class by cw[c] so L1 sums reproduce class weights
        class_sig = (cw[c] * class_sig).astype(np.float32)
        # print('aaaaa')
        # print("class sig: {}".format(class_sig.shape))
        sig_parts.append(class_sig)

    return np.concatenate(sig_parts, axis=0)  # shape (C*P*Q,)

def LOAD_HEATMAPS(path):
    with open(path, "r") as f:
        data = json.load(f)
        arr = np.asarray(data, dtype=np.float64)
        # assert arr.shape == (C, H, W), f"JSON heatmaps shape mismatch in {path}: {arr.shape} != {(C,H,W)}"
        return arr

def build_signatures(items, P=32, Q=32, class_weights=None, seed=123):
    sigs = []
    ids  = []
    # file_dir = '/data/nuscenes_heatmap2'
    file_dir = '/data/waymo_heatmap'
    for item in items:
        # item = item + '.pcd.txt' # nus
        item = item + '.txt' # waymo
        path = os.path.join(file_dir, item)

        Hmap = LOAD_HEATMAPS(path)#.astype(np.float32)
        # build signature from heatmap
        try:
            sig  = build_sw_signature(Hmap, P=P, Q=Q, class_weights=class_weights, seed=seed)
            # print(sig)
        except Exception as e:
            print('error in file: ', path)
            print(Hmap.shape)
        # print('type of sig: ', type(sig))

        sigs.append(sig)
        ids.append(item.replace('.pcd.txt',''))
    return np.stack(sigs, 0), ids

def load_all_signatures(path = '/data/nuscenes_heatmap_signatures.txt'):
    with open(path, 'r') as f:
        sig_dict = json.load(f)
        # print(sig_dict.keys())
    return sig_dict


def get_signature_by_ids(item_ids, sig_dict):
    sigs = []
    ids = []
    for item_id in item_ids:
        item_id = item_id + '.txt'
        sig = np.array(sig_dict[item_id], dtype=np.float32)
        sigs.append(sig)
    return np.stack(sigs, 0), ids


# def tree_nn_with_sw_signatures(
#     ref_items, cand_items, C, H, W,
#     P=32, Q=32, class_weights=None, seed=123, leaf_size=40
# ):
#     X_ref, ref_ids   = build_signatures(ref_items,  C,H,W, P,Q, class_weights, seed)
#     X_cand, cand_ids = build_signatures(cand_items, C,H,W, P,Q, class_weights, seed)

#     tree = BallTree(X_ref, metric='manhattan', leaf_size=leaf_size)
#     dists, idxs = tree.query(X_cand, k=1)

#     return {
#         "id_list": cand_ids,
#         "score_list": dists.ravel().tolist(),             # ≈ SW (scaled)
#         "neighbor_list": [ref_ids[i] for i in idxs.ravel().tolist()],
#         "meta": {"P": P, "Q": Q}
#     }



# def load_feature(file_path):
#     with open(file_path, 'r') as f:
#         data = json.load(f)
#     return np.array(data, dtype=np.float32)

def build_ball_tree():
    # with open('nus_pretrain_file_list.txt', 'r') as f:
    #     pretrain_idx = json.load(f)
    # file_dir = '/data/feature_nuscenes_train'

    with open('waymo_output/pretrain_file_list.txt', 'r') as f:
        pretrain_idx = json.load(f)
    file_dir = '/data/feature_waymo_train'

    all_idx = []
    all_items = os.listdir(file_dir)
    for item in all_items:
        if item.endswith('.txt'):
            item_id = item.split('.')[0]
            all_idx.append(item_id)

    for i in range(len(pretrain_idx)):
        pretrain_idx[i] = pretrain_idx[i].replace("samples/LIDAR_TOP/","").replace('.pcd.bin', '')

    # all_feature_list = []
    all_idx_list = []
    for idx in all_idx:
        if idx not in pretrain_idx:
            all_idx_list.append(idx)

    start_time = time.time()

    X_ref, ref_ids   = build_signatures(pretrain_idx[:1], 64 ,64)
    X_cand, cand_ids = build_signatures(all_idx_list[:1], 64 ,64)

    # all_signatures = load_all_signatures(path='/data/waymo_heatmap_signatures.txt')

    # X_ref, ref_ids  = get_signature_by_ids(pretrain_idx, all_signatures)
    # X_cand, ref_ids  = get_signature_by_ids(all_idx_list, all_signatures)

    print('build ball tree')
    tree = BallTree(X_ref, metric='manhattan', leaf_size=40)
    dists, idxs = tree.query(X_cand, k=1)

    end_time = time.time()
    elapsed_time = end_time - start_time
    # print(f"Execution time: {elapsed_time:.4f} seconds")

    score_list = dists.ravel().tolist()
    print(f"Score list: {score_list}")
    # print(all_idx_list)

    output_dict = {
        'id_list': all_idx_list,
        'score_list': score_list,
        'neighbor_list': idxs.ravel().tolist()
    }

    # output_file_name = 'nuscenes_heatmap_tree2.txt'
    # with open(output_file_name, 'w') as f: 
    #     json.dump(output_dict, f)
    # output_file_name = 'waymo_output/heatmap_tree.txt'
    # with open(output_file_name, 'w') as f: 
    #     json.dump(output_dict, f)

def get_heatmap_mean(items):
    file_dir = '/data/nuscenes_heatmap2'
    # for item in items:
    #     item = item + '.pcd.txt'
    #     path = os.path.join(file_dir, item)
    #     Hmap = LOAD_HEATMAPS(path)

    return np.mean(np.stack([LOAD_HEATMAPS(os.path.join(file_dir, f + '.pcd.txt')) for f in items], axis=0), axis=0)
def get_heatmap_mean_waymo(items):
    file_dir = '/data/waymo_heatmap'
    # for item in items:
    #     item = item + '.pcd.txt'
    #     path = os.path.join(file_dir, item)
    #     Hmap = LOAD_HEATMAPS(path)

    return np.mean(np.stack([LOAD_HEATMAPS(os.path.join(file_dir, f+ '.txt')) for f in items], axis=0), axis=0)
    
def _normalize01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo <= 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return (x - lo) / (hi - lo)

import numpy as np

def allocate_budget_from_distances(
    scene_ids,
    d_pre,                       # array-like, distance to nearest pretrain for each scene
    d_scene_knn_mean,            # array-like, mean distance to k nearest scenes (excl. self)
    total_budget: int,
    *,
    w_pre: float = 0.7,
    w_scene: float = 0.3,
    alpha: float = 1.0,          # >1 sharpens, <1 smooths (optional)
    temperature: Optional[float] = None,            # softmax temperature (optional alternative to alpha)
    caps: Optional[Dict[str, int]] = None,         # {scene_id: max selectable frames}; defaults to +inf
    mins: Optional[Dict[str, int]] = None,         # {scene_id: minimum frames}; defaults to 0
    available: Optional[Dict[str, int]] = None     # {scene_id: available frames} (optional additional cap)
):
    """
    Returns: dict {scene_id: allocated_int}
    Strategy:
      1) weight(s) = score(s) from distances (normalized)
      2) proportional share of total_budget
      3) largest remainder rounding with caps/mins
    """
    scene_ids = list(scene_ids)
    d_pre = np.asarray(d_pre, dtype=np.float64)
    d_nn  = np.asarray(d_scene_knn_mean, dtype=np.float64)

    # 1) weights from your score definition
    d_pre_n = _normalize01(d_pre)
    d_nn_n  = _normalize01(d_nn)
    w = w_pre * d_pre_n + w_scene * (1.0 - d_nn_n)  # higher is better

    # Optional shaping
    if temperature is not None:
        # softmax with temperature
        x = w / max(temperature, 1e-6)
        x = x - np.max(x)
        w = np.exp(x)
    elif alpha != 1.0:
        w = np.power(np.maximum(w, 0.0), alpha)

    # Guard: if all weights zero, fall back to equal split
    if not np.any(w > 0):
        w = np.ones_like(w)

    # Caps / minimums per scene
    n = len(scene_ids)
    cap_arr = np.full(n, np.inf, dtype=float)
    min_arr = np.zeros(n, dtype=int)

    if caps:
        for i, sid in enumerate(scene_ids):
            if sid in caps:
                cap_arr[i] = float(caps[sid])
    if available:
        for i, sid in enumerate(scene_ids):
            if sid in available:
                cap_arr[i] = min(cap_arr[i], float(available[sid]))
    if mins:
        for i, sid in enumerate(scene_ids):
            if sid in mins:
                min_arr[i] = int(max(0, mins[sid]))

    # Satisfy minimums first
    base = min_arr.copy()
    remaining = max(0, int(total_budget) - int(base.sum()))
    if remaining == 0:
        # Just clamp to caps and return
        base = np.minimum(base, cap_arr).astype(int)
        return {sid: int(base[i]) for i, sid in enumerate(scene_ids)}

    # 2) proportional shares for the remaining budget
    w = w / w.sum()
    ideal = w * remaining
    add = np.floor(ideal).astype(int)
    rem = remaining - int(add.sum())

    # 3) largest remainder with caps
    frac = ideal - add
    order = np.argsort(-frac)  # descending fractional parts

    alloc = base + add
    # respect caps while distributing remainders
    for i in order:
        if rem <= 0:
            break
        if alloc[i] < cap_arr[i]:
            alloc[i] += 1
            rem -= 1

    # If some scenes are already at cap and we still have remainder, sweep again
    if rem > 0:
        # find eligible indices not at cap
        eligible = [i for i in range(n) if alloc[i] < cap_arr[i]]
        for i in eligible:
            if rem <= 0:
                break
            alloc[i] += 1
            rem -= 1

    # Final clamp to caps
    alloc = np.minimum(alloc, cap_arr).astype(int)

    # It’s possible (rare) that clamping caused sum(alloc) < total_budget (e.g., caps too tight).
    # If you want to force full usage, you can redistribute here to any scenes still below cap.

    return {sid: int(alloc[i]) for i, sid in enumerate(scene_ids)}


def round_size_from_scores(scores, remaining, T=0.5, rho_min=0.05, rho_max=0.25):
    s = np.asarray(scores, dtype=np.float64)
    x = (s - s.max()) / max(T, 1e-6)
    w = np.exp(x); w /= w.sum()
    H = -(w * (np.log(w + 1e-12))).sum() / np.log(len(w) + 1e-12)  # in [0,1]
    p = 1.0 - H
    rho = rho_min + (rho_max - rho_min) * p
    return int(min(remaining, max(1, np.ceil(rho * remaining))))   

def build_ball_tree_with_scene():

    with open('nus_pretrain_file_list.txt', 'r') as f:
        pretrain_idx = json.load(f)
    

    file_dir = '/data/feature_nuscenes_train'
    
    all_idx = []
    all_items = os.listdir(file_dir)
    for item in all_items:
        if item.endswith('.txt'):
            item_id = item.split('.')[0]
            all_idx.append(item_id)

    for i in range(len(pretrain_idx)):
        pretrain_idx[i] = pretrain_idx[i].replace("samples/LIDAR_TOP/","").replace('.pcd.bin', '')

    # scenes -> frames
    with open('nuscenes_scenes.txt', "r") as f:
        scene_dict = json.load(f)

    for scene in scene_dict.keys():
        frames = scene_dict[scene]
        for i in range(len(frames)):
            frames[i] = frames[i].replace("samples/LIDAR_TOP/","").replace('.pcd.bin', '')
            # print(frames[i])
            if frames[i] not in all_idx:
                frames[i] = None
                
        frames = [f for f in frames if f is not None]

        scene_dict[scene] = frames
    
    scene_heatmaps = getSceneHeatmapFromFile()
    
    # scene_heatmaps = {}
    # counter = 0
    # for scene in scene_dict.keys():
    #     counter += 1
    #     frames = scene_dict[scene]
    #     if len(frames) == 0:
    #         continue
    #     heatmap_mean = get_heatmap_mean(frames)
    #     scene_heatmaps[scene] = heatmap_mean.tolist()
        
        # if counter % 50 == 0 or counter == len(scene_dict):
        #     print('processing scene: ', counter)
        #     output_file_name = '/data/nuscenes_scene_heatmap/{}.txt'.format(counter)
        #     with open(output_file_name, 'w') as f: 
        #         json.dump(scene_heatmaps, f)
        #         scene_heatmaps = {}
    P, Q, seed = 64, 64, 123
    metric = "manhattan"
    leaf_size = 40
    k_scene = 5
    
    scene_ids = list(scene_heatmaps.keys())
    # scene_ids = scene_ids[:5]  # for testing, use only first 50 scenes
    # pretrain_idx = pretrain_idx[:5]  # for testing, use only first 500 pretrain samples
    X_scene = np.stack(
        [build_sw_signature(scene_heatmaps[sid], P=P, Q=Q, class_weights=None, seed=seed)
         for sid in scene_ids],
        axis=0
    ).astype(np.float32) 

    all_signatures = load_all_signatures(path='/data/waymo_heatmap_signatures.txt')

    # X_pretrain, ref_ids  = build_signatures(pretrain_idx, P=P, Q=Q)
    X_pretrain, ref_ids  = get_signature_by_ids(pretrain_idx, all_signatures)

    total_budget = 20000

    selected_frames = []
    selected_scene_ids = set()
    
    round_size_list = []
    while True:
        if total_budget >= 0:
            # step 1: build ball tree for pretrain
            tree_pre = BallTree(X_pretrain, metric=metric, leaf_size=leaf_size)
            d_pre = tree_pre.query(X_scene, k=1)[0][:, 0]

            tree_scene = BallTree(X_scene, metric=metric, leaf_size=leaf_size)
            k_eff = min(k_scene + 1, X_scene.shape[0])  # +1 because self-distance is 0
            
            if k_eff <= 1:
                d_scene = np.zeros(len(scene_ids), dtype=np.float64)
            else:
                knn_d = tree_scene.query(X_scene, k=k_eff)[0]    # includes self at [:,0] == 0
                d_scene = np.mean(knn_d[:, 1:], axis=1)          # exclude self
            # get round size, and ranking
            ranking, scores = get_ranking(d_pre, d_scene, scene_ids) # (scene_heatmaps, pretrain_idx, scene_dict)
            round_size = round_size_from_scores(scores, total_budget, T=0.5, rho_min=0.05, rho_max=0.25)
            
            print(f"Round size: {round_size}")
            round_size_list.append(round_size)
            # sample from top 5 scenes
            left_scene_ids = [r["scene_id"] for r in ranking if r["scene_id"] not in selected_scene_ids]

            if len(left_scene_ids) == 0:
                print("No more scenes to sample from.")
                break
            elif len(left_scene_ids) > 5:
                round_selected_scene_ids = left_scene_ids[:10]
            round_selected_frames = doSample(tree_pre, scene_dict, ranking, round_size, round_selected_scene_ids, all_signatures)
            selected_frames.extend(round_selected_frames)
            
            selected_scene_ids.update(round_selected_scene_ids)

            # prepare next round sampling
            # 1. Add sample to pretrain set
            # X_pretrain_new, ref_ids  = build_signatures(round_selected_frames, P=P, Q=Q)
            X_pretrain_new, ref_ids  = get_signature_by_ids(round_selected_frames, all_signatures)
            X_pretrain = np.concatenate([X_pretrain, X_pretrain_new], axis=0)

            # 2. Remove sample from scene_dict
            # 3. Update total budget
            total_budget -= round_size
            # 4. update tree
        else:
            print("Budget exhausted.")
            break
    
    if len(selected_frames) < total_budget:
        budget = total_budget - len(selected_frames)
        frame_left = all_idx - selected_frames - pretrain_idx
        random.seed(25)
        random_numbers = random.sample(range(0, len(frame_left)-1), budget)
        random_select_frame = [frame_left[i] for i in random_numbers]
        selected_frames.extend(random_select_frame)
    
    print("round sizes: ", round_size_list)
    output_file_name = 'scene_frame_selection_3.txt'
    with open(output_file_name, 'w') as f: 
        json.dump(selected_frames, f)

    
def saveSceneHeatmapToFile():
    with open('waymo_output/pretrain_file_list.txt', 'r') as f:
        pretrain_idx = json.load(f)
    
    with open('waymo_output/waymo_scenes.txt', "r") as f:
        scene_dict = json.load(f)
    
    file_dir = '/data/feature_waymo_train'
    all_idx = []
    all_items = os.listdir(file_dir)
    for item in all_items:
        if item.endswith('.txt'):
            item_id = item.split('.')[0]
            all_idx.append(item_id)


    # scenes -> frames
    with open('waymo_output/waymo_scenes.txt', "r") as f:
        scene_dict = json.load(f)

    print(len(scene_dict.keys()))


    for scene in scene_dict.keys():
        frames = scene_dict[scene]
        for i in range(len(frames)):
            # print(frames[i])
            if frames[i] not in all_idx:
                frames[i] = None
                
        frames = [f for f in frames if f is not None]
        scene_dict[scene] = frames
        
    scene_heatmaps = {}
    counter = 0
    for scene in scene_dict.keys():
        counter += 1
        frames = scene_dict[scene]
        if len(frames) == 0:
            continue
        heatmap_mean = get_heatmap_mean_waymo(frames)
        scene_heatmaps[scene] = heatmap_mean.tolist()
        
        if counter % 50 == 0 or counter == len(scene_dict):
            print('processing scene: ', counter)
            output_file_name = '/data/waymo_scene_heatmap/{}.txt'.format(counter)
            with open(output_file_name, 'w') as f: 
                json.dump(scene_heatmaps, f)
                scene_heatmaps = {}

def build_ball_tree_with_scene_waymo():
    with open('waymo_output/pretrain_file_list.txt', 'r') as f:
        pretrain_idx = json.load(f)
    
    file_dir = '/data/feature_waymo_train'
    all_idx = []
    all_items = os.listdir(file_dir)
    for item in all_items:
        if item.endswith('.txt'):
            item_id = item.split('.')[0]
            all_idx.append(item_id)

    # for i in range(len(pretrain_idx)):
    #     pretrain_idx[i] = pretrain_idx[i].replace("samples/LIDAR_TOP/","").replace('.pcd.bin', '')

    # scenes -> frames
    with open('waymo_output/waymo_scenes.txt', "r") as f:
        scene_dict = json.load(f)

    for scene in scene_dict.keys():
        frames = scene_dict[scene]
        for i in range(len(frames)):
            # frames[i] = frames[i].replace("samples/LIDAR_TOP/","").replace('.pcd.bin', '')
            # print(frames[i])
            if frames[i] not in all_idx:
                frames[i] = None
                
        frames = [f for f in frames if f is not None]

        scene_dict[scene] = frames
    
    scene_heatmaps = getSceneHeatmapFromFile()
    
    # scene_heatmaps = {}
    # counter = 0
    # for scene in scene_dict.keys():
    #     counter += 1
    #     frames = scene_dict[scene]
    #     if len(frames) == 0:
    #         continue
    #     heatmap_mean = get_heatmap_mean(frames)
    #     scene_heatmaps[scene] = heatmap_mean.tolist()
        
        # if counter % 50 == 0 or counter == len(scene_dict):
        #     print('processing scene: ', counter)
        #     output_file_name = '/data/nuscenes_scene_heatmap/{}.txt'.format(counter)
        #     with open(output_file_name, 'w') as f: 
        #         json.dump(scene_heatmaps, f)
        #         scene_heatmaps = {}
    P, Q, seed = 64, 64, 123
    metric = "manhattan"
    leaf_size = 40
    k_scene = 5
    
    scene_ids = list(scene_heatmaps.keys())
    # scene_ids = scene_ids[:5]  # for testing, use only first 50 scenes
    # pretrain_idx = pretrain_idx[:5]  # for testing, use only first 500 pretrain samples
    X_scene = np.stack(
        [build_sw_signature(scene_heatmaps[sid], P=P, Q=Q, class_weights=None, seed=seed)
         for sid in scene_ids],
        axis=0
    ).astype(np.float32) 

    all_signatures = load_all_signatures(path='/data/waymo_heatmap_signatures.txt')

    # X_pretrain, ref_ids  = build_signatures(pretrain_idx, P=P, Q=Q)
    X_pretrain, ref_ids  = get_signature_by_ids(pretrain_idx, all_signatures)

    total_budget = 25000

    selected_frames = []
    selected_scene_ids = set()
    
    round_size_list = []
    while True:
        if total_budget >= 0:
            # step 1: build ball tree for pretrain
            tree_pre = BallTree(X_pretrain, metric=metric, leaf_size=leaf_size)
            d_pre = tree_pre.query(X_scene, k=1)[0][:, 0]

            tree_scene = BallTree(X_scene, metric=metric, leaf_size=leaf_size)
            k_eff = min(k_scene + 1, X_scene.shape[0])  # +1 because self-distance is 0
            
            if k_eff <= 1:
                d_scene = np.zeros(len(scene_ids), dtype=np.float64)
            else:
                knn_d = tree_scene.query(X_scene, k=k_eff)[0]    # includes self at [:,0] == 0
                d_scene = np.mean(knn_d[:, 1:], axis=1)          # exclude self
            # get round size, and ranking
            ranking, scores = get_ranking(d_pre, d_scene, scene_ids) # (scene_heatmaps, pretrain_idx, scene_dict)
            round_size = round_size_from_scores(scores, total_budget, T=0.5, rho_min=0.05, rho_max=0.25)
            
            print(f"Round size: {round_size}")
            round_size_list.append(round_size)
            # sample from top 5 scenes
            left_scene_ids = [r["scene_id"] for r in ranking if r["scene_id"] not in selected_scene_ids]

            if len(left_scene_ids) == 0:
                print("No more scenes to sample from.")
                break
            elif len(left_scene_ids) > 5:
                round_selected_scene_ids = left_scene_ids[:10]
            round_selected_frames = doSample(tree_pre, scene_dict, ranking, round_size, round_selected_scene_ids, all_signatures)
            selected_frames.extend(round_selected_frames)
            
            selected_scene_ids.update(round_selected_scene_ids)

            # prepare next round sampling
            # 1. Add sample to pretrain set
            # X_pretrain_new, ref_ids  = build_signatures(round_selected_frames, P=P, Q=Q)
            X_pretrain_new, ref_ids  = get_signature_by_ids(round_selected_frames, all_signatures)
            X_pretrain = np.concatenate([X_pretrain, X_pretrain_new], axis=0)

            # 2. Remove sample from scene_dict
            # 3. Update total budget
            total_budget -= round_size
            # 4. update tree
        else:
            print("Budget exhausted.")
            break
    
    if len(selected_frames) < total_budget:
        budget = total_budget - len(selected_frames)
        frame_left = all_idx - selected_frames - pretrain_idx
        random.seed(25)
        random_numbers = random.sample(range(0, len(frame_left)-1), budget)
        random_select_frame = [frame_left[i] for i in random_numbers]
        selected_frames.extend(random_select_frame)
    
    print("round sizes: ", round_size_list)
    output_file_name = 'waymo_output/scene_frame_selection.txt'
    with open(output_file_name, 'w') as f: 
        json.dump(selected_frames, f)
    
def doSample(tree_pre, scene_dict, ranking, budget, selected_scene_ids, all_signatures):
    # scene_ids = [r["scene_id"] for r in ranking]
    # get the scene_ids from scene_dict
    selected_scenes = {i: scene_dict[i] for i in selected_scene_ids}
    # print(selected_scenes.keys())
    budget_per_scene = budget // len(selected_scenes)
    if budget_per_scene == 0:
        budget_per_scene = 1

    selected_frames = []
    for scene_id, frames in selected_scenes.items():
        # X_ref, ref_ids   = build_signatures(frames, 64 ,64)
        X_ref, ref_ids   = get_signature_by_ids(frames, all_signatures)
        dists, idxs = tree_pre.query(X_ref, k=1)

        paired = list(zip(
                dists.ravel().tolist(),
                frames,
                # result_dict['neighbor_list']
            ))
        paired_sorted = sorted(
                paired,
                key=lambda x: x[0],
                # key=lambda x: (x[2] in prune_idx, -x[0]),
                reverse=True
                )
        selected = paired_sorted[:budget_per_scene]
        for dist, frame in selected:
            # print("len of frames: ", len(frames))
            # print("selected idx: ", frame)
            selected_frames.append(frame)
    print(f"Selected frames: {selected_frames}")
    
    return selected_frames

        # score_list = dists.ravel().tolist()
        # # print(f"Score list: {score_list}")
        # # print(idxs.ravel().tolist())

        # output_dict = {
        #     'id_list': all_idx_list,
        #     'score_list': score_list,
        #     'neighbor_list': idxs.ravel().tolist()
        # }



def getSceneHeatmapFromFile():
    # dir_path = '/data/nuscenes_scene_heatmap'
    dir_path = '/data/waymo_scene_heatmap'
    # files = sorted(
    #     glob.glob(os.path.join(dir_path, "*.txt")),
    #     key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
    # )
    files = os.listdir(dir_path)

    merged = {}
    for fp in files:
        fp = os.path.join(dir_path, fp)
        with open(fp, "r") as f:
            batch = json.load(f)  # {scene: heatmap_list}
        for scene, heatmap_list in batch.items():
            # If a scene appears twice, the later file wins; change policy if you prefer.
            merged[scene] = heatmap_list

    return merged


def get_ranking(d_pre, d_scene, scene_ids):
    w_pre, w_scene = 0.7, 0.3
    # normalize and score
    d_pre_n = _normalize01(d_pre)
    d_scene_n = _normalize01(d_scene)
    scores = w_pre * d_pre_n + w_scene * (1.0 - d_scene_n)  # larger is better

    # Return a sorted ranking table (desc by score)
    order = np.argsort(-scores)
    ranking = [
        {
            "scene_id": scene_ids[i],
            "score": float(scores[i]),
            "d_pre": float(d_pre[i]),
            "d_scene_knn_mean": float(d_scene[i]),
        }
        for i in order
    ]
    scene_ids = [r["scene_id"] for r in ranking]
    d_pre = np.array([r["d_pre"] for r in ranking], dtype=np.float64)
    d_scene = np.array([r["d_scene_knn_mean"] for r in ranking], dtype=np.float64)

    return ranking, scores

    # Optional caps = number of frames available per scene
    # available = {sid: len(scene_dict.get(sid, [])) for sid in scene_ids}

    # # Allocate a total budget B across scenes
    # B = 10000
    # per_scene_budget = allocate_budget_from_distances(
    #     scene_ids, d_pre, d_scene, B,
    #     w_pre=0.7, w_scene=0.3,
    #     alpha=1.0,               # try 1.2 to emphasize top scenes
    #     temperature=None,        # or set e.g. 0.5 to make distribution peakier
    #     caps=available,          # don’t allocate more than available frames
    #     mins=None,               # e.g., {"scene-001": 5}
    #     available=available      # double-safety: cap by availability
    # )

    # print(per_scene_budget)  # {'scene-001': 17, 'scene-002': 9, ...}

    # output_file_name = 'scene_budget.txt'
    # with open(output_file_name, 'w') as f: 
    #     json.dump(per_scene_budget, f)


    # return ranking

    # sig  = build_sw_signature(Hmap, P=32, Q=32, class_weights=None, seed=123)
    # tree = BallTree(X_ref, metric='manhattan', leaf_size=40)
    # dists, idxs = tree.query(X_cand, k=1)


def getSignature():
    file_dir = '/data/waymo_heatmap'

    all_idx = []
    all_items = os.listdir(file_dir)
    for item in all_items:
        if item.endswith('.txt'):
            item_id = item.split('.')[0]
            all_idx.append(item_id)

    X_cand, cand_ids = build_signatures(all_idx, 64 ,64)

    output_dict = dict(zip(cand_ids, X_cand))
    to_save = {str(k): v.tolist() for k, v in output_dict.items()}
    # print(X_cand.shape)
    # print(cand_ids)
    # print(output_dict)
    output_file_name = '/data/waymo_heatmap_signatures.txt'
    with open(output_file_name, 'w') as f: 
        json.dump(to_save, f)
def test():
    # with open('/data/waymo_heatmap_signatures.txt', 'r') as f:
    #     sig_dict = json.load(f)
    #     print(len(sig_dict.keys()))
    # with open('nuscenes_scenes.txt', "r") as f:
    with open('waymo_output/waymo_scenes.txt', "r") as f:
        scene_dict = json.load(f)
    print(len(scene_dict.keys()))


# getSignature()
build_ball_tree()
# test()
# build_ball_tree_with_scene_waymo()
# saveSceneHeatmapToFile()
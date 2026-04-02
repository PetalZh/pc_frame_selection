import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import BallTree
import json, os, time

# --------- Embedding utilities ---------
def flatten_frames(X):
    """
    X: array of shape (N, 128, 128) or (N, 128, 128, C)
    returns: (N, 128*128*C)
    """
    X = np.asarray(X)
    if X.ndim == 3:      # (N, H, W)
        N, H, W = X.shape
        C = 1
    elif X.ndim == 4:    # (N, H, W, C)
        N, H, W, C = X.shape
    else:
        raise ValueError("X must be (N,128,128) or (N,128,128,C)")
    return X.reshape(X.shape[0], H*W*C)

def l2_normalize(Z, eps=1e-8):
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    return Z / np.maximum(norms, eps)

class FrameEmbedder:
    """
    Fit PCA on pretrained frames; reuse to embed candidates consistently.
    """
    def __init__(self, n_components=128, random_state=0):
        self.n_components = n_components
        self.random_state = random_state
        self.pca = None

    def fit(self, X_pretrain):
        Xf = flatten_frames(X_pretrain)
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        Z = self.pca.fit_transform(Xf)       # (Ns, d)
        return l2_normalize(Z)

    def transform(self, X):
        if self.pca is None:
            raise RuntimeError("Call fit(...) first on the pretrained set.")
        Xf = flatten_frames(X)
        Z = self.pca.transform(Xf)
        return l2_normalize(Z)

# --------- Tree builders ---------
def build_seed_tree(Z_seed, metric="euclidean", leaf_size=40):
    """
    Z_seed: (Ns, d) L2-normalized embeddings of the pretrained set.
    """
    # leaf_size = len(Z_seed)
    return BallTree(Z_seed, metric=metric, leaf_size=leaf_size)

def build_per_class_trees(Z_seed, y_seed, metric="euclidean", leaf_size=40):
    """
    y_seed: (Ns,) integer labels (e.g., 0=car,1=ped,2=cyclist)
    returns dict: class_id -> BallTree (may be None if no samples for a class)
    """
    classes = np.unique(y_seed)
    trees = {}
    for c in classes:
        idx = (y_seed == c)
        if np.sum(idx) == 0:
            trees[c] = None
        else:
            trees[c] = BallTree(Z_seed[idx], metric=metric, leaf_size=leaf_size)
    return trees

# --------- Example usage ---------
# 1) Pretrained frames: X_seed in shape (Ns, 128, 128) or (Ns,128,128,C)
#    Optional seed labels: y_seed (Ns,) with integers for classes.
#    Candidate frames: X_cand in same shape.
# Replace these with your real arrays.
start_time = time.perf_counter()

Ns, Nc = 500, 20000

def load_feature(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return np.array(data, dtype=np.float32)


with open('waymo_output/pretrain_file_list.txt', 'r') as f:
# with open('nus_pretrain_file_list.txt', 'r') as f:
    pretrain_idx = json.load(f)
    file_dir = '/data/feature_waymo_train'
    # file_dir = '/data/feature_nuscenes_train'

    all_idx = []
    all_items = os.listdir(file_dir)
    for item in all_items:
        if item.endswith('.txt'):
            item_id = item.split('.')[0]
            all_idx.append(item_id)

for i in range(len(pretrain_idx)):
    pretrain_idx[i] = pretrain_idx[i].replace("samples/LIDAR_TOP/","").replace('.pcd.bin', '')

pretrain_feature_list = []
for idx in pretrain_idx:
    file_name = idx + '.txt'#'.pcd.txt'
    path = os.path.join(file_dir, file_name)
    feature = load_feature(path)
    feature = feature.squeeze()
    pretrain_feature_list.append(feature)

all_feature_list = []
all_idx_list = []
for idx in all_idx:
    if idx not in pretrain_idx:
        all_idx_list.append(idx)
        file_name = idx + '.txt'#'.pcd.txt'
        path = os.path.join(file_dir, file_name)
        feature = load_feature(path)
        feature = feature.squeeze()
        all_feature_list.append(feature)

pretrain_feature_list = np.array(pretrain_feature_list, dtype=np.float32)
print(f"Pretrain feature shape: {pretrain_feature_list.shape}")
all_feature_list = np.array(all_feature_list, dtype=np.float32)

# X_seed  = np.random.rand(Ns, 128, 128).astype(np.float32)       # toy
# y_seed  = np.random.randint(0, 3, size=Ns)                      # 0:car,1:ped,2:cyclist (example)
# X_cand  = np.random.rand(Nc, 128, 128).astype(np.float32)       # toy

# 2) Fit embedder on pretrained set and embed both sets
embedder = FrameEmbedder(n_components=2813)   # 64–256 are solid starting points
Z_seed   = embedder.fit(pretrain_feature_list)              # (Ns, d)
Z_cand   = embedder.transform(all_feature_list)        # (Nc, d)

# 3) Build BallTree over pretrained set (global)
seed_tree = build_seed_tree(Z_seed)

# 4) Compute nearest-to-set distance for ALL candidates in one batched call
#    This is the distance each candidate has to its closest pretrained frame.
d_to_seed, idx_seed = seed_tree.query(Z_cand, k=1)   # shapes: (Nc,1), (Nc,1)
# d_to_seed = d_to_seed.ravel()
score_list = d_to_seed.ravel().tolist()
output_dict = {
    'id_list': all_idx_list,
    'score_list': score_list,
    'neighbor_list': idx_seed.ravel().tolist()
}

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time:.4f} seconds")


# output_file_name = 'nuscenes_feature_distance_tree3.txt'
# with open(output_file_name, 'w') as f: 
#     json.dump(output_dict, f)

# output_file_name = 'waymo_output/waymo_feature_distance.txt'
# with open(output_file_name, 'w') as f: 
#     json.dump(output_dict, f)

# print(f"Distances to closest pretrained frame (first 10): {d_to_seed[:10]}")
# print(f"frame id: {idx_seed[:10]}")
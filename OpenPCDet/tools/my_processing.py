import json, os, pickle
import numpy as np
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from datetime import datetime, timezone

def combine_loss_info():
    folder_path = 'output_nuscenes_loss'
    file_ids = list(range(1000, 123001, 1000))
    file_ids.append(123579)
    id_list = []
    loss_list = []
    for file_id in file_ids:
        path = f'{folder_path}/{file_id}.txt'
        with open(path, 'r') as f:
            loss_dict = json.load(f)
            # paired = list(zip(loss_dict['loss_list'], loss_dict['id_list']))
            id_list.extend(loss_dict['id_list'])
            loss_list.extend(loss_dict['loss_list'])
    output_dict = {
        'id_list': id_list,
        'loss_list': loss_list
    }
    print(f"Total number of IDs: {len(output_dict['id_list'])}")
    print(output_dict['id_list'][:10])
    with open('output_nuscenes_loss/output_loss_nuscenes.txt', 'w') as f:
        json.dump(output_dict, f, indent=4)


def load_feature(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return np.array(data, dtype=np.float32)

def compute_mean_feature_incremental(file_dir, pretrain_idx):
    """Compute mean feature from large files one by one."""
    running_sum = None
    for idx in tqdm(pretrain_idx, desc="Computing mean incrementally"):
        file_name = idx + '.txt'       
        path = os.path.join(file_dir, file_name)
        if os.path.exists(path):
            feature = load_feature(path)
            # print(feature)
        else:
            continue
        if running_sum is None:
            running_sum = np.zeros_like(feature, dtype=np.float64)
        running_sum += feature
    mean_feature = running_sum / len(pretrain_idx)
    return mean_feature.astype(np.float32)

def compute_sorted_distances(file_dir, all_indices, pretrain_idx, mean_feature):
    """Compute distances from mean, sort from farthest to nearest."""
    remaining_indices = list(set(all_indices) - set(pretrain_idx))

    distances = []
    for idx in tqdm(remaining_indices, desc="Computing distances"):
        file_name = idx + '.txt'
        path = os.path.join(file_dir, file_name)
        feature = load_feature(path)
        distance = np.linalg.norm(feature - mean_feature)  # Euclidean distance
        distances.append((idx, distance))

    # Sort by distance in descending order (farthest to nearest)
    sorted_indices = [idx for idx, _ in sorted(distances, key=lambda x: -x[1])]
    return sorted_indices

def getDivergence():

    # path = '/home/xiaoyu/experiments/OpenPCDet/output/kitti_models/centerpoint_car_only/centerpoint_500_pretrain/default/eval/epoch_30/val/default/result.pkl'
    path = '/home/xiaoyu/experiments/OpenPCDet/output/nuscenes_models/centerpoint/centerpoint_0.1_pretrain/default/eval/epoch_1/val/default/result1.pkl'
    path = '/home/xiaoyu/experiments/OpenPCDet/output/waymo_models/centerpoint/centerpoint_pillar_0.1_pretrain/default/eval/epoch_1/val/default/result.pkl'
    start = datetime.now(timezone.utc)
    
    lam_max = 1.0
    lam_var = 1.0
    lam_entropy = 0.5
    with open(path, 'rb') as f:
        frames = pickle.load(f)
        # print(data[0].keys())
        # print(data[100]["name"])
        # print(data[100]["score"])
        id_list = []
        loss_list = []
        for frame in frames:
            objs = np.asarray(frame['name'])
            scores = np.asarray(frame['score'])

            # --- per-class mean uncertainty ---------------------------------
            classes = np.unique(objs)
            U_c = np.array([scores[objs == c].mean() for c in classes])

            # print(U_c)

            # --- category entropy (diverse mix of object types) -------------
            counts = np.array([np.sum(objs == c) for c in classes], dtype=float)
            p = counts / counts.sum()
            Hcat = -(p * np.log(p + 1e-12)).sum()
            # print(Hcat)

            # --- combine ----------------------------------------------------
            D = lam_max * float(U_c.max()) + lam_var * float(U_c.var()) + lam_entropy * float(Hcat)

            # print([lam_max * float(U_c.max()) + lam_var * float(U_c.var()), lam_entropy * float(Hcat)])
            id_list.append(frame['frame_id'])
            loss_list.append(D)

    output_dict = {
        'id_list': id_list,
        'loss_list': loss_list
    }

    print(output_dict['id_list'][:10])
    print(f"Total number of IDs: {len(output_dict['id_list'])}")
    end = datetime.now(timezone.utc)
    elapsed = end - start 
    print("elapsed seconds:", elapsed.total_seconds())
    # with open('output_divergence_nuscenes.txt', 'w') as f:
    #     json.dump(output_dict, f, indent=4)
    # with open('waymo_output/waymo_divergence.txt', 'w') as f:
    #     json.dump(output_dict, f, indent=4)


def getFeatureTopList():
    # file_dir = '/data/feature_kitti_train'
    file_dir = '/data/feature_nuscenes_train'
    with open('nus_pretrain_file_list.txt', 'r') as f:
        pretrain_idx = json.load(f)

        all_idx = []
        all_items = os.listdir(file_dir)
        for item in all_items:
            if item.endswith('.txt'):
                item_id = item.split('.')[0]
                all_idx.append(item_id)
    mean_feature = compute_mean_feature_incremental(file_dir, pretrain_idx)
    sorted_indices = compute_sorted_distances(file_dir, all_idx, pretrain_idx, mean_feature)
    print(f"Top 10 farthest indices from mean feature: {sorted_indices[:10]}")

    file_name = 'nuscenes_feature_distance_0.1.txt'
    with open(file_name, 'w') as f: 
        json.dump(sorted_indices, f)
                    
def getNuscenesGroup():
    nusc = NuScenes(version='v1.0-trainval', dataroot='/data/nuscenes', verbose=True)
    scene_to_lidar_files = {}

    for scene in nusc.scene:
        scene_name = scene["name"]
        sample_token = scene["first_sample_token"]

        files = []
        while sample_token:
            sample = nusc.get("sample", sample_token)
            # LIDAR_TOP keyframe for this sample
            sd_token = sample["data"]["LIDAR_TOP"]
            sd = nusc.get("sample_data", sd_token)
            files.append(sd["filename"])
            sample_token = sample["next"]  # advance to next keyframe sample

        scene_to_lidar_files[scene_name] = files
    print(scene_to_lidar_files.keys())

    # file_name = 'nuscenes_scenes.txt'
    # with open(file_name, 'w') as f: 
    #     json.dump(scene_to_lidar_files, f)

    # Example: print one scene’s files
    # first_scene = nusc.scene[0]["name"]
    # print(first_scene, len(scene_to_lidar_files[first_scene]))
    # print("\n".join(scene_to_lidar_files[first_scene][:5]))

def fedCSBaseline():
    file_dir = '/data/feature_nuscenes_train'
    with open('nuscenes_scenes.txt', "r") as f:
        scene_dict = json.load(f)

    with open('nus_pretrain_file_list.txt', 'r') as f:
        pretrain_idx = json.load(f)

    scene_mean_features = {}
    key_list = list(scene_dict.keys())
    for scene in key_list:
        frames = scene_dict[scene]
        scene_frame_formated = []
        for i in range(len(frames)):#len(frames)
            frames[i] = frames[i].replace("samples/LIDAR_TOP/","").replace('.pcd.bin', '.pcd')
            # print(frames[i])
            path = os.path.join(file_dir, frames[i]+'.txt')
            if not os.path.exists(path):
                continue
            scene_frame_formated.append(frames[i])
        if len(scene_frame_formated) == 0:
            continue
        
        mean_feature = compute_mean_feature_incremental(file_dir, scene_frame_formated)
        scene_mean_features[scene] = mean_feature
    
    # -------------------------------
    # 1. Global mean feature of all scenes
    # -------------------------------
    if len(scene_mean_features) == 0:
        raise ValueError("No scene mean features were computed. Check your data paths.")

    all_scene_means = np.stack(list(scene_mean_features.values()), axis=0)
    global_mean_feature = all_scene_means.mean(axis=0)

    # -------------------------------
    # 2. Rank all frames by |d1 - d2|
    #    d1: distance to scene center
    #    d2: distance to global center
    # -------------------------------
    ranked_frames = []  # list of (score, scene_id, frame_id)

    for scene, scene_center in scene_mean_features.items():
        frames = scene_dict[scene]
        for frame in frames:
            # convert to feature file name in the same way as above
            frame_id = frame.replace("samples/LIDAR_TOP/", "").replace('.pcd.bin', '.pcd')
            feature_path = os.path.join(file_dir, frame_id + '.txt')
            if not os.path.exists(feature_path):
                continue

            # load frame feature
            frame_feature = load_feature(path)

            # ensure numpy arrays
            frame_feature = np.asarray(frame_feature, dtype=float)
            scene_center_arr = np.asarray(scene_center, dtype=float)
            global_center_arr = np.asarray(global_mean_feature, dtype=float)

            # distances
            d1 = np.linalg.norm(frame_feature - scene_center_arr)
            d2 = np.linalg.norm(frame_feature - global_center_arr)

            score = abs(d1 - d2)
            ranked_frames.append((score, scene, frame_id))

    # sort all frames across all scenes by |d1 - d2|
    ranked_frames.sort(key=lambda x: x[0])  # ascending

    # print(ranked_frames[:10])
    score_list = []
    id_list = []
    scene_list = []
    for item in ranked_frames:
        score_list.append(item[0])
        id_list.append(item[2].replace('.pcd', ''))
        scene_list.append(item[1])
    output_dict = {'score_list': score_list, 'id_list': id_list, 'scene_list': scene_list}

    with open('nuscenes_fedcs_ranking.txt', 'w') as f:
        json.dump(output_dict, f)

    return ranked_frames
    



# combine_loss_info()
# getFeatureTopList()
getDivergence()

# getNuscenesGroup()
# fedCSBaseline()
# with open('/data/feature_kitti_train/000000.txt', 'r') as f:
#         pretrain_idx = json.load(f)
#         print(np.array(pretrain_idx).shape)

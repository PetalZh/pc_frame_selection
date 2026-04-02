import copy
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import common_utils
from ..dataset import DatasetTemplate
from pyquaternion import Quaternion
from PIL import Image
import random
import pcdet.constants as constants
import json, os, math
from collections import Counter
from itertools import combinations


class NuScenesDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        root_path = (root_path if root_path is not None else Path(dataset_cfg.DATA_PATH)) / dataset_cfg.VERSION
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.infos = []
        self.camera_config = self.dataset_cfg.get('CAMERA_CONFIG', None)
        if self.camera_config is not None:
            self.use_camera = self.camera_config.get('USE_CAMERA', True)
            self.camera_image_config = self.camera_config.IMAGE
        else:
            self.use_camera = False

        self.include_nuscenes_data(self.mode)
        # if self.training and self.dataset_cfg.get('BALANCED_RESAMPLING', False):
        #     self.infos = self.balanced_infos_resampling(self.infos)
        #     print('After balanced resampling, the number of samples is %d' % len(self.infos))
        #     # print(self.infos[0]['gt_names'])

    def key_from_path(self, path_or_name: str) -> str:
        """
        Returns basename without the .pcd.bin suffix.
        Works for either a full path or an ID that is already just the name.
        """
        base = os.path.basename(path_or_name)
        return base.replace('.bin', '')   
    
    def top_k_loss(self, infos, pre_trained_infos, k):
        with open('output_nuscenes_loss/output_loss_nuscenes.txt', 'r') as f:
            loss_dict = json.load(f)
            paired = list(zip(loss_dict['loss_list'], loss_dict['id_list']))
            paired_sorted = sorted(paired, key=lambda x: x[0], reverse=True)

            # info['lidar_path']
            # info_map = [self.key_from_path(info['lidar_path']) for info in infos]
            pre_trained_set = {self.key_from_path(info['lidar_path']) for info in pre_trained_infos}
            
            # print(pre_trained_set)
            selected_infos  = []

            for _, idx in paired_sorted:
                if idx in pre_trained_set:          # already used → skip
                    continue
                # print(item[1])
                matched_info = next((info for info in infos if idx in info['lidar_path']), None)
                selected_infos.append(matched_info)

                if len(selected_infos) == k:        # done
                    break
            print(f"Selected {len(selected_infos)} items based on loss from pre-trained set")
            return selected_infos

    def top_k_fedcs(self, infos, pre_trained_infos, k):
        with open('output/nuscenes_fedcs_ranking.txt', 'r') as f:
            loss_dict = json.load(f)
            paired = list(zip(loss_dict['score_list'], loss_dict['id_list']))
            paired_sorted = sorted(paired, key=lambda x: x[0], reverse=True)

            # info['lidar_path']
            # info_map = [self.key_from_path(info['lidar_path']) for info in infos]
            pre_trained_set = {self.key_from_path(info['lidar_path']) for info in pre_trained_infos}
            
            # print(pre_trained_set)
            selected_infos  = []

            for _, idx in paired_sorted:
                if idx in pre_trained_set:          # already used → skip
                    continue
                # print(item[1])
                matched_info = next((info for info in infos if idx in info['lidar_path']), None)
                selected_infos.append(matched_info)

                if len(selected_infos) == k:        # done
                    break
            print(f"Selected {len(selected_infos)} items based on loss from pre-trained set")
            return selected_infos
        
    def random_select(self, infos, start_idx, M):
        random.seed(112)
        random_numbers = random.sample(range(start_idx, len(infos)), M)

        selected_infos = [infos[i] for i in random_numbers]
        
        print(f"Randomly selected {len(selected_infos)} items from index {start_idx} to {len(infos)}")

        return selected_infos
    
    def top_k_divergence(self, infos, pre_trained_infos, k):
        with open('output_divergence_nuscenes.txt', 'r') as f:
            loss_dict = json.load(f)
            paired = list(zip(loss_dict['loss_list'], loss_dict['id_list']))
            paired_sorted = sorted(paired, key=lambda x: x[0], reverse=True)
            pre_trained_set = {self.key_from_path(info['lidar_path']) for info in pre_trained_infos}
            
            selected_infos  = []

            for _, idx in paired_sorted:
                if idx in pre_trained_set:          # already used → skip
                    continue
                matched_info = next((info for info in infos if idx in info['lidar_path']), None)
                selected_infos.append(matched_info)

                if len(selected_infos) == k:        # done
                    break
            print(f"Selected {len(selected_infos)} items based on loss from pre-trained set")
            return selected_infos
    
    
    def top_distance_euclidean(self, infos ,k):
        with open('nuscenes_feature_distance_0.1.txt', 'r') as f:
            sorted_idx = json.load(f)

            info_map = {info['lidar_path']: info for info in infos}
            # print(info_map.keys())

            selected_infos = []
            for idx in sorted_idx:
                idx = f"samples/LIDAR_TOP/{idx}.pcd.bin"
                selected_infos.append(info_map[idx])
                if len(selected_infos) == k:        # done
                    break
            return selected_infos

    def getEmptyInfoId(self, pretrain_infos, thresh):
        with open('nus_pretrain_file_list.txt', 'r') as f:
            pretrain_idx = json.load(f)
        
        # print(pretrain_idx[:10])
        pretrain_infos_sorted = sorted(
            pretrain_infos,
            key=lambda info: sum(1 for obj in info['gt_names'] if obj != 'ignore'),
            reverse=False
        ) # less object first
        
        sorted_index = []
        for info in pretrain_infos_sorted:
            # id = info['lidar_path'].replace("samples/LIDAR_TOP/","").replace('.pcd.bin', '')
            # print(info['lidar_path'])
            sorted_index.append(pretrain_idx.index(info['lidar_path']))
            
        return sorted_index[:thresh]#2600
    
    def sample_rank_biased(self, topk_frames, m, tau=3000.0, eps=0.02, seed=None):
        """
        topk_frames: list sorted best->worst (len ~ 30000)
        m: number of unique samples
        tau: larger => more randomness (slower decay)
        eps: uniform-mix fraction (keeps tail alive)
        """
        rng = np.random.default_rng(seed)
        n = len(topk_frames)
        m = min(m, n)

        ranks = np.arange(n, dtype=np.float64)  # 0 is best
        w = np.exp(-ranks / tau)                # rank weights (exp decay)

        p = (1.0 - eps) * (w / w.sum()) + eps * (1.0 / n)  # blend with uniform
        idx = rng.choice(n, size=m, replace=False, p=p)
        
        return [topk_frames[i] for i in idx]
    
    def getNumberOfNoSampleScene(self, pretrained_ids, scenes)-> int:
        counter = 0
        for scene_id, frame_ids in scenes.items():
            frame_ids = [fid.replace('.pcd.bin', '').replace("samples/LIDAR_TOP/","") for fid in frame_ids]
            available_frames = [fid for fid in frame_ids if fid not in pretrained_ids]
            # print("scene id: {}, available frames: {}".format(scene_id, len(available_frames)))
            if len(available_frames) == 0:
                counter += 1
        return counter
    
    def loss_and_divergence_scene_even_select(self, infos, pre_trained_infos, method, k):
        with open('nuscenes_scenes.txt', 'r') as f:
            scenes = json.load(f)

        if method == 'loss':
            with open('output_nuscenes_loss/output_loss_nuscenes.txt', 'r') as f:
                loss_dict = json.load(f)
        else:
            with open('output_divergence_nuscenes.txt', 'r') as f:
                loss_dict = json.load(f)
            
        # paired = list(zip(loss_dict['loss_list'], loss_dict['id_list']))
        loss_dict['id_list'] = [fid.replace('.pcd', '') for fid in loss_dict['id_list']]
        frame_to_score = {fid: score for fid, score in zip(loss_dict['id_list'], loss_dict['loss_list'])}
        # print(frame_to_score)
        pretrained_ids = set()
        for info in pre_trained_infos:
            frame_id = info['lidar_path'].replace('.pcd.bin', '')  # extract the frame id from path
            frame_id = frame_id.replace("samples/LIDAR_TOP/","")
            pretrained_ids.add(frame_id)
        
        num_of_non_sample = self.getNumberOfNoSampleScene(pretrained_ids, scenes)
        m = math.ceil(k/(len(scenes) - num_of_non_sample))

        selected_frames = []
        for scene_id, frame_ids in scenes.items():
            frame_ids = [fid.replace('.pcd.bin', '').replace("samples/LIDAR_TOP/","") for fid in frame_ids]
            valid_frames = [
                (fid, frame_to_score[fid]) for fid in frame_ids
                if fid in frame_to_score and fid not in pretrained_ids
            ]
            # print('length of valid frames in scene {}: {}'.format(scene_id, len(valid_frames)))
            # Sort by descending score and pick top k
            # topk_frames = [fid for fid, _ in sorted(valid_frames, key=lambda x: x[1], reverse=True)[:m]]
            # selected_frames.extend(topk_frames)

            topk_frames = [fid for fid, _ in sorted(valid_frames, key=lambda x: x[1], reverse=True)]
            if len(topk_frames) >= m:
                # print(len(valid_frames))
                # print(len(topk_frames))
                # print('m: {}'.format(m))
                picked = self.sample_rank_biased(topk_frames, m, tau=10, eps=0.03, seed=0)
                selected_frames.extend(picked)
            else:
                selected_frames.extend(topk_frames)
        
        if len(selected_frames) > k:
            selected_frames = selected_frames[:k]
            
        selected_infos = [
            info for info in infos
            if any(fid in info['lidar_path'] for fid in selected_frames)
        ]
        return selected_infos
    
    def top_k_with_balltree_by_scene(self, infos, pre_trained_infos, k, metric = 'feature', prune=False, thresh = 0):
        if metric == 'feature':
            file_name = 'nuscenes_feature_distance_tree2.txt'
        else:
            file_name = 'nuscenes_heatmap_tree.txt'
        with open(file_name, 'r') as f:
            result_dict = json.load(f)

        with open('nuscenes_scenes.txt', 'r') as f:
            scenes = json.load(f)
        

        # m = k // len(scenes)

        pretrained_ids = set()
        for info in pre_trained_infos:
            frame_id = info['lidar_path'].replace('.pcd.bin', '')  # extract the frame id from path
            frame_id = frame_id.replace("samples/LIDAR_TOP/","")
            pretrained_ids.add(frame_id)
        # print(pretrained_ids)

        num_of_non_sample = self.getNumberOfNoSampleScene(pretrained_ids, scenes)
        m = math.ceil(k/(len(scenes) - num_of_non_sample))
        
        frame_to_score = {fid: score for fid, score in zip(result_dict['id_list'], result_dict['score_list'])}
        #id only

        selected_frames = []
        for scene_id, frame_ids in scenes.items():
            # Filter out pretrained frames and missing scores
            frame_ids = [fid.replace('.pcd.bin', '').replace("samples/LIDAR_TOP/","") for fid in frame_ids]
            valid_frames = [
                (fid, frame_to_score[fid]) for fid in frame_ids
                if fid in frame_to_score and fid not in pretrained_ids
            ]
            # print(valid_frames)
            # Sort by descending score and pick top k
            # topk_frames = [fid for fid, _ in sorted(valid_frames, key=lambda x: x[1], reverse=True)[:m]]
            # selected_frames.extend(topk_frames)

            topk_frames = [fid for fid, _ in sorted(valid_frames, key=lambda x: x[1], reverse=True)]
            if len(topk_frames) >= m:
                # print(len(valid_frames))
                # print(len(topk_frames))
                # print('m: {}'.format(m))
                picked = self.sample_rank_biased(topk_frames, m, tau=10, eps=0.03, seed=0)
                selected_frames.extend(picked)
            else:
                selected_frames.extend(topk_frames)
        
        selected_infos = [
            info for info in infos
            if any(fid in info['lidar_path'] for fid in selected_frames)
        ]
        return selected_infos

    def top_k_with_balltree(self, infos, pre_trained_infos, k, metric = 'feature', prune=False, thresh = 0):
        #with open('nuscenes_feature_distance_tree2.txt', 'r') as f:
        if metric == 'feature':
            file_name = 'nuscenes_feature_distance_tree2.txt'
        else:
            file_name = 'nuscenes_heatmap_tree.txt'
        with open(file_name, 'r') as f:
            result_dict = json.load(f)
            prune_idx = self.getEmptyInfoId(pre_trained_infos, thresh)

            paired = list(zip(
                result_dict['score_list'],
                result_dict['id_list'],
                result_dict['neighbor_list']
            ))
            
            if prune:
                paired_sorted = sorted(
                paired,
                key=lambda x: (x[2] in prune_idx, -x[0])
                )

                # # Remove items whose neighbor is in prune_idx
                # filtered = [x for x in paired if x[2] not in prune_idx]

                # # Sort only the remaining items by score (descending)
                # paired_sorted = sorted(filtered, key=lambda x: -x[0])
            else:
                paired_sorted = sorted(paired, key=lambda x: x[0], reverse=True)
            # print(paired_sorted[:10])

            print('lenght of heatmap: {}'.format(len(paired_sorted)))

            pre_trained_set = {self.key_from_path(info['lidar_path']) for info in pre_trained_infos}
            selected_infos  = []

            for _, idx ,_ in paired_sorted[1500:]:
                if idx in pre_trained_set:          # already used → skip
                    continue
                # print(item[1])
                matched_info = next((info for info in infos if idx in info['lidar_path']), None)
                selected_infos.append(matched_info)

                if len(selected_infos) == k:        # done
                    break
            print(f"Selected {len(selected_infos)} items based on heatmap_emd from pre-trained set")
            return selected_infos

    def top_k_heatmap_emd(self, infos, pre_trained_infos, k):
        with open('nuscenes_heatmap_EMD_4.txt', 'r') as f:
            loss_dict = json.load(f)
            print('lenght of heatmap: {}'.format(len(loss_dict['score_list'])))
            paired = list(zip(loss_dict['score_list'], loss_dict['id_list']))
            paired_sorted = sorted(paired, key=lambda x: x[0], reverse=False)
            # print(paired_sorted[:10])

            pre_trained_set = {self.key_from_path(info['lidar_path']) for info in pre_trained_infos}
            selected_infos  = []

            for _, idx in paired_sorted:
                if idx in pre_trained_set:          # already used → skip
                    continue
                # print(item[1])
                matched_info = next((info for info in infos if idx in info['lidar_path']), None)
                selected_infos.append(matched_info)

                if len(selected_infos) == k:        # done
                    break
            print(f"Selected {len(selected_infos)} items based on heatmap_emd from pre-trained set")
            return selected_infos
    
    
    def getMoreFrames(self, sample_size, pretrain_set, selected_id_list):
        file_name = 'nuscenes_heatmap_tree.txt'
        with open(file_name, 'r') as f:
            result_dict = json.load(f)
        full_frames = result_dict['id_list']
        candidate_list = []
        for frame in full_frames:
            if frame in pretrain_set or frame in set(selected_id_list):
                continue
            candidate_list.append(frame)

        return random.sample(candidate_list, sample_size)
    
    def top_k_scene_heatmap(self, infos, pre_trained_infos, k):
        with open('scene_frame_selection_2.txt', 'r') as f:
            id_list = json.load(f)
            # print('lenght of heatmap: {}'.format(len(loss_dict['score_list'])))
            # paired = list(zip(loss_dict['score_list'], loss_dict['id_list']))
            # paired_sorted = sorted(paired, key=lambda x: x[0], reverse=False)
            # print(paired_sorted[:10])
            print("length of id list: {}".format(len(id_list)))
            #pool_size = k + 3000
            pool_size = k + 4000#+ 1000
            if pool_size > len(id_list):
                pool_size = len(id_list)
            pre_trained_set = {self.key_from_path(info['lidar_path']) for info in pre_trained_infos}
            selected_infos  = []
            selected_infos_id = []
            for idx in id_list:
                if idx in pre_trained_set:          # already used → skip
                    continue
                matched_info = next((info for info in infos if idx in info['lidar_path']), None)
                selected_infos.append(matched_info)
                selected_infos_id.append(matched_info['lidar_path'])

                # if len(selected_infos) == k:        # done
                #     break
            random.seed(10) #10
            # random_numbers = random.sample(range(0, len(selected_infos)-6000), k)
            print('pool size: {}'.format(pool_size))
            print('k: {}'.format(k))
            random_numbers = random.sample(range(0, pool_size), k)

            selected_infos = [selected_infos[i] for i in random_numbers]
            # print(f"Selected {len(selected_infos)} items based on heatmap_emd from pre-trained set")

            if len(selected_infos) < k:
                sample_size = k - len(selected_infos)
                more_frames = self.getMoreFrames(sample_size, pre_trained_set, selected_infos_id)
                
                for idx in  more_frames:
                    matched_info = next((info for info in infos if idx in info['lidar_path']), None)
                    selected_infos.append(matched_info)
                print('more frames {} added'.format(len(more_frames)))

            return selected_infos
    def uniform_select(self, frames, m):
        n = len(frames)
        if m >= n:
            return frames
        if m == 1:
            return [frames[n // 2]]

        # evenly spaced indices from 0 .. n-1 (inclusive)
        idx = [round(i * (n - 1) / (m - 1)) for i in range(m)]
        # guarantee uniqueness in rare rounding collisions
        used = set()
        out = []
        for j in idx:
            jj = j
            while jj in used and jj + 1 < n:
                jj += 1
            while jj in used and jj - 1 >= 0:
                jj -= 1
            used.add(jj)
            out.append(frames[jj])
        return out
    

    def random_scene_even_select(self, infos, pre_trained_infos, method, k):
        with open('nuscenes_scenes.txt', 'r') as f:
            scenes = json.load(f)
        
        pretrained_ids = set()
        for info in pre_trained_infos:
            frame_id = info['lidar_path']  # extract the frame id from path
            pretrained_ids.add(frame_id)
        print('k:', k)
        num_of_non_sample = self.getNumberOfNoSampleScene(pretrained_ids, scenes)
        m = math.ceil(k/(len(scenes) - num_of_non_sample))

        selected_frames = []
        for scene_id, frame_ids in scenes.items():
            # Filter out pretrained frames and missing scores
            frame_ids = [fid.replace('.pcd.bin', '').replace("samples/LIDAR_TOP/","") for fid in frame_ids]
            available_frames = [fid for fid in frame_ids if fid not in pretrained_ids]
            # print("scene id: {}, available frames: {}".format(scene_id, len(available_frames)))
            if len(available_frames) < m:
                selected_frames.extend(available_frames)
                continue
            if method == 'random':
                random.seed(112)
                sampled = random.sample(available_frames, m)
                selected_frames.extend(sampled)
            else:
                sampled = self.uniform_select(available_frames, m)
                selected_frames.extend(sampled)
        
        if len(selected_frames) > k:
            selected_frames = selected_frames[:k]
            
        selected_infos = [
            info for info in infos
            if any(fid in info['lidar_path'] for fid in selected_frames)
        ]
        return selected_infos
        
    def getSampledData(self, infos, sample_rate, sample_method):
        # print("Total number of samples before sampling: {}".format(len(infos)))
        # sample_rate = constants.sample_rate
        # sample_method = constants.sample_method
        
        pretrain_size = int(len(infos)* 0.1)
        selected_infos = infos[:pretrain_size]

        sample_size = int(len(infos) * sample_rate)

        if sample_method == 'random':
            selected_infos=self.random_select(infos, pretrain_size, sample_size)
        elif sample_method == 'loss':
            selected_infos=self.top_k_loss(infos, infos[:pretrain_size], sample_size)
        elif sample_method == 'divergence':
            selected_infos=self.top_k_divergence(infos, infos[:pretrain_size], sample_size)
        elif sample_method == 'heatmap_emd':
            selected_infos=self.top_k_heatmap_emd(infos, infos[:pretrain_size], sample_size)
        elif sample_method == 'euclidean':
            selected_infos=self.top_distance_euclidean(infos, sample_size)
        elif sample_method == 'euclidean_tree':
            selected_infos=self.top_k_with_balltree(infos, infos[:pretrain_size], sample_size, metric ='feature', prune=False)
        elif sample_method == 'euclidean_tree_prune':
            selected_infos=self.top_k_with_balltree(infos, infos[:pretrain_size], sample_size, metric ='feature', prune=True, thresh=300)
        elif sample_method == 'heatmap_emd_tree':
            selected_infos=self.top_k_with_balltree(infos, infos[:pretrain_size], sample_size, metric ='heatmap_emd', prune=False)
        elif sample_method == 'heatmap_emd_tree_prune':
            selected_infos=self.top_k_with_balltree(infos, infos[:pretrain_size], sample_size, metric ='heatmap_emd', prune=True, thresh=2600)
        elif sample_method == 'heatmap_scene_tree':
            selected_infos=self.top_k_scene_heatmap(infos, infos[:pretrain_size], sample_size)
        elif sample_method == 'scene_even_heatmap':
            selected_infos = self.top_k_with_balltree_by_scene(infos, infos[:pretrain_size], sample_size, metric='heatmap_emd', prune=False)
        elif sample_method == 'scene_even_euclidean':
            selected_infos = self.top_k_with_balltree_by_scene(infos, infos[:pretrain_size], sample_size, metric='feature', prune=False)
        elif sample_method == 'random_scene_even':
            selected_infos = self.random_scene_even_select(infos, infos[:pretrain_size], 'random', sample_size)
        elif sample_method == 'uniform_scene_even':
            selected_infos = self.random_scene_even_select(infos, infos[:pretrain_size], 'uniform', sample_size)
        elif sample_method == 'loss_scene_even':
            selected_infos = self.loss_and_divergence_scene_even_select(infos, infos[:pretrain_size], 'loss',sample_size)
        elif sample_method == 'divergence_scene_even':
            selected_infos = self.loss_and_divergence_scene_even_select(infos, infos[:pretrain_size], 'divergence',sample_size)
        elif sample_method == 'baseline_fedcs':
            selected_infos = self.top_k_fedcs(infos, infos[:pretrain_size], sample_size)
        elif sample_method == 'full':
            selected_infos = infos
        return selected_infos
    
    def getStaAll(self, infos):
        sample_rate_list = [0.6, 0.7] #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        sample_method_list = ['random', 'loss_scene_even', 'divergence_scene_even', 'scene_even_euclidean', 'heatmap_scene_tree', 'scene_even_heatmap']
        sample_method_list = ['loss_scene_even', 'divergence_scene_even', 'scene_even_euclidean', 'scene_even_heatmap']
        sample_method_list = ['heatmap_scene_tree'] #heatmap_emd_tree_prune
        sample_method_list = ['random', 'random_scene_even', 'loss_scene_even', 'divergence_scene_even', 'scene_even_euclidean', 'heatmap_emd_tree', 'scene_even_heatmap']
        sample_method_list = ['euclidean_tree']
        output_dict = {}
        for rate in sample_rate_list:
            for method in sample_method_list:
                print("Sample method: {}, sample rate: {}".format(method, rate))
                sampled_data = self.getSampledData(infos, rate, method)
                print('Sampled data size: {}'.format(len(sampled_data)))
                output = self.getSta(sampled_data)
                key = "{} {}".format(method, rate)
                output_dict[key] = output

        # output_file_name = 'output/sta.txt'
        # with open(output_file_name, 'w') as f: 
        #     json.dump(output_dict, f)
    def getStaPerFrame(self, infos):
        objects = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
        output_dict = {}
        count_dict = {obj: 0 for obj in objects}
        for info in infos:
            for name in info['gt_names']:
                if name in objects:
                    count_dict[name] += 1
            output_dict[info['lidar_path']] = copy.deepcopy(count_dict)
            count_dict = {obj: 0 for obj in objects}
        
        output_file_name = 'output/sta_per_frame.txt'
        with open(output_file_name, 'w') as f: 
            json.dump(output_dict, f)
    
    def getStaIDList(self, infos):
        sample_rate_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        sample_method_list = ['random', 'loss_scene_even', 'divergence_scene_even', 'scene_even_euclidean', 'heatmap_scene_tree', 'scene_even_heatmap']
        output_dict = {}
        for rate in sample_rate_list:
            for method in sample_method_list:
                print("Sample method: {}, sample rate: {}".format(method, rate))
                id_list = []
                sampled_data = self.getSampledData(infos, rate, method)
                for info in sampled_data:
                    id_list.append(info['lidar_path'])
                key = "{}_{}".format(method, rate)
                output_dict[key] = id_list
                
                output_file_name = 'output/select_frame_id_list.txt'
                with open(output_file_name, 'w') as f: 
                    json.dump(output_dict, f)
    
    # def get_cooccurrence_stats(self, infos,
    #                        pretrain_list_path='nus_pretrain_file_list.txt',
    #                        use_pretrain_filter=False):
    #     objects = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']

    #     obj_to_idx = {o: i for i, o in enumerate(objects)}
    #     n = len(objects)

    #     # Optional: filter frames by a pretrain list
    #     pretrain_idx = None
    #     if use_pretrain_filter:
    #         with open(pretrain_list_path, "r") as f:
    #             pretrain_idx = set(json.load(f))

    #     # Frame-level presence count (how many frames contain each object)
    #     frame_count = {obj: 0 for obj in objects}

    #     # Frame-level co-occurrence: co_mat[i, j] = #frames containing both i and j
    #     co_mat = np.zeros((n, n), dtype=np.int64)

    #     # Also keep a dict for pairs if you want easy printing
    #     pair_count = {(a, b): 0 for a, b in combinations(objects, 2)}

    #     for info in infos:
    #         if use_pretrain_filter and info.get('lidar_path') not in pretrain_idx:
    #             continue

    #         # presence set for this frame
    #         present = {name for name in info.get('gt_names', []) if name in obj_to_idx}
    #         # print(info.get('gt_names', []))
    #         # print(obj_to_idx)

    #         # update per-object frame counts + diagonal
    #         for name in present:
    #             frame_count[name] += 1
    #             i = obj_to_idx[name]
    #             co_mat[i, i] += 1

    #         # update per-pair co-occurrence by +1 per frame
    #         present_sorted = sorted(present, key=lambda x: obj_to_idx[x])
    #         # print(pair_count.keys())
    #         for a, b in combinations(present_sorted, 2):
    #             pair_count[(a, b)] += 1
    #             i, j = obj_to_idx[a], obj_to_idx[b]
    #             co_mat[i, j] += 1
    #             co_mat[j, i] += 1  # symmetric

    #     for (a, b), c in sorted(pair_count.items(), key=lambda x: x[1], reverse=True):
    #         print(f"({a}, {b}): {c}")

    #     return frame_count, pair_count, objects, co_mat
        
    def get_cooccurrence_stats(self, infos,
                           pretrain_list_path='nus_pretrain_file_list.txt',
                           use_pretrain_filter=False):
        objects = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']

        obj_to_idx = {o: i for i, o in enumerate(objects)}
        n = len(objects)

        pretrain_idx = None
        if use_pretrain_filter:
            with open(pretrain_list_path, "r") as f:
                pretrain_idx = set(json.load(f))

        # how many frames contain each class (presence)
        frame_count = {obj: 0 for obj in objects}

        # 2-way matrix (optional, keep if you want)
        co_mat = np.zeros((n, n), dtype=np.int64)

        # 2-way counts for all pairs (includes zeros)
        pair_count = {(a, b): 0 for a, b in combinations(objects, 2)}

        # 3-way counts for all triples (includes zeros)
        triple_count = {(a, b, c): 0 for a, b, c in combinations(objects, 3)}

        for info in infos:
            if use_pretrain_filter and info.get('lidar_path') not in pretrain_idx:
                continue

            # set of classes present in this frame (binary presence)
            present = {name for name in info.get('gt_names', []) if name in obj_to_idx}

            # IMPORTANT: sort by your fixed objects order to keep tuple keys consistent
            present_sorted = sorted(present, key=lambda x: obj_to_idx[x])

            # per-class frame count + diagonal
            for name in present_sorted:
                frame_count[name] += 1
                i = obj_to_idx[name]
                co_mat[i, i] += 1

            # 2-way co-occurrence (+1 per frame)
            for a, b in combinations(present_sorted, 2):
                pair_count[(a, b)] += 1
                i, j = obj_to_idx[a], obj_to_idx[b]
                co_mat[i, j] += 1
                co_mat[j, i] += 1

            # 3-way co-occurrence (+1 per frame)
            for a, b, c in combinations(present_sorted, 3):
                triple_count[(a, b, c)] += 1

        # Print all 2-way combinations
        print("=== Pair co-occurrence (2-way) ===")
        for (a, b), cnt in sorted(pair_count.items(), key=lambda x: x[1], reverse=True):
            print(f"({a}, {b}): {cnt}")

        # Print all 3-way combinations
        print("=== Triple co-occurrence (3-way) ===")
        for (a, b, c), cnt in sorted(triple_count.items(), key=lambda x: x[1], reverse=True):
            print(f"({a}, {b}, {c}): {cnt}")

        return frame_count, pair_count, triple_count, objects, co_mat

    def getSta(self, infos):
        objects = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
        count_dict = {obj: 0 for obj in objects}
        with open('nus_pretrain_file_list.txt', 'r') as f:
            pretrain_idx = json.load(f)

        for info in infos:
            # if info['lidar_path'] in pretrain_idx:
                # print(info['gt_names'])
                for name in info['gt_names']:
                    if name in objects:
                        count_dict[name] += 1
        print(count_dict)
        return count_dict
    
    def include_nuscenes_data(self, mode):
        self.logger.info('Loading NuScenes dataset')
        nuscenes_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                
                # nuscenes_infos.extend(infos)
                for info in infos:
                    if 'n015-2018-09-27-15-33-17+0800__LIDAR_TOP__1538033870447840' in info['lidar_path']:
                        print(info['gt_names'])
                        break
                # print(infos[0]['gt_names'])

                if self.training:
                    sample_rate = constants.sample_rate
                    sample_method = constants.sample_method
                    nuscenes_infos = self.getSampledData(infos, sample_rate, sample_method)
                    # self.getStaIDList(infos)
                    # self.getStaPerFrame(infos)
                    self.getStaAll(infos)
                    # self.get_cooccurrence_stats(infos)
                    # for info in nuscenes_infos:
                    #     print(info.keys())
                    #     break

                else:
                    nuscenes_infos.extend(infos)
                
                
                # nuscenes_infos.extend(infos)
        # for info in infos:
        #     print("lidar path: {}".format(info['lidar_path']))
        #     print("sweep: {}".format(info['sweeps']))
        #     for item in info['sweeps']:
        #         print("sweep lidar path: {}".format(item['sample_data_token']))
        #     break
        
        # sample_length = int(len(nuscenes_infos) * 0.2)
        # self.infos.extend(nuscenes_infos[:sample_length])
        self.infos.extend(nuscenes_infos)
        self.getSta(self.infos)
        self.logger.info('Total samples for NuScenes dataset: %d' % (len(nuscenes_infos)))

    def balanced_infos_resampling(self, infos):
        """
        Class-balanced sampling of nuScenes dataset from https://arxiv.org/abs/1908.09492
        """
        if self.class_names is None:
            return infos

        cls_infos = {name: [] for name in self.class_names}
        for info in infos:
            for name in set(info['gt_names']):
                if name in self.class_names:
                    cls_infos[name].append(info)

        duplicated_samples = sum([len(v) for _, v in cls_infos.items()])
        cls_dist = {k: len(v) / duplicated_samples for k, v in cls_infos.items()}

        sampled_infos = []

        frac = 1.0 / len(self.class_names)
        ratios = [frac / v for v in cls_dist.values()]

        for cur_cls_infos, ratio in zip(list(cls_infos.values()), ratios):
            sampled_infos += np.random.choice(
                cur_cls_infos, int(len(cur_cls_infos) * ratio)
            ).tolist()
        self.logger.info('Total samples after balanced resampling: %s' % (len(sampled_infos)))

        cls_infos_new = {name: [] for name in self.class_names}
        for info in sampled_infos:
            for name in set(info['gt_names']):
                if name in self.class_names:
                    cls_infos_new[name].append(info)

        cls_dist_new = {k: len(v) / len(sampled_infos) for k, v in cls_infos_new.items()}

        return sampled_infos

    def get_sweep(self, sweep_info):
        def remove_ego_points(points, center_radius=1.0):
            mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
            return points[mask]

        lidar_path = self.root_path / sweep_info['lidar_path']
        points_sweep = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
        points_sweep = remove_ego_points(points_sweep).T
        if sweep_info['transform_matrix'] is not None:
            num_points = points_sweep.shape[1]
            points_sweep[:3, :] = sweep_info['transform_matrix'].dot(
                np.vstack((points_sweep[:3, :], np.ones(num_points))))[:3, :]

        cur_times = sweep_info['time_lag'] * np.ones((1, points_sweep.shape[1]))
        return points_sweep.T, cur_times.T

    def get_lidar_with_sweeps(self, index, max_sweeps=1):
        info = self.infos[index]
        lidar_path = self.root_path / info['lidar_path']
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]

        sweep_points_list = [points]
        sweep_times_list = [np.zeros((points.shape[0], 1))]

        for k in np.random.choice(len(info['sweeps']), max_sweeps - 1, replace=False):
            points_sweep, times_sweep = self.get_sweep(info['sweeps'][k])
            sweep_points_list.append(points_sweep)
            sweep_times_list.append(times_sweep)

        points = np.concatenate(sweep_points_list, axis=0)
        times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

        points = np.concatenate((points, times), axis=1)
        return points

    def crop_image(self, input_dict):
        W, H = input_dict["ori_shape"]
        imgs = input_dict["camera_imgs"]
        img_process_infos = []
        crop_images = []
        for img in imgs:
            if self.training == True:
                fH, fW = self.camera_image_config.FINAL_DIM
                resize_lim = self.camera_image_config.RESIZE_LIM_TRAIN
                resize = np.random.uniform(*resize_lim)
                resize_dims = (int(W * resize), int(H * resize))
                newW, newH = resize_dims
                crop_h = newH - fH
                crop_w = int(np.random.uniform(0, max(0, newW - fW)))
                crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            else:
                fH, fW = self.camera_image_config.FINAL_DIM
                resize_lim = self.camera_image_config.RESIZE_LIM_TEST
                resize = np.mean(resize_lim)
                resize_dims = (int(W * resize), int(H * resize))
                newW, newH = resize_dims
                crop_h = newH - fH
                crop_w = int(max(0, newW - fW) / 2)
                crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            
            # reisze and crop image
            img = img.resize(resize_dims)
            img = img.crop(crop)
            crop_images.append(img)
            img_process_infos.append([resize, crop, False, 0])
        
        input_dict['img_process_infos'] = img_process_infos
        input_dict['camera_imgs'] = crop_images
        return input_dict
    
    def load_camera_info(self, input_dict, info):
        input_dict["image_paths"] = []
        input_dict["lidar2camera"] = []
        input_dict["lidar2image"] = []
        input_dict["camera2ego"] = []
        input_dict["camera_intrinsics"] = []
        input_dict["camera2lidar"] = []

        for _, camera_info in info["cams"].items():
            input_dict["image_paths"].append(camera_info["data_path"])

            # lidar to camera transform
            lidar2camera_r = np.linalg.inv(camera_info["sensor2lidar_rotation"])
            lidar2camera_t = (
                camera_info["sensor2lidar_translation"] @ lidar2camera_r.T
            )
            lidar2camera_rt = np.eye(4).astype(np.float32)
            lidar2camera_rt[:3, :3] = lidar2camera_r.T
            lidar2camera_rt[3, :3] = -lidar2camera_t
            input_dict["lidar2camera"].append(lidar2camera_rt.T)

            # camera intrinsics
            camera_intrinsics = np.eye(4).astype(np.float32)
            camera_intrinsics[:3, :3] = camera_info["camera_intrinsics"]
            input_dict["camera_intrinsics"].append(camera_intrinsics)

            # lidar to image transform
            lidar2image = camera_intrinsics @ lidar2camera_rt.T
            input_dict["lidar2image"].append(lidar2image)

            # camera to ego transform
            camera2ego = np.eye(4).astype(np.float32)
            camera2ego[:3, :3] = Quaternion(
                camera_info["sensor2ego_rotation"]
            ).rotation_matrix
            camera2ego[:3, 3] = camera_info["sensor2ego_translation"]
            input_dict["camera2ego"].append(camera2ego)

            # camera to lidar transform
            camera2lidar = np.eye(4).astype(np.float32)
            camera2lidar[:3, :3] = camera_info["sensor2lidar_rotation"]
            camera2lidar[:3, 3] = camera_info["sensor2lidar_translation"]
            input_dict["camera2lidar"].append(camera2lidar)
        # read image
        filename = input_dict["image_paths"]
        images = []
        for name in filename:
            images.append(Image.open(str(self.root_path / name)))
        
        input_dict["camera_imgs"] = images
        input_dict["ori_shape"] = images[0].size
        
        # resize and crop image
        input_dict = self.crop_image(input_dict)

        return input_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs

        return len(self.infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])
        points = self.get_lidar_with_sweeps(index, max_sweeps=self.dataset_cfg.MAX_SWEEPS)

        input_dict = {
            'points': points,
            'frame_id': Path(info['lidar_path']).stem,
            'metadata': {'token': info['token']}
        }

        if 'gt_boxes' in info:
            if self.dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
                mask = (info['num_lidar_pts'] > self.dataset_cfg.FILTER_MIN_POINTS_IN_GT - 1)
            else:
                mask = None

            input_dict.update({
                'gt_names': info['gt_names'] if mask is None else info['gt_names'][mask],
                'gt_boxes': info['gt_boxes'] if mask is None else info['gt_boxes'][mask]
            })
        if self.use_camera:
            input_dict = self.load_camera_info(input_dict, info)

        data_dict = self.prepare_data(data_dict=input_dict)

        if self.dataset_cfg.get('SET_NAN_VELOCITY_TO_ZEROS', False) and 'gt_boxes' in info:
            gt_boxes = data_dict['gt_boxes']
            gt_boxes[np.isnan(gt_boxes)] = 0
            data_dict['gt_boxes'] = gt_boxes

        if not self.dataset_cfg.PRED_VELOCITY and 'gt_boxes' in data_dict:
            data_dict['gt_boxes'] = data_dict['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6, -1]]

        return data_dict

    def evaluation(self, det_annos, class_names, **kwargs):
        import json
        from nuscenes.nuscenes import NuScenes
        from . import nuscenes_utils
        nusc = NuScenes(version=self.dataset_cfg.VERSION, dataroot=str(self.root_path), verbose=True)
        nusc_annos = nuscenes_utils.transform_det_annos_to_nusc_annos(det_annos, nusc)
        nusc_annos['meta'] = {
            'use_camera': False,
            'use_lidar': True,
            'use_radar': False,
            'use_map': False,
            'use_external': False,
        }

        output_path = Path(kwargs['output_path'])
        output_path.mkdir(exist_ok=True, parents=True)
        res_path = str(output_path / 'results_nusc.json')
        with open(res_path, 'w') as f:
            json.dump(nusc_annos, f)

        self.logger.info(f'The predictions of NuScenes have been saved to {res_path}')

        if self.dataset_cfg.VERSION == 'v1.0-test':
            return 'No ground-truth annotations for evaluation', {}

        from nuscenes.eval.detection.config import config_factory
        from nuscenes.eval.detection.evaluate import NuScenesEval

        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
            'v1.0-test': 'test'
        }
        try:
            eval_version = 'detection_cvpr_2019'
            eval_config = config_factory(eval_version)
        except:
            eval_version = 'cvpr_2019'
            eval_config = config_factory(eval_version)

        nusc_eval = NuScenesEval(
            nusc,
            config=eval_config,
            result_path=res_path,
            eval_set=eval_set_map[self.dataset_cfg.VERSION],
            output_dir=str(output_path),
            verbose=True,
        )
        metrics_summary = nusc_eval.main(plot_examples=0, render_curves=False)

        with open(output_path / 'metrics_summary.json', 'r') as f:
            metrics = json.load(f)

        result_str, result_dict = nuscenes_utils.format_nuscene_results(metrics, self.class_names, version=eval_version)
        return result_str, result_dict

    def create_groundtruth_database(self, used_classes=None, max_sweeps=10):
        import torch

        database_save_path = self.root_path / f'gt_database_{max_sweeps}sweeps_withvelo'
        db_info_save_path = self.root_path / f'nuscenes_dbinfos_{max_sweeps}sweeps_withvelo.pkl'

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        for idx in tqdm(range(len(self.infos))):
            sample_idx = idx
            info = self.infos[idx]
            points = self.get_lidar_with_sweeps(idx, max_sweeps=max_sweeps)
            gt_boxes = info['gt_boxes']
            gt_names = info['gt_names']

            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
            ).long().squeeze(dim=0).cpu().numpy()

            for i in range(gt_boxes.shape[0]):
                filename = '%s_%s_%d.bin' % (sample_idx, gt_names[i], i)
                filepath = database_save_path / filename
                gt_points = points[box_idxs_of_pts == i]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or gt_names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': gt_names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}
                    if gt_names[i] in all_db_infos:
                        all_db_infos[gt_names[i]].append(db_info)
                    else:
                        all_db_infos[gt_names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)


def create_nuscenes_info(version, data_path, save_path, max_sweeps=10, with_cam=False):
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils import splits
    from . import nuscenes_utils
    data_path = data_path / version
    save_path = save_path / version

    assert version in ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise NotImplementedError

    nusc = NuScenes(version=version, dataroot=data_path, verbose=True)
    available_scenes = nuscenes_utils.get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in train_scenes])
    val_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in val_scenes])

    print('%s: train scene(%d), val scene(%d)' % (version, len(train_scenes), len(val_scenes)))

    train_nusc_infos, val_nusc_infos = nuscenes_utils.fill_trainval_infos(
        data_path=data_path, nusc=nusc, train_scenes=train_scenes, val_scenes=val_scenes,
        test='test' in version, max_sweeps=max_sweeps, with_cam=with_cam
    )

    if version == 'v1.0-test':
        print('test sample: %d' % len(train_nusc_infos))
        with open(save_path / f'nuscenes_infos_{max_sweeps}sweeps_test.pkl', 'wb') as f:
            pickle.dump(train_nusc_infos, f)
    else:
        print('train sample: %d, val sample: %d' % (len(train_nusc_infos), len(val_nusc_infos)))
        with open(save_path / f'nuscenes_infos_{max_sweeps}sweeps_train.pkl', 'wb') as f:
            pickle.dump(train_nusc_infos, f)
        with open(save_path / f'nuscenes_infos_{max_sweeps}sweeps_val.pkl', 'wb') as f:
            pickle.dump(val_nusc_infos, f)


if __name__ == '__main__':
    import yaml
    import argparse
    from pathlib import Path
    from easydict import EasyDict

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--func', type=str, default='create_nuscenes_infos', help='')
    parser.add_argument('--version', type=str, default='v1.0-trainval', help='')
    parser.add_argument('--with_cam', action='store_true', default=False, help='use camera or not')
    args = parser.parse_args()

    if args.func == 'create_nuscenes_infos':
        dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        dataset_cfg.VERSION = args.version
        create_nuscenes_info(
            version=dataset_cfg.VERSION,
            data_path=ROOT_DIR / 'data' / 'nuscenes',
            save_path=ROOT_DIR / 'data' / 'nuscenes',
            max_sweeps=dataset_cfg.MAX_SWEEPS,
            with_cam=args.with_cam
        )

        nuscenes_dataset = NuScenesDataset(
            dataset_cfg=dataset_cfg, class_names=None,
            root_path=ROOT_DIR / 'data' / 'nuscenes',
            logger=common_utils.create_logger(), training=True
        )
        nuscenes_dataset.create_groundtruth_database(max_sweeps=dataset_cfg.MAX_SWEEPS)

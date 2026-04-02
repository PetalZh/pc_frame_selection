import json, pickle
import os
from sklearn.metrics import accuracy_score
import math
import torch
import torch.nn.functional as F
from openpyxl import load_workbook
from openpyxl.utils.cell import coordinate_from_string, column_index_from_string
from openpyxl import Workbook
import numpy as np
from sklearn.metrics import confusion_matrix
import cv2
import random
from datetime import datetime


def load_eval_file(file_path):
    if os.path.exists(file_path):
        result_dict = json.load(open(file_path))
    return result_dict


def sigmoid(x):
    y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
    return y

def extract_centers(heatmap, obj_name, thresh=0.5, radius=2):
    device = heatmap.device
    batch, c, h, w = heatmap.shape
    all_centers = []

    for b in range(batch):
        batch_centers = []
        for ch in range(c):
            heatmap_bch = heatmap[b, ch, :, :]

            # print(heatmap_bch.shape)
            hm_np = heatmap.numpy()
            hm = (hm_np * 255).astype('uint8')

            # Apply threshold to the heatmap
            keep = heatmap_bch > thresh
            heatmap_bch = heatmap_bch * keep.float()

            # Suppress non-local maxima
            heatmap_max = F.max_pool2d(heatmap_bch.unsqueeze(0).unsqueeze(0), (2 * radius + 1, 2 * radius + 1), stride=1, padding=radius)
            keep = (heatmap_bch == heatmap_max.squeeze(0).squeeze(0))
            heatmap_bch = heatmap_bch * keep.float()

            # Get indices of the center points
            y_idx, x_idx = torch.nonzero(heatmap_bch, as_tuple=True)

            for i in range(len(y_idx)):
                batch_centers.append((x_idx[i].item(), y_idx[i].item()))
        
        all_centers.append(batch_centers)

    # Convert list of centers to tensor
    centers_tensor = [torch.tensor(centers, dtype=torch.float32, device=device) for centers in all_centers]
    max_len = max(len(centers) for centers in centers_tensor)
    padded_centers = torch.zeros(batch, max_len, 2, device=device)
    for i, centers in enumerate(centers_tensor):
        if len(centers) > 0:
            padded_centers[i, :len(centers), :] = centers
    
    num_centers = torch.tensor([len(centers) for centers in all_centers], dtype=torch.float32, device=device)
    
    return num_centers, padded_centers

def extract_centers_dynamic(heatmap, thresh=0.5, radius=2):
    device = heatmap.device
    batch, c, h, w = heatmap.shape
    all_centers = []

    for b in range(batch):
        batch_centers = []
        for ch in range(c):
            heatmap_bch = heatmap[b, ch, :, :]

            # print(heatmap_bch.shape)
            hm_np = heatmap.numpy()
            hm = (hm_np * 255).astype('uint8')
            
            ret, otsu_thresh = cv2.threshold(hm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # ret is the treshold
            # print(ret/255 + 0.1)
            if ret == 0:
                dynamic_thresh = 1
            else:
                dynamic_thresh = ret/255 + thresh
            # print('thresh: {} '.format(dynamic_thresh))

            # Apply threshold to the heatmap
            keep = heatmap_bch > dynamic_thresh
            heatmap_bch = heatmap_bch * keep.float()

            # Suppress non-local maxima
            heatmap_max = F.max_pool2d(heatmap_bch.unsqueeze(0).unsqueeze(0), (2 * radius + 1, 2 * radius + 1), stride=1, padding=radius)
            keep = (heatmap_bch == heatmap_max.squeeze(0).squeeze(0))
            heatmap_bch = heatmap_bch * keep.float()

            # Get indices of the center points
            y_idx, x_idx = torch.nonzero(heatmap_bch, as_tuple=True)

            for i in range(len(y_idx)):
                batch_centers.append((x_idx[i].item(), y_idx[i].item()))
        
        all_centers.append(batch_centers)

    # Convert list of centers to tensor
    centers_tensor = [torch.tensor(centers, dtype=torch.float32, device=device) for centers in all_centers]
    max_len = max(len(centers) for centers in centers_tensor)
    padded_centers = torch.zeros(batch, max_len, 2, device=device)
    for i, centers in enumerate(centers_tensor):
        if len(centers) > 0:
            padded_centers[i, :len(centers), :] = centers
    
    num_centers = torch.tensor([len(centers) for centers in all_centers], dtype=torch.float32, device=device)
    
    return num_centers, padded_centers

def extract_centers_dynamic2(heatmap, obj_name, thresh=0.5, radius=2):
    device = heatmap.device
    batch, c, h, w = heatmap.shape
    all_centers = []

    # print('heat map shape: {}'.format(heatmap.shape))
    # print(heatmap[1, 0, :, :])
    for b in range(batch):
        batch_centers = []
        for ch in range(c):
            heatmap_bch = heatmap[b, ch, :, :]

            # print(heatmap_bch.shape)
            hm_np = heatmap.numpy()
            hm = (hm_np * 255).astype('uint8')
            
            ret, otsu_thresh = cv2.threshold(hm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # ret is the treshold
            # print(ret/255 + 0.1)

            if obj_name == 'motorcycle' or obj_name == 'bicycle':
                thresh = 0.12
            if ret == 0:
                dynamic_thresh = 1
            else:
                dynamic_thresh = ret/255 + thresh
            # print('thresh: {} '.format(dynamic_thresh))

            # Apply threshold to the heatmap
            keep = heatmap_bch > dynamic_thresh
            heatmap_bch = heatmap_bch * keep.float()

            # Suppress non-local maxima
            heatmap_max = F.max_pool2d(heatmap_bch.unsqueeze(0).unsqueeze(0), (2 * radius + 1, 2 * radius + 1), stride=1, padding=radius)
            keep = (heatmap_bch == heatmap_max.squeeze(0).squeeze(0))
            heatmap_bch = heatmap_bch * keep.float()

            # Get indices of the center points
            y_idx, x_idx = torch.nonzero(heatmap_bch, as_tuple=True)

            for i in range(len(y_idx)):
                batch_centers.append((x_idx[i].item(), y_idx[i].item()))
        
        all_centers.append(batch_centers)

    # Convert list of centers to tensor
    centers_tensor = [torch.tensor(centers, dtype=torch.float32, device=device) for centers in all_centers]
    max_len = max(len(centers) for centers in centers_tensor)
    padded_centers = torch.zeros(batch, max_len, 2, device=device)
    for i, centers in enumerate(centers_tensor):
        if len(centers) > 0:
            padded_centers[i, :len(centers), :] = centers
    
    num_centers = torch.tensor([len(centers) for centers in all_centers], dtype=torch.float32, device=device)

    # print('num_centers: {}'.format(num_centers))
    # print('padded centers: ')
    # print(padded_centers)
    
    return num_centers, padded_centers


def hm(thresh, radius):
    print('thresh: {}, radius: {}'.format(thresh, radius))
    # objs = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
    objs = ['car', 'pedestrian', 'cyclist']
    gt_path = '/data/kitti_gt_list3.txt'#'/data/kitti_gt_list2.txt' #'gt_list.txt'
    if os.path.exists(gt_path):
        gt_list = json.load(open(gt_path))

    result_dict = {}
    for i in range(3000): #15050
        obj_idx = i % 3
        file_path = '/data/kitti_hm4/{}.txt'.format(i)
        if os.path.exists(file_path):
            hm= json.load(open(file_path))
            hm= torch.tensor(hm, dtype=torch.float32)
            hm=hm.unsqueeze(1)
            hm = hm + 25
            
            # print(hm)
            # hm = hm.permute(1, 0, 2, 3)
            # hm = sigmoid(hm)
            # print(hm)
            num_centers, padded_centers = extract_centers(hm, objs[obj_idx], thresh, radius=radius)
            # print('number of centers: {}'.format(num_centers))
            
            num_centers = num_centers.tolist()
            gt = gt_list[i]

        obj = objs[obj_idx]
        if obj in result_dict.keys():
            result_dict[obj]['gt_count'].extend(gt)
            result_dict[obj]['center_count'].extend(num_centers)
        else:
            result_dict[obj] = {'gt_count': gt, 'center_count': num_centers}
        
        # if obj == 'car':
        #     print('gt count: {}, pred: {}'.format(gt, num_centers))

        if i % 2000 == 0:
            print(i)
    # eval2(result_dict)
    file = 'result_output/kitti/result_counternet2_{}_{}.txt'.format(thresh, radius)
    # file = 'result_output/kitti_result_counternet.txt'
    with open(file, 'w') as f: 
        json.dump(result_dict, f)
    return result_dict

def hm_partition_kitti(thresh, radius):
    print('thresh: {}, radius: {}'.format(thresh, radius))
    objs = ['car', 'pedestrian', 'cyclist']
    gt_path = '/data/kitti_gt_list3.txt'
    if os.path.exists(gt_path):
        gt_list = json.load(open(gt_path))

    print(len(gt_list))

    result_dict = {}
    start_time = datetime.now()
    for i in range(2826): #15050
        obj_idx = i % 3
        file_path = '/data/kitti_hm_partition9/{}.txt'.format(i)
        if os.path.exists(file_path):
            hms= json.load(open(file_path))
            # print(len(hms[0]))
            center_counts = torch.zeros(len(hms[0])) # batch size 4
            for hm in hms:
                hm= torch.tensor(hm, dtype=torch.float32)
                hm=hm.unsqueeze(1)
                hm = hm + 25
                # print(hm.shape)

                # hm = sigmoid(hm)
                num_centers, padded_centers = extract_centers_dynamic(hm, thresh, radius=radius)
               
                center_counts += num_centers
                # print('hm shpe: ')
                # print(num_centers.shape)
                
            num_centers = center_counts.tolist()
            # print(num_centers)
            
            gt = gt_list[i]

            # if obj_idx == 0:
            #     print('gt {}, pred {}'.format(gt, num_centers))

            obj = objs[obj_idx]
            if obj in result_dict.keys():
                result_dict[obj]['gt_count'].extend(gt)
                result_dict[obj]['center_count'].extend(num_centers)
            else:
                result_dict[obj] = {'gt_count': gt, 'center_count': num_centers}
        else:
            print('not exist')
        
        # if obj == 'car':
        #     print('gt count: {}, pred: {}'.format(gt, num_centers))

        if i % 1000 == 0:
            print(i)
            # end_time = datetime.now()
    
    file = 'result_output/kitti/result_center_partition9_dynamic_{}_{}.txt'.format(thresh, radius)
    # file = 'result_output/result_center_partition_9_dynamic.txt'
    with open(file, 'w') as f: 
        json.dump(result_dict, f)
    return result_dict

def remove_duplicate_centers(tensor, radius=3):
    # Get the batch size and dimensions of each feature map
    batch_size, height, width = tensor.shape

    # Process each batch independently
    for b in range(batch_size):
        for i in range(height):
            for j in range(width):
                # If the current center is 1, process its neighborhood
                if tensor[b, i, j] == 1:
                    # Calculate the neighborhood bounds
                    min_x = max(0, i - radius)
                    max_x = min(height, i + radius + 1)
                    min_y = max(0, j - radius)
                    max_y = min(width, j + radius + 1)
                    
                    # Set the surrounding centers within the radius to 0, except the center itself
                    for x in range(min_x, max_x):
                        for y in range(min_y, max_y):
                            # Check if the point is within the circle of given radius
                            if (x - i)**2 + (y - j)**2 <= radius**2:
                                if x != i or y != j:
                                    tensor[b, x, y] = 0
    return tensor


def hm_partition_overlap(thresh, radius, radius_duplicate = 10):
    print('thresh: {}, radius: {}, radius duplicate: {}'.format(thresh, radius, radius_duplicate))
    objs = ['car', 'pedestrian', 'cyclist']
    gt_path = '/data/kitti_gt_list3.txt'
    if os.path.exists(gt_path):
        gt_list = json.load(open(gt_path))

    result_dict = {}

    expanding_rate = 0.2
    expanding = math.floor(200 * expanding_rate)
    # # x_start, x_end, y_start, y_end
    # ranges = [[0, 200+expanding, 0, 176+expanding],[0, 200+expanding, 176-expanding, 352],[200-expanding, 400, 0, 176+expanding],[200-expanding, 400, 176-expanding, 352]]
    # ranges = [[0, 200+expanding, 0, 352],[200-expanding, 400, 0, 352]]
    ranges = [[0, 130+expanding, 0, 115+expanding], [130-expanding, 260+expanding, 0, 115+expanding], [260-expanding, 400, 0, 115+expanding], [0, 130+expanding, 115-expanding, 230+expanding], [130-expanding, 260+expanding, 115-expanding, 230+expanding], [260-expanding, 400, 115-expanding, 230+expanding], [0, 130+expanding, 230-expanding, 352], [130-expanding, 260+expanding, 230-expanding, 352], [260-expanding, 400, 230-expanding, 352]]
    
    start_time = datetime.now()
    for i in range(2826): #150
        obj_idx = i % 3
        file_path = '/data/kitti_hm_partition9_overlap/{}.txt'.format(i)
        if os.path.exists(file_path):
            hms= json.load(open(file_path))
            # partition number of hms
            maps = np.zeros((4, 400, 400)) # batch size
            for hm_idx, hm in enumerate(hms):
                coord_range = ranges[hm_idx]
                hm= torch.tensor(hm, dtype=torch.float32)
                hm=hm.unsqueeze(1)
                hm = hm + 25
                # print(hm)
                # hm = sigmoid(hm)
                num_centers, padded_centers = extract_centers_dynamic2(hm, objs[obj_idx], thresh, radius=radius)
               
                for batch_idx, batch_centers in enumerate(padded_centers):
                    for center in batch_centers:
                        y = int(center[0])
                        x = int(center[1])
                        if x != 0 and y != 0:
                            maps[batch_idx, coord_range[2] + y, coord_range[0] + x] = 1

            final_maps = remove_duplicate_centers(maps, radius_duplicate)

            num_centers = np.sum(final_maps, axis=(1, 2))

            #Print the sums for each batch
            # print("Sum of each batch:", num_centers)
            
            gt = gt_list[i]

            obj = objs[obj_idx]
            if obj in result_dict.keys():
                result_dict[obj]['gt_count'].extend(gt)
                result_dict[obj]['center_count'].extend(num_centers.tolist())
            else:
                result_dict[obj] = {'gt_count': gt, 'center_count': num_centers.tolist()}
        else:
            print('not exist')
        
        # if obj == 'car':
        #     print('gt count: {}, pred: {}'.format(gt, num_centers))

        if i % 1000 == 0 and i != 0:
            print(i)
            # end_time = datetime.now()
            # time_diff = (end_time - start_time).total_seconds()
            # print('time: {}'.format(time_diff/100))
            # break
    
    # eval2(result_dict)
    file = 'result_output/kitti/result_center_partition9_overlap_{}.txt'.format(thresh)
    with open(file, 'w') as f: 
        json.dump(result_dict, f)
    return result_dict
def eval(thresh, radius, ep=0.1):
    # result_dict = load_eval_file('result_output/kitti/result_counternet2_23.9_10.txt') #24.2
    # result_dict = load_eval_file('result_output/kitti/result_center_partition2_dynamic_23.3_10.txt')
    # result_dict = load_eval_file('result_output/kitti/result_center_partition2_overlap_23.8.txt') #23.8
    # result_dict = load_eval_file('result_output/kitti/result_center_partition9_dynamic_23.3_10.txt') 
    # result_dict = load_eval_file('result_output/kitti/result_center_partition9_overlap_23.4.txt')
    # result_dict = load_eval_file('result_output/kitti/results/result_model_selection_adjustment_0.1.txt')
    # result_dict = load_eval_file('result_output/kitti/results/partition_2.txt')
    # result_dict = load_eval_file('result_output/kitti/result_center_partition_overlap_23.8.txt') #23.3
    # result_dict = load_eval_file('result_output/kitti/partition_4_overlap_0.2.txt')
    # result_dict = load_eval_file('result_output/kitti/partition_4.txt')
    # result_dict = load_eval_file('result_output/kitti/counternet.txt')
    
    # result_dict = load_eval_file('result_output/kitti_sta_center_point_0.28.txt')

    result_dict = load_eval_file('result_output/kitti/results/partition_4_overlap_0.2.txt')
    # result_dict = load_eval_file('result_output/kitti/results/partition_4.txt')
    # result_dict = load_eval_file('result_output/kitti/results/counternet.txt')
    # result_dict = load_eval_file('result_output/kitti_sta_center_point_0.28.txt')

    obj_list = []
    acc_list = []

    print(result_dict.keys())
    for key in result_dict.keys():
        pred = result_dict[key]['center_count']
        gt = result_dict[key]['gt_count']

        # print('pred:  {}, gt: {}'.format(pred, gt))
        # acc = accuracy_score(gt, pred)
        # print('category: {}, acc: {}'.format(key, acc))
        correct = 0
        print(key)
        
        diff = math.ceil(max(gt) * 0.05)
        print('diff {}'.format(diff))
        for i in range(len(gt)):
            if gt[i] >= pred[i] - diff and gt[i] <= pred[i] + diff:
                correct += 1

        acc = correct/len(pred)
        print('category: {}, acc: {}'.format(key, acc))
        
        obj_list.append(key)
        acc_list.append(acc)

    return obj_list,acc_list

def evalRecall():
    # result_dict = load_eval_file('result_output/kitti/partition_4_overlap_0.2.txt')
    # result_dict = load_eval_file('result_output/kitti/partition_4.txt')
    # result_dict = load_eval_file('result_output/kitti/counternet.txt')
    result_dict = load_eval_file('result_output/kitti_sta_center_point_0.28.txt')

    obj_list = []
    acc_list = []

    print(result_dict.keys())
    for key in result_dict.keys():
        pred = result_dict[key]['center_count']
        gt = result_dict[key]['gt_count']

        # print('pred:  {}, gt: {}'.format(pred, gt))
        # acc = accuracy_score(gt, pred)
        # print('category: {}, acc: {}'.format(key, acc))
        correct = 0
        all = 0
        print(key)
        
        diff = math.ceil(max(gt) * 0.05)
        print('diff {}'.format(diff))
        for i in range(len(gt)):
            if gt[i] != 0:
                all += 1
                if gt[i] >= pred[i] - diff and gt[i] <= pred[i] + diff:
                    correct += 1

        acc = correct/all #len(pred)
        print('category: {}, recall: {}'.format(key, acc))
        
        obj_list.append(key)
        acc_list.append(acc)

    return obj_list,acc_list

def eval_binary():
    result_dict = load_eval_file('result_output/kitti/partition_4_overlap_0.2.txt')
    # result_dict = load_eval_file('result_output/kitti/partition_4.txt')
    # result_dict = load_eval_file('result_output/kitti/counternet.txt')
    # result_dict = load_eval_file('result_output/kitti_sta_center_point_0.28.txt')
    
    obj_list = []
    acc_list = []

    print(result_dict.keys())
    for key in result_dict.keys():
        pred = result_dict[key]['center_count']
        gt = result_dict[key]['gt_count']

        # print('pred:  {}, gt: {}'.format(pred, gt))
        # acc = accuracy_score(gt, pred)
        # print('category: {}, acc: {}'.format(key, acc))
        correct = 0
        total = 0
        print(key)
        
        diff = 0 #math.ceil(max(gt) * 0.05)
        # print('diff {}'.format(diff))
        # for i in range(len(gt)):
        #     if gt[i] >= pred[i] - diff and gt[i] <= pred[i] + diff:
        #         correct += 1

        for i in range(len(gt)):
            if gt[i] != 0 and pred[i] != 0:
                correct += 1
            if gt[i] == 0 and pred[i] == 0:
                correct += 1

        print('total {}'.format(total))
        acc = correct/len(pred)
        print('category: {}, acc: {}'.format(key, acc))
        
        obj_list.append(key)
        acc_list.append(acc)

    return obj_list,acc_list

def eval_count():
    result_dict = load_eval_file('result_output/kitti/partition_4_overlap_0.2.txt')
    result_dict = load_eval_file('result_output/kitti/partition_4.txt')
    result_dict = load_eval_file('result_output/kitti/counternet.txt')
    # result_dict = load_eval_file('result_output/kitti_sta_center_point_0.28.txt')


    obj_list = []
    acc_list = []

    print(result_dict.keys())
    for key in result_dict.keys():
        pred = result_dict[key]['center_count']
        gt = result_dict[key]['gt_count']

        # print('pred:  {}, gt: {}'.format(pred, gt))
        # acc = accuracy_score(gt, pred)
        # print('category: {}, acc: {}'.format(key, acc))
        correct = 0
        total = 0
        print(key)
        
        diff = math.ceil(max(gt) * 0.05)
        # print('diff {}'.format(diff))
        # for i in range(len(gt)):
        #     if gt[i] >= pred[i] - diff and gt[i] <= pred[i] + diff:
        #         correct += 1
        
        total_acc = 0
        for j in range(1000):
            correct = 0
            total = 0
            random_number = random.randint(0, len(gt) - 1 - 250)
            for i in range(random_number, random_number + 250 +1):
                if gt[i] == 5:
                    total += 1
                    # if gt[i] >= pred[i] - diff and gt[i] <= pred[i] + diff:
                    if gt[i] >= pred[i] - diff and gt[i] <= pred[i] + diff:
                        correct += 1
            if total == correct:
                total_acc += 1
            else:
                total_acc += correct/total


        acc = total_acc/1000 #correct/len(pred)
        print('category: {}, acc: {}'.format(key, acc))
        
        obj_list.append(key)
        acc_list.append(acc)

    return obj_list,acc_list

def eval_agg():
    result_dict = load_eval_file('result_output/kitti/partition_4_overlap_0.2.txt')
    # result_dict = load_eval_file('result_output/kitti/partition_4.txt')
    # result_dict = load_eval_file('result_output/kitti/counternet.txt')
    # result_dict = load_eval_file('result_output/kitti_sta_center_point_0.28.txt')
    
    obj_list = []
    acc_list = []

    print(result_dict.keys())
    for key in result_dict.keys():
        pred = result_dict[key]['center_count']
        gt = result_dict[key]['gt_count']

        # print('pred:  {}, gt: {}'.format(pred, gt))
        # acc = accuracy_score(gt, pred)
        # print('category: {}, acc: {}'.format(key, acc))
        correct = 0
        total = 0
        print(key)
        
        diff = math.ceil(max(gt) * 0.05)
        # print('diff {}'.format(diff))
        # for i in range(len(gt)):
        #     if gt[i] >= pred[i] - diff and gt[i] <= pred[i] + diff:
        #         correct += 1
        
        total_diff_avg = 0
        for j in range(1000):
            gt_sum = 0
            pred_sum = 0
            random_number = random.randint(0, len(gt) - 1 - 150)
            for i in range(random_number, random_number + 150 +1):
                gt_sum += gt[i]
                pred_sum += pred[i]


            # if total == correct:
            #     total_acc += 1
            # else:
            #     total_acc += correct/total
            total_diff_avg += abs(gt_sum - pred_sum)/250


        acc = total_diff_avg/1000
        print('category: {}, acc: {}'.format(key, acc))
        
        obj_list.append(key)
        acc_list.append(acc)

    return obj_list,acc_list

def eval_agg_q_error():
    result_dict = load_eval_file('result_output/kitti/results/partition_4_overlap_0.2.txt')
    # result_dict = load_eval_file('result_output/kitti/results/partition_4.txt')
    # result_dict = load_eval_file('result_output/kitti/results/counternet.txt')
    # result_dict = load_eval_file('result_output/kitti_sta_center_point_0.28.txt')
    
    obj_list = []
    acc_list = []

    print(result_dict.keys())
    for key in result_dict.keys():
        pred = result_dict[key]['center_count']
        gt = result_dict[key]['gt_count']
       
        print(key)
        
        total_diff_avg = 0
        for j in range(1000):
            q_errors = []
            gt_sum = []
            pred_sum = []
            random_number = random.randint(0, len(gt) - 1 - 250)
            for i in range(random_number, random_number + 250 +1):
                gt_sum.append(gt[i])
                pred_sum.append(pred[i])

            
            for t, e in zip(gt_sum, pred_sum):
                if t == 0 and e == 0:
                    q_errors.append(1.0)
                elif t == 0 or e == 0:
                    t += 1
                    e += 1  
                    q_errors.append(np.maximum(t / e, e / t))
                else:
                    q_errors.append(np.maximum(t / e, e / t))

            total_diff_avg += np.median(q_errors)

        acc = total_diff_avg/1000
        print('category: {}, acc: {}'.format(key, acc))
        
        obj_list.append(key)
        acc_list.append(acc)

    return obj_list,acc_list


def eval_agg_q_error2():
    result_dict = load_eval_file('result_output/kitti/results/partition_4_overlap_0.2.txt')
    # result_dict = load_eval_file('result_output/kitti/results/partition_4.txt')
    # result_dict = load_eval_file('result_output/kitti/results/counternet.txt')
    result_dict = load_eval_file('result_output/kitti_sta_center_point_0.28.txt')
    
    obj_list = []
    acc_list = []

    print(result_dict.keys())
    for key in result_dict.keys():
        pred = result_dict[key]['center_count']
        gt = result_dict[key]['gt_count']
       
        print(key)
        
        total_diff_avg = []
        for j in range(1000):
            q_errors = []
            gt_sum = []
            pred_sum = []
            random_number = random.randint(0, len(gt) - 1 - 250)
            for i in range(random_number, random_number + 250 +1):
                gt_sum.append(gt[i])
                pred_sum.append(pred[i])

            
            gt1 = sum(gt_sum) + 1
            pred1 = sum(pred_sum) + 1
            q_errors = np.maximum(gt1/ pred1, pred1 / gt1)
            total_diff_avg.append(q_errors)

        acc = np.median(q_errors)
        print('category: {}, acc: {}'.format(key, acc))
        
        obj_list.append(key)
        acc_list.append(acc)

    return obj_list,acc_list

def eval_combine():
    # object_combines = [['Car', 'Pedestrian'], ['Car', 'Cyclist'], ['Pedestrian', 'Cyclist'], ['Car', 'Pedestrian', 'Cyclist']]
    object_combines = [['car', 'pedestrian'], ['car', 'cyclist'], ['pedestrian', 'cyclist'], ['car', 'pedestrian', 'cyclist']]

    result_dict = load_eval_file('result_output/kitti/partition_4_overlap_0.2.txt')
    # result_dict = load_eval_file('result_output/kitti/partition_4.txt')
    # result_dict = load_eval_file('result_output/kitti/counternet.txt')
    # result_dict = load_eval_file('result_output/kitti_sta_center_point_0.28.txt')

    for combine in object_combines:
        frame_ids = len(result_dict['car']['center_count'])
        correct = 0

        for i in range(frame_ids):
            is_correct = True
            for item in combine:
                pred = result_dict[item]['center_count']
                gt = result_dict[item]['gt_count']

                diff = math.ceil(max(gt) * 0.1)
                
                if (gt[i] >= pred[i] - diff and gt[i] <= pred[i] + diff) == False:
                        is_correct = False
                
            if is_correct:
                correct += 1

        acc = correct/frame_ids
        print('category: {}, acc: {}'.format(combine, acc))

def eval_single_frame(model_name, frame_idx, objs):
    pred_list = []
    gt_list = []

    path = 'result_output/kitti/results/{}.txt'.format(model_name)
    # print(path)
    result_dict = load_eval_file(path)
    # print(result_dict.keys())
    for key in objs:
        pred = result_dict[key]['center_count'][frame_idx]
        gt = result_dict[key]['gt_count'][frame_idx]

        pred_list.append(pred)
        gt_list.append(gt)
    return pred_list, gt_list

def getFeature():
    objs = ['car', 'pedestrian', 'cyclist']
    model_list = ['partition_2_overlap_0.2', 'partition_2', 'partition_4_overlap_0.2',  'partition_4', 'partition_9_overlap_0.2', 'partition_9']

    model_frame_dict = {string: [] for string in model_list}

    frame_idx = 0
    for i in range(4000): #15050
        # for each i, find 10 obj in each model
        file_path = '/data/feature_kitti/' + '{}.txt'.format(i)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                batch_tensor = pickle.load(f)
                # print(batch_tensor.shape)
                
                for tensor in batch_tensor:
                    
                    selected_model_idx = 0
                    min_diff = 1000
                    for m_idx, model in enumerate(model_list):
                        # get model
                        try:
                            preds, gts = eval_single_frame(model, frame_idx, objs) # expect to return count of all object and the corresponding gt in list
                        except IndexError:
                            break
                        # print(preds, gts)
                        # diff = sum(abs(a - b) for a, b in zip(preds, gts))

                        diff = 0
                        # weight = [0.2, 0.1 ,0.1 ,0.1 ,0.1 , 0.1, 0.1 , 0.1, 0.2, 0.1]
                        for idx, obj in enumerate(objs):
                            # diff += weight[idx] * abs(preds[idx] - gts[idx])
                            diff += abs(preds[idx] - gts[idx])

                        # print('frame_id: {}, model: {}, diff: {}'.format(frame_idx, model, diff))
                        if diff < min_diff:
                            selected_model_idx = m_idx
                            min_diff = diff
                    model_frame_dict[model_list[selected_model_idx]].append(tensor.tolist())    
                    frame_idx += 1
                    
    for key in model_frame_dict.keys():
        print('key: {}, length: {}'.format(key, len(model_frame_dict[key])))
    # print(model_frame_dict)
    file = 'result_output/kitti/model_frame_dict.txt'
    with open(file, 'w') as f: 
        json.dump(model_frame_dict, f)

def getCenter():
    center_dict = {}
    file_path = 'result_output/kitti/model_frame_dict.txt'
    if os.path.exists(file_path):
            model_frame_dict = json.load(open(file_path))
            for key in model_frame_dict.keys():
                data_array = np.array(model_frame_dict[key])
                avg_pool = np.mean(data_array, axis=0)
                
                distance_sum = 0
                for array in data_array:
                    distance = np.linalg.norm(array - avg_pool)
                    distance_sum += distance
                avg_distance = distance_sum/len(data_array) # sigma square
                print(avg_distance)

                # adjustment chernoff bound
                epsilon = 0.05
                # print('key: {}, avg_dist: {}'.format(key, avg_distance))
                power = -(len(data_array) * epsilon*epsilon) / (2 * avg_distance)
                adjustment = math.exp(power)
                print('key: {}, adjustment: {}'.format(key, adjustment))
                # model_frame_dict[key] = avg_pool.tolist()

                if key not in center_dict.keys():
                    center_dict[key] = {'center': avg_pool, 'adjustment': adjustment}
    return center_dict

def evalModelSelection():
    center_dict = getCenter()
    output_dict = {}
    objs = ['car', 'pedestrian', 'cyclist']

    model_list = ['partition_2_overlap_0.2', 'partition_2', 'partition_4_overlap_0.2',  'partition_4', 'partition_9_overlap_0.2', 'partition_9']

    frame_idx = 0
    model_count = {}
    for i in range(4000): #15050
        # for each i, find 10 obj in each model
        file_path = '/data/feature_kitti/' + '{}.txt'.format(i)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                batch_tensor = pickle.load(f)
                # print(batch_tensor.shape)
                for tensor in batch_tensor:
                    selected_model = model_list[0]
                    distance_min = 1000
                    for m_idx, model in enumerate(model_list):
                        center = center_dict[model]['center'] #'adjustment'
                        adjustment = center_dict[model]['adjustment']
                        distance = np.linalg.norm(center - tensor.tolist())
                        distance += adjustment * distance
                        # print('model: {}, adjust: {}, distance: {}'.format(model, adjustment, distance))
                        if distance < distance_min:
                            distance_min = distance
                            selected_model = model

                    if selected_model not in model_count.keys():
                        model_count[selected_model] = 1
                    else:
                        model_count[selected_model] += 1

                    try:
                        preds, gts = eval_single_frame(selected_model, frame_idx, objs) # expect to return count of all object and the corresponding gt in list
                    except IndexError:
                        break

                    for idx, pred in enumerate(preds):
                        obj = objs[idx]
                        if obj not in output_dict.keys():
                            output_dict[obj] = {'gt_count': [gts[idx]], 'center_count':[pred]}
                        else:
                            output_dict[obj]['gt_count'].append(gts[idx])
                            output_dict[obj]['center_count'].append(pred)
                    frame_idx += 1
    # print(frame_idx)
    # print(len(output_dict['pedestrian']['gt_count']))
    print(model_count)
    file = 'result_output/kitti/results/result_model_selection_adjustment_0.05.txt'
    with open(file, 'w') as f: 
        json.dump(output_dict, f)


def combine():
    output_dict = {}
    
    objs = ['car', 'pedestrian', 'cyclist']
    # result_dict1 = load_eval_file('result_output/kitti/result_counternet2_23.9_10.txt')
    # result_dict2 = load_eval_file('result_output/kitti/result_counternet2_24.2_10.txt')
    # result_dict1 = load_eval_file('result_output/kitti/result_center_partition2_overlap_23.3.txt')
    # result_dict2 = load_eval_file('result_output/kitti/result_center_partition2_overlap_23.8.txt')
    result_dict1 = load_eval_file('result_output/kitti/result_center_partition2_dynamic_23.3_10.txt')
    result_dict2 = load_eval_file('result_output/kitti/result_center_partition2_dynamic_23.8_10.txt')

    for obj in objs:
        if obj == 'pedestrian' or obj == 'cyclist':
            output_dict[obj] = result_dict2[obj]
        else:
            output_dict[obj] = result_dict1[obj]
    file = 'result_output/kitti/results/partition_2.txt'
    with open(file, 'w') as f: 
        json.dump(output_dict, f)



# getFeature()
# getCenter()
# evalModelSelection()
# combine()

eval(0.1, 10)

# eval_agg_q_error2()
# evalRecall()
# eval_binary()
# eval_count()
# eval_agg()
# eval_combine()
# hm_partition_kitti(23.3, 10)
# hm_partition_kitti(23.4, 10)
# hm_partition_kitti(23.5, 10)
# hm_partition_kitti(23, 10)
# hm_partition_kitti(23.1, 10)
# hm_partition_kitti(23.2, 10)
# hm_partition_kitti(23.6, 10)
# hm_partition_kitti(23.7, 10)
# hm_partition_kitti(23.8, 10)


# hm_partition_overlap(22.9, 10, radius_duplicate = 10)
# hm_partition_overlap(22.8, 10, radius_duplicate = 10)
# hm_partition_overlap(22.7, 10, radius_duplicate = 10)
# hm_partition_overlap(23, 10, radius_duplicate = 10)
# hm_partition_overlap(23.1, 10, radius_duplicate = 10)
# hm_partition_overlap(23.2, 10, radius_duplicate = 10)
# hm_partition_overlap(23.3, 10, radius_duplicate = 10)
# hm_partition_overlap(23.4, 10, radius_duplicate = 10)
# hm_partition_overlap(23.5, 10, radius_duplicate = 10)
# hm_partition_overlap(23.6, 10, radius_duplicate = 10)
# hm_partition_overlap(23.7, 10, radius_duplicate = 10)
# hm_partition_overlap(23.8, 10, radius_duplicate = 10)
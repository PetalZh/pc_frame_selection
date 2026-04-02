import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_
from ..model_utils import model_nms_utils
from ..model_utils import centernet_utils
from ...utils import loss_utils
from functools import partial
from ...constants import result, gt_list
import json


class SeparateHead(nn.Module):
    #-2.19
    def __init__(self, input_channels, sep_head_dict, init_bias=-2.19, use_bias=False, norm_func=None):
        super().__init__()
        self.sep_head_dict = sep_head_dict
        # print('sep head dict: ')
        # print(self.sep_head_dict)
        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']

            fc_list = []
            for k in range(num_conv - 1):
                fc_list.append(nn.Sequential(
                    nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
                    nn.BatchNorm2d(input_channels) if norm_func is None else norm_func(input_channels),
                    nn.ReLU()
                ))
            fc_list.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True))
            fc = nn.Sequential(*fc_list)
            if 'hm' in cur_name:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_normal_(m.weight.data)
                        if hasattr(m, "bias") and m.bias is not None:
                            nn.init.constant_(m.bias, 0)

            self.__setattr__(cur_name, fc)

    def forward(self, x):
        ret_dict = {}
        for cur_name in self.sep_head_dict:
            ret_dict[cur_name] = self.__getattr__(cur_name)(x)

        return ret_dict

class CounterHeadPartitionMultiScale(nn.Module):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size,
                 predict_boxes_when_training=True):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.feature_map_stride = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', None)

        self.class_names = class_names
        self.class_names_each_head = []
        self.class_id_mapping_each_head = []
        self.progress = 0

        for cur_class_names in self.model_cfg.CLASS_NAMES_EACH_HEAD:
            self.class_names_each_head.append([x for x in cur_class_names if x in class_names])
            cur_class_id_mapping = torch.from_numpy(np.array(
                [self.class_names.index(x) for x in cur_class_names if x in class_names]
            )).cuda()
            self.class_id_mapping_each_head.append(cur_class_id_mapping)

        total_classes = sum([len(x) for x in self.class_names_each_head])
        assert total_classes == len(self.class_names), f'class_names_each_head={self.class_names_each_head}'

        norm_func = partial(nn.BatchNorm2d, eps=self.model_cfg.get('BN_EPS', 1e-5), momentum=self.model_cfg.get('BN_MOM', 0.1))
        self.shared_conv = nn.Sequential(
            nn.Conv2d(
                input_channels, self.model_cfg.SHARED_CONV_CHANNEL, 3, stride=1, padding=1,
                bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)
            ),
            norm_func(self.model_cfg.SHARED_CONV_CHANNEL),
            nn.ReLU(),
        )

        self.heads_list = nn.ModuleList()
        self.separate_head_cfg = self.model_cfg.SEPARATE_HEAD_CFG
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            cur_head_dict = copy.deepcopy(self.separate_head_cfg.HEAD_DICT)
            cur_head_dict['hm'] = dict(out_channels=len(cur_class_names), num_conv=self.model_cfg.NUM_HM_CONV)
            self.heads_list.append(
                SeparateHead(
                    input_channels=self.model_cfg.SHARED_CONV_CHANNEL,
                    sep_head_dict=cur_head_dict,
                    init_bias=-2.19,
                    use_bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False),
                    norm_func=norm_func
                )
            )
        self.predict_boxes_when_training = predict_boxes_when_training
        self.forward_ret_dict = {}
        self.build_losses()

    def build_losses(self):
        self.add_module('hm_loss_func', loss_utils.FocalLossCenterNet())
        self.add_module('reg_loss_func', loss_utils.RegLossCenterNet())

    def assign_target_of_single_head(
            self, partitions, num_classes, gt_boxes, feature_map_size, feature_map_stride, num_max_objs=500,
            gaussian_overlap=0.1, min_radius=2
    ):
        """
        Args:
            gt_boxes: (N, 8)
            feature_map_size: (2), [x, y]

        Returns:

        """
        heatmap = gt_boxes.new_zeros(num_classes, feature_map_size[1], feature_map_size[0])
        ret_boxes = gt_boxes.new_zeros((num_max_objs, gt_boxes.shape[-1] - 1 + 1))
        inds = gt_boxes.new_zeros(num_max_objs).long()
        mask = gt_boxes.new_zeros(num_max_objs).long()
        ret_boxes_src = gt_boxes.new_zeros(num_max_objs, gt_boxes.shape[-1])
        ret_boxes_src[:gt_boxes.shape[0]] = gt_boxes

        x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / feature_map_stride
        coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / feature_map_stride
        coord_x = torch.clamp(coord_x, min=0, max=feature_map_size[0] - 0.5)  # bugfixed: 1e-6 does not work for center.int()
        coord_y = torch.clamp(coord_y, min=0, max=feature_map_size[1] - 0.5)  #
        center = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
        center_int = center.int()
        center_int_float = center_int.float()
        
        # print('center int: ')
        # print(center_int)

        dx, dy, dz = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
        dx = dx / self.voxel_size[0] / feature_map_stride
        dy = dy / self.voxel_size[1] / feature_map_stride

        # print('dx: {}, dy: {}, '.format(dx, dy))
        # print('point cloud range: {}'.format(self.point_cloud_range))

        radius = centernet_utils.gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
        radius = torch.clamp_min(radius.int(), min=min_radius)

        mask_list = []
        inds_list = []
        heatmap_list = []
        # mask_list = gt_boxes.new_zeros()
        for idx, partition in enumerate(partitions):
            # print('partition {}:'.format(idx))
            for k in range(min(num_max_objs, gt_boxes.shape[0])):
                if dx[k] <= 0 or dy[k] <= 0:
                    continue

                if not (partition[0] <= center_int[k][0] <= partition[1] and partition[2] <= center_int[k][1] <= partition[3]):
                    continue

                cur_class_id = (gt_boxes[k, -1] - 1).long()
                centernet_utils.draw_gaussian_to_heatmap(heatmap[cur_class_id], center[k], radius[k].item())
                inds[k] = center_int[k, 1] * feature_map_size[0] + center_int[k, 0]
                mask[k] = 1

            mask_list.append(mask) 
            inds_list.append(inds)

            heatmap_list.append(heatmap)
            mask = gt_boxes.new_zeros(num_max_objs).long()
            heatmap = gt_boxes.new_zeros(num_classes, feature_map_size[1], feature_map_size[0])

        mask_list = torch.stack(mask_list)
        inds_list = torch.stack(inds_list)
        heatmap_list = torch.stack(heatmap_list)

        # print('heatmap list: ')
        # for item in heatmap_list:
        #     print(len(item))
        # print(heatmap_list[0])

        return heatmap_list, ret_boxes, inds_list, mask_list, ret_boxes_src

    def assign_targets(self, ranges, gt_boxes, feature_map_size=None, **kwargs):
        """
        Args:
            gt_boxes: (B, M, 8)
            range_image_polar: (B, 3, H, W)
            feature_map_size: (2) [H, W]
            spatial_cartesian: (B, 4, H, W)
        Returns:

        """
        feature_map_size = feature_map_size[::-1]  # [H, W] ==> [x, y]
        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        # feature_map_size = self.grid_size[:2] // target_assigner_cfg.FEATURE_MAP_STRIDE

        batch_size = gt_boxes.shape[0]
        ret_dict = {
            'heatmaps': [],
            'target_boxes': [],
            'inds': [],
            'masks': [],
            'heatmap_masks': [],
            'target_boxes_src': [],
        }
        # print('gt box: ')
        # print(gt_boxes.shape)

        all_names = np.array(['bg', *self.class_names])

        for idx, cur_class_names in enumerate(self.class_names_each_head):
            heatmap_list, target_boxes_list, inds_list, masks_list, target_boxes_src_list = [], [], [], [], []
            for bs_idx in range(batch_size):
                cur_gt_boxes = gt_boxes[bs_idx]
                gt_class_names = all_names[cur_gt_boxes[:, -1].cpu().long().numpy()]

                gt_boxes_single_head = []

                for idx, name in enumerate(gt_class_names):
                    if name not in cur_class_names:
                        continue
                    temp_box = cur_gt_boxes[idx]
                    temp_box[-1] = cur_class_names.index(name) + 1
                    gt_boxes_single_head.append(temp_box[None, :])

                if len(gt_boxes_single_head) == 0:
                    gt_boxes_single_head = cur_gt_boxes[:0, :]
                else:
                    gt_boxes_single_head = torch.cat(gt_boxes_single_head, dim=0)

                heatmap, ret_boxes, inds, mask, ret_boxes_src = self.assign_target_of_single_head(
                    ranges, num_classes=len(cur_class_names), gt_boxes=gt_boxes_single_head.cpu(),
                    feature_map_size=feature_map_size, feature_map_stride=target_assigner_cfg.FEATURE_MAP_STRIDE,
                    num_max_objs=target_assigner_cfg.NUM_MAX_OBJS,
                    gaussian_overlap=target_assigner_cfg.GAUSSIAN_OVERLAP,
                    min_radius=target_assigner_cfg.MIN_RADIUS,
                )
                heatmap_list.append(heatmap.to(gt_boxes_single_head.device))

                target_boxes_list.append(ret_boxes.to(gt_boxes_single_head.device))
                inds_list.append(inds.to(gt_boxes_single_head.device))
                masks_list.append(mask.to(gt_boxes_single_head.device))
                target_boxes_src_list.append(ret_boxes_src.to(gt_boxes_single_head.device))

            ret_dict['heatmaps'].append(torch.stack(heatmap_list, dim=0))
            ret_dict['target_boxes'].append(torch.stack(target_boxes_list, dim=0))
            ret_dict['inds'].append(torch.stack(inds_list, dim=0))
            ret_dict['masks'].append(torch.stack(masks_list, dim=0))
            ret_dict['target_boxes_src'].append(torch.stack(target_boxes_src_list, dim=0))
        return ret_dict

    def sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y

    def get_loss(self):
        pred_dicts = self.forward_ret_dict['pred_dicts']
        target_dicts = self.forward_ret_dict['target_dicts']
        partitions = self.forward_ret_dict['patitions']

        # print(pred_dicts[0].keys())
        # print(target_dicts.keys())

        tb_dict = {}
        loss = 0
        scales = [0, 0, 1, 1, 2, 2, 2, 2]

        for idx, pred_dict in enumerate(pred_dicts):
            head_hm_loss = 0
            head_count_loss = 0

            scales_total_obj = []
            batch_size = pred_dict[0]['hm'].shape[0]
            # (batch_size, partition_num)
            # weight = torch.full((batch_size, len(partitions)), 0.25, device='cuda', dtype=torch.float32)

            # In this version of implementation, no consideration of weight
            # if weight is considered, the weight should be init properly
            # weight = torch.full((batch_size, len(partitions)), 0, device='cuda', dtype=torch.float32)
            partition_weight = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25], device='cuda', dtype=torch.float32)
            weight = partition_weight.repeat(batch_size, 1)
            count_loss_l1 = torch.full((batch_size, len(partitions)), 0, device='cuda', dtype=torch.float32)
            
            
            for i in range(batch_size):
                # total_obj.append(torch.sum(target_dicts['masks'][idx][i, :, :]).to('cuda'))
                scale1 = torch.sum(target_dicts['masks'][idx][i, 0:1, :]).to('cuda')
                scale2 = torch.sum(target_dicts['masks'][idx][i, 2:3, :]).to('cuda')
                scale3 = torch.sum(target_dicts['masks'][idx][i, 4:7, :]).to('cuda')

                scale_total_count = [scale1, scale2, scale3]
                scales_total_obj.append(scale_total_count)


            for i in range(len(partitions)):
                # pred_dict[i]['hm']
                # print('partition {}'.format(i))
                scale_id = scales[i]
                
                hm = self.sigmoid(pred_dict[i]['hm'])
                par = partitions[i]

                center_count, centers = self.extract_centers(pred_dict[i]['hm'], threshold=0.6)
                
                for j in range(batch_size):
                    gt_count = torch.sum(target_dicts['masks'][idx][j, i, :])
                    count_loss_l1[j, i] = torch.abs(center_count[j] - gt_count)

                    # print(gt_count)

                    if scales_total_obj[j][scale_id] != 0:
                        weight[j, i] += gt_count/scales_total_obj[j][scale_id]
                    
                
                hm_loss = self.hm_loss_func(hm, target_dicts['heatmaps'][idx][:, i, :, par[0]:par[1], par[2]:par[3]])
                hm_loss *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
                head_hm_loss += hm_loss

            # print(count_loss_l1)
            head_count_loss = torch.sum(weight * count_loss_l1)   
            # head_count_loss = torch.sum(weight * count_loss_l1)   

            loss += head_hm_loss + head_count_loss
            tb_dict['hm_loss_head_%d' % idx] = head_hm_loss.item() + head_count_loss.item()
        return loss, tb_dict
    
    def extract_centers(self, heatmap, threshold=0.5, radius=2):
        """
        Extract centers from the heatmap.
        Args:
            heatmap: Predicted heatmap from the network (batch x c x h x w).
            threshold: Confidence threshold to filter out low-confidence points.
            radius: Radius for local maxima suppression.
        Returns:
            num_centers: Tensor representing the number of detected centers.
            centers: Tensor of shape (n, 2) with (x, y) coordinates for detected centers.
        """
        device = heatmap.device
        batch, c, h, w = heatmap.shape
        all_centers = []
        heatmap= self.sigmoid(heatmap)

        for b in range(batch):
            batch_centers = []
            for ch in range(c):
                heatmap_bch = heatmap[b, ch]
                # Apply threshold to the heatmap
                keep = heatmap_bch > threshold
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
    
    def generate_predicted_boxes(self, batch_size, pred_dicts):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        post_center_limit_range = torch.tensor(post_process_cfg.POST_CENTER_LIMIT_RANGE).cuda().float()

        ret_dict = [{
            'pred_boxes': [],
            'pred_scores': [],
            'pred_labels': [],
        } for k in range(batch_size)]
        for idx, pred_dict in enumerate(pred_dicts):
            batch_hm = pred_dict['hm'].sigmoid()
            batch_center = pred_dict['center']
            batch_center_z = pred_dict['center_z']


            for k, final_dict in enumerate(final_pred_dicts):
                final_dict['pred_labels'] = self.class_id_mapping_each_head[idx][final_dict['pred_labels'].long()]

                if post_process_cfg.get('USE_IOU_TO_RECTIFY_SCORE', False) and 'pred_iou' in final_dict:
                    pred_iou = torch.clamp(final_dict['pred_iou'], min=0, max=1.0)
                    IOU_RECTIFIER = final_dict['pred_scores'].new_tensor(post_process_cfg.IOU_RECTIFIER)
                    final_dict['pred_scores'] = torch.pow(final_dict['pred_scores'], 1 - IOU_RECTIFIER[final_dict['pred_labels']]) * torch.pow(pred_iou, IOU_RECTIFIER[final_dict['pred_labels']])

                if post_process_cfg.NMS_CONFIG.NMS_TYPE not in  ['circle_nms', 'class_specific_nms']:
                    selected, selected_scores = model_nms_utils.class_agnostic_nms(
                        box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=None
                    )

                elif post_process_cfg.NMS_CONFIG.NMS_TYPE == 'class_specific_nms':
                    selected, selected_scores = model_nms_utils.class_specific_nms(
                        box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                        box_labels=final_dict['pred_labels'], nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.NMS_CONFIG.get('SCORE_THRESH', None)
                    )
                elif post_process_cfg.NMS_CONFIG.NMS_TYPE == 'circle_nms':
                    raise NotImplementedError

                final_dict['pred_boxes'] = final_dict['pred_boxes'][selected]
                final_dict['pred_scores'] = selected_scores
                final_dict['pred_labels'] = final_dict['pred_labels'][selected]

                ret_dict[k]['pred_boxes'].append(final_dict['pred_boxes'])
                ret_dict[k]['pred_scores'].append(final_dict['pred_scores'])
                ret_dict[k]['pred_labels'].append(final_dict['pred_labels'])

        for k in range(batch_size):
            ret_dict[k]['pred_boxes'] = torch.cat(ret_dict[k]['pred_boxes'], dim=0)
            ret_dict[k]['pred_scores'] = torch.cat(ret_dict[k]['pred_scores'], dim=0)
            ret_dict[k]['pred_labels'] = torch.cat(ret_dict[k]['pred_labels'], dim=0) + 1

        return ret_dict

    @staticmethod
    def reorder_rois_for_refining(batch_size, pred_dicts):
        num_max_rois = max([len(cur_dict['pred_boxes']) for cur_dict in pred_dicts])
        num_max_rois = max(1, num_max_rois)  # at least one faked rois to avoid error
        pred_boxes = pred_dicts[0]['pred_boxes']

        rois = pred_boxes.new_zeros((batch_size, num_max_rois, pred_boxes.shape[-1]))
        roi_scores = pred_boxes.new_zeros((batch_size, num_max_rois))
        roi_labels = pred_boxes.new_zeros((batch_size, num_max_rois)).long()

        for bs_idx in range(batch_size):
            num_boxes = len(pred_dicts[bs_idx]['pred_boxes'])

            rois[bs_idx, :num_boxes, :] = pred_dicts[bs_idx]['pred_boxes']
            roi_scores[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_scores']
            roi_labels[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_labels']
        return rois, roi_scores, roi_labels

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']
        x = self.shared_conv(spatial_features_2d)
        # print(x.shape)

        # x_start, x_end, y_start, y_end
        # par of 2: [[0, 64, 0, 128], [64, 128, 0, 128]], [[0, 128, 0, 64], [0, 128, 64, 128]]
        #
        #
        # ranges = [[[0, 64, 0, 128], [64, 128, 0, 128]], [[0, 128, 0, 64], [0, 128, 64, 128]], [[0, 64, 0, 64],[0, 64, 64, 128],[64, 128, 0, 64],[64, 128, 64, 128]]]
   

        # 1 2 scale1, 3 4 scale2, 5678 scale3
        ranges = [[0, 64, 0, 128], [64, 128, 0, 128], [0, 128, 0, 64], [0, 128, 64, 128], [0, 64, 0, 64],[0, 64, 64, 128],[64, 128, 0, 64],[64, 128, 64, 128]]
   
        pred_dicts = []
        # # print(self.heads_list)
        for head in self.heads_list:
            head_output= []
            for coor_range in ranges:
                head_output.append(head(x[:, :, coor_range[0]:coor_range[1], coor_range[2]:coor_range[3]]))
            pred_dicts.append(head_output)
            
        if self.training:
            target_dict = self.assign_targets(
                ranges, data_dict['gt_boxes'], feature_map_size=spatial_features_2d.size()[2:],
                feature_map_stride=data_dict.get('spatial_features_2d_strides', None)
            )
            self.forward_ret_dict['target_dicts'] = target_dict

        self.forward_ret_dict['pred_dicts'] = pred_dicts
        self.forward_ret_dict['patitions'] = ranges

        if not self.training or self.predict_boxes_when_training:
            pred_dicts = self.forward_ret_dict['pred_dicts']

            target_dicts = self.assign_targets(
                ranges, data_dict['gt_boxes'], feature_map_size=spatial_features_2d.size()[2:],
                feature_map_stride=data_dict.get('spatial_features_2d_strides', None)
            )

            objects = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']

            for idx, pred_dict in enumerate(pred_dicts):
                # pred_dict['hm'] #self.sigmoid(pred_dict['hm'])
                hm_list = []
                gt_list = []
                # print(len(pred_dict))
                for i in range(len(pred_dict)):
                    # print('aaaaaaa')
                    hm = pred_dict[i]['hm']
                    
                    # print(len(target_dicts['masks']))
                    # print(target_dicts['masks'][idx].shape)
                    mask = target_dicts['masks'][idx][:, i, :]

                    hm_list.append(self.sigmoid(hm).tolist())
                    gt_list.append(torch.sum(mask).item())

                    # print(hm.shape)

                obj = objects[idx]
                data_dict['gt_count'] = 0#gt_count
                data_dict['center_count'] = 0#center_count
                # data_dict['hm'] = pred_dict['hm']

                if obj in result.keys():
                    result[obj]['gt_count'].append(0) #gt_count.item()
                    result[obj]['center_count'].append(0)#center_count.item()
                    # result[obj]['hm'].append(pred_dict['hm'].toList())
                else:
                    result[obj] = {'gt_count': [0], 'center_count': [0]} # gt_count.item(), center_count.item()

                file = '/data/hms_partition_multiscale_ep3/{}.txt'.format(self.progress)
                with open(file, 'w') as f: 
                    json.dump(hm_list, f)

                file = '/data/hms_partition_multiscale_ep3/gt_{}.txt'.format(self.progress)
                with open(file, 'w') as f: 
                    json.dump(gt_list, f)
                
                self.progress += 1

        return data_dict

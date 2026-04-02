import torch 
import numpy as np
import torch.nn as nn
from ...utils import box_coder_utils, common_utils, loss_utils
from ...constants import result, gt_list
from functools import partial
from six.moves import map, zip
import pickle
import json


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.
    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.
    Args:
        func (Function): A function that will be applied to a list of
            arguments
    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))

class CounterHeadKitti(nn.Module):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size,
                 predict_boxes_when_training=True):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.class_names = [class_names]
        self.predict_boxes_when_training = predict_boxes_when_training
        self.use_multihead = self.model_cfg.get('USE_MULTIHEAD', False)

        target_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG

        self.target_cfg = target_cfg 
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.progress = 0

        self.forward_ret_dict = {}

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, 8,
            kernel_size=1
        )

        self.loss_cls = GaussianFocalLoss(reduction='mean')

        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def process_nested_list(self, nested_list):
        # Base case: if the current list contains tensors
        if all(isinstance(item, torch.Tensor) for item in nested_list):
            return torch.stack(nested_list)  # Stack the tensors at this level

        # Recursive case: if the current list contains further nested lists
        return torch.stack([self.process_nested_list(sublist) for sublist in nested_list if sublist])  # Recurse and stack
    
    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']
        # print('spatial feature 2d: {}'.format(spatial_features_2d.shape))

        output_tensor = F.max_pool2d(spatial_features_2d, kernel_size=(128, 128))
        # # The shape of output_tensor will be [4, 512, 1, 1], so we can remove the last two dimensions by squeezing
        output_tensor = output_tensor.squeeze(-1).squeeze(-1) 

        # file = '/data/feature_kitti/{}.txt'.format(self.progress)
        # with open(file, 'wb') as f: 
        #     pickle.dump(output_tensor, f)

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            # hm_list = []
            pred_dicts = self.forward_ret_dict['cls_preds']
            # print('pred dicts: ')
            # print(pred_dicts.shape)

            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            gt_heatmaps = self.process_nested_list(targets_dict ['heatmaps']).squeeze()
            # print('gt heatmaps shape: ')
            # print(gt_heatmaps.shape)
            

            for i in range(pred_dicts.size(-1)):
                pred_heatmap = pred_dicts[:, :, :, i]
                # print('pred heatmap shape: ')
                # print(pred_heatmap)

                batch_gt_count = []
                for j in range(gt_heatmaps[:, i, :, :].size(0)):
                    gt_count = gt_heatmaps[j, i, :, :].eq(1).float().sum().item()
                    batch_gt_count.append(gt_count)
                
                # hm_list.append(pred_heatmap.tolist())
                gt_list.append(batch_gt_count)

                # file = '/data/kitti_hm4/{}.txt'.format(self.progress)
                # with open(file, 'w') as f: 
                #     json.dump(pred_heatmap.tolist(), f)
                self.progress += 1


            # gt_file = '/data/kitti_gt_list3.txt'
            # with open(gt_file, 'w') as f: 
            #     json.dump(gt_list, f)
            

            
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=None
            )
    
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict

    def _gather_feat(self, feat, ind, mask=None):
        """Gather feature map.

        Given feature map and index, return indexed feature map.

        Args:
            feat (torch.tensor): Feature map with the shape of [B, H*W, 10].
            ind (torch.Tensor): Index of the ground truth boxes with the
                shape of [B, max_obj].
            mask (torch.Tensor): Mask of the feature map with the shape
                of [B, max_obj]. Default: None.

        Returns:
            torch.Tensor: Feature map after gathering with the shape
                of [B, max_obj, 10].
        """
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def assign_targets(self, gt_boxes):
        """Generate targets.

        Args:
            gt_boxes: (B, M, 8) box + cls 

        Returns:
            Returns:
                tuple[list[torch.Tensor]]: Tuple of target including \
                    the following results in order.

                    - list[torch.Tensor]: Heatmap scores.
                    - list[torch.Tensor]: Ground truth boxes.
                    - list[torch.Tensor]: Indexes indicating the \
                        position of the valid boxes.
                    - list[torch.Tensor]: Masks indicating which \
                        boxes are valid.
        """
        gt_bboxes_3d, gt_labels_3d = gt_boxes[..., :-1], gt_boxes[..., -1]

        heatmaps, anno_boxes, inds, masks = multi_apply(
            self.get_targets_single, gt_bboxes_3d, gt_labels_3d)
        # transpose heatmaps, because the dimension of tensors in each task is
        # different, we have to use numpy instead of torch to do the transpose.
        # heatmaps = np.array(heatmaps).transpose(1, 0).tolist()

        if type(heatmaps) != list:
            heatmaps = heatmaps.cpu().numpy().transpose(1, 0).tolist()
        heatmaps = [torch.stack(hms_) for hms_ in heatmaps]
        # transpose anno_boxes
        if type(anno_boxes) != list:
            anno_boxes = anno_boxes.cpu().numpy().transpose(1, 0).tolist()
            # anno_boxes = np.array(anno_boxes).transpose(1, 0).tolist()
        anno_boxes = [torch.stack(anno_boxes_) for anno_boxes_ in anno_boxes]
        # transpose inds
        if type(inds) != list:
            inds = inds.cpu().numpy().transpose(1, 0).tolist()
        # inds = np.array(inds).transpose(1, 0).tolist()
        inds = [torch.stack(inds_) for inds_ in inds]

        # transpose inds
        if type(masks) != list:
            masks = masks.cpu().numpy().transpose(1, 0).tolist()
        # masks = np.array(masks).transpose(1, 0).tolist()
        masks = [torch.stack(masks_) for masks_ in masks]

        
        all_targets_dict = {
            'heatmaps': heatmaps,
            'anno_boxes': anno_boxes,
            'inds': inds,
            'masks': masks
        }
        
        return all_targets_dict

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.

        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including \
                the following results in order.

                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position \
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes \
                    are valid.
        """
        device = gt_labels_3d.device
        """gt_bboxes_3d = torch.cat(
            (gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]),
            dim=1).to(device)
        """

        max_objs = self.target_cfg.MAX_OBJS
        grid_size = torch.tensor(self.grid_size)
        pc_range = torch.tensor(self.point_cloud_range)
        voxel_size = torch.tensor(self.target_cfg.VOXEL_SIZE)

        feature_map_size = grid_size[:2] // self.target_cfg.OUT_SIZE_FACTOR

        """
        # reorganize the gt_dict by tasks
        task_masks = []
        flag = 0
        for class_name in self.class_names:
            print(gt_labels_3d)
            task_masks.append([
                torch.where(gt_labels_3d == class_name.index(i) + flag)
                for i in class_name
            ])
            flag += len(class_name)

        task_boxes = []
        task_classes = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            for m in mask:
                task_box.append(gt_bboxes_3d[m])
                # 0 is background for each task, so we need to add 1 here.
                task_class.append(gt_labels_3d[m] - flag2)
            task_boxes.append(torch.cat(task_box, axis=0).to(device))
            task_classes.append(torch.cat(task_class).long().to(device))
            flag2 += len(mask)
        """

        task_boxes = [gt_bboxes_3d]
        task_classes = [gt_labels_3d]

        draw_gaussian = draw_heatmap_gaussian
        heatmaps, anno_boxes, inds, masks = [], [], [], []

        for idx in range(1):
            heatmap = gt_bboxes_3d.new_zeros(
                (len(self.class_names[idx]), feature_map_size[1],
                 feature_map_size[0]))

            anno_box = gt_bboxes_3d.new_zeros((max_objs, 8),
                                              dtype=torch.float32)

            ind = gt_labels_3d.new_zeros((max_objs), dtype=torch.int64)
            mask = gt_bboxes_3d.new_zeros((max_objs), dtype=torch.uint8)

            num_objs = min(task_boxes[idx].shape[0], max_objs)

            for k in range(num_objs):
                cls_id = (task_classes[idx][k] - 1).int()

                width = task_boxes[idx][k][3]
                length = task_boxes[idx][k][4]
                width = width / voxel_size[0] / self.target_cfg.OUT_SIZE_FACTOR
                length = length / voxel_size[1] / self.target_cfg.OUT_SIZE_FACTOR

                if width > 0 and length > 0:
                    radius = gaussian_radius(
                        (length, width),
                        min_overlap=self.target_cfg.GAUSSIAN_OVERLAP)
                    radius = max(self.target_cfg.MIN_RADIUS, int(radius))

                    # be really careful for the coordinate system of
                    # your box annotation.
                    x, y, z = task_boxes[idx][k][0], task_boxes[idx][k][
                        1], task_boxes[idx][k][2]

                    coor_x = (
                        x - pc_range[0]
                    ) / voxel_size[0] / self.target_cfg.OUT_SIZE_FACTOR
                    coor_y = (
                        y - pc_range[1]
                    ) / voxel_size[1] / self.target_cfg.OUT_SIZE_FACTOR

                    center = torch.tensor([coor_x, coor_y],
                                          dtype=torch.float32,
                                          device=device)
                    center_int = center.to(torch.int32)

                    # throw out not in range objects to avoid out of array
                    # area when creating the heatmap
                    if not (0 <= center_int[0] < feature_map_size[0]
                            and 0 <= center_int[1] < feature_map_size[1]):
                        continue

                    draw_gaussian(heatmap[cls_id], center_int, radius)

                    new_idx = k
                    x, y = center_int[0], center_int[1]

                    assert (y * feature_map_size[0] + x <
                            feature_map_size[0] * feature_map_size[1])

                    ind[new_idx] = y * feature_map_size[0] + x
                    mask[new_idx] = 1
                    rot = task_boxes[idx][k][6]
                    box_dim = task_boxes[idx][k][3:6]
                    box_dim = box_dim.log()
                    anno_box[new_idx] = torch.cat([
                        center - torch.tensor([x, y], device=device),
                        z.unsqueeze(0), box_dim,
                        torch.sin(rot).unsqueeze(0),
                        torch.cos(rot).unsqueeze(0),
                    ])

            heatmaps.append(heatmap)
            anno_boxes.append(anno_box)
            masks.append(mask)
            inds.append(ind)
        return heatmaps, anno_boxes, inds, masks

    def generate_predicted_boxes(self, batch_size, cls_preds, box_preds, dir_cls_preds=None):
        """
        Args:
            batch_size:
            cls_preds: (N, H, W, C1)
            box_preds: (N, H, W, C2)
            dir_cls_preds: (N, H, W, C3)

        Returns:
            batch_cls_preds: (B, num_boxes, num_classes)
            batch_box_preds: (B, num_boxes, 7+C)

        """
        batch, H, W, code_size = box_preds.size()
        box_preds = box_preds.reshape(batch, H*W, code_size)

        batch_reg = box_preds[..., 0:2]
        batch_hei = box_preds[..., 2:3]

        batch_dim = torch.exp(box_preds[..., 3:6])

        batch_rots = box_preds[..., 6:7]
        batch_rotc = box_preds[..., 7:8]

        ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
        ys = ys.view(1, H, W).repeat(batch, 1, 1).to(cls_preds.device)
        xs = xs.view(1, H, W).repeat(batch, 1, 1).to(cls_preds.device)

        xs = xs.view(batch, -1, 1) + batch_reg[:, :, 0:1]
        ys = ys.view(batch, -1, 1) + batch_reg[:, :, 1:2]

        xs = xs * self.target_cfg.OUT_SIZE_FACTOR * self.target_cfg.VOXEL_SIZE[0] + self.point_cloud_range[0]
        ys = ys * self.target_cfg.OUT_SIZE_FACTOR * self.target_cfg.VOXEL_SIZE[1] + self.point_cloud_range[1]

        rot = torch.atan2(batch_rots, batch_rotc)

        batch_box_preds = torch.cat([xs, ys, batch_hei, batch_dim, rot], dim=2)

        batch_cls_preds = cls_preds.view(batch, H*W, -1)
        return batch_cls_preds, batch_box_preds

    def get_loss(self):
        # print('heatmap: ')
        # print(self.forward_ret_dict['cls_preds'].shape)
        # print('length of mask: {}'.format(len(self.forward_ret_dict['masks'])))
        # print('mask: ')
        # print(self.forward_ret_dict['masks'][0].shape)

        cls_loss, tb_dict = self.get_cls_layer_loss()
        # box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        counter_loss = self.get_counter_loss()
        # tb_dict.update(tb_dict_box)
        rpn_loss = cls_loss + counter_loss

        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict

    def sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y
    
    def extract_centers2(self, heatmap, threshold=0.5, radius=2):
        device = heatmap.device
        batch, c, h, w = heatmap.shape
        all_centers = []

        for b in range(batch):
            batch_centers = []
            for ch in range(c):
                heatmap_bch = heatmap[b, ch, :, :]

                # print(heatmap_bch.shape)

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
    
    def get_counter_loss(self):
        # pred_dicts = self.forward_ret_dict['pred_dicts']
        # target_dicts = self.forward_ret_dict['target_dicts']

        # heatmaps = self.sigmoid(self.forward_ret_dict['cls_preds']).permute(0, 3, 1, 2) 
        heatmaps = self.forward_ret_dict['cls_preds'].permute(0, 3, 1, 2) 
        masks = self.forward_ret_dict['masks']

        center_count, centers = self.extract_centers2(heatmaps, threshold=0.5, radius=5)
        # print(heatmaps.shape)
        # print('center_count: {}'.format(center_count))
        # print(masks[0])
        center_count = torch.sum(center_count)
        gt_count = torch.sum(torch.stack(masks)).to('cuda')

        # print('center_count: {}'.format(center_count))
        # print('gt_count: {}'.format(gt_count))

        count_loss = torch.abs(gt_count - center_count)

        return count_loss
    
    def get_cls_layer_loss(self):
        # NHWC -> NCHW 
        pred_heatmaps = clip_sigmoid(self.forward_ret_dict['cls_preds']).permute(0, 3, 1, 2) 
        gt_heatmaps =  self.forward_ret_dict['heatmaps'][0]
        num_pos = gt_heatmaps.eq(1).float().sum().item()

        cls_loss = self.loss_cls(
                pred_heatmaps,
                gt_heatmaps,
                avg_factor=max(num_pos, 1))

        cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        tb_dict = {
            'rpn_loss_cls': cls_loss.item()
        }
        return cls_loss, tb_dict


    def get_box_reg_layer_loss(self):
        # Regression loss for dimension, offset, height, rotation
        target_box, inds, masks = self.forward_ret_dict['anno_boxes'][0], self.forward_ret_dict['inds'][0], self.forward_ret_dict['masks'][0]

        ind = inds
        num = masks.float().sum()
        pred = self.forward_ret_dict['box_preds'] # N x (HxW) x 7 
        pred = pred.view(pred.size(0), -1, pred.size(3))
        pred = self._gather_feat(pred, ind)
        mask = masks.unsqueeze(2).expand_as(target_box).float()
        isnotnan = (~torch.isnan(target_box)).float()
        mask *= isnotnan

        code_weights = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights']
        bbox_weights = mask * mask.new_tensor(code_weights)
        
        loc_loss = l1_loss(
            pred, target_box, bbox_weights, avg_factor=(num + 1e-4))

        loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
        box_loss = loc_loss
        tb_dict = {
            'rpn_loss_loc': loc_loss.item()
        }

        return box_loss, tb_dict

"""
The following is some util files, we will move it to separate files later
"""

import numpy as np
import torch

def clip_sigmoid(x, eps=1e-4):
    """Sigmoid function for input feature.

    Args:
        x (torch.Tensor): Input feature map with the shape of [B, N, H, W].
        eps (float): Lower bound of the range to be clamped to. Defaults
            to 1e-4.

    Returns:
        torch.Tensor: Feature map after sigmoid.
    """
    y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
    return y

def gaussian_2d(shape, sigma=1):
    """Generate gaussian map.

    Args:
        shape (list[int]): Shape of the map.
        sigma (float): Sigma to generate gaussian map.
            Defaults to 1.

    Returns:
        np.ndarray: Generated gaussian map.
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_heatmap_gaussian(heatmap, center, radius, k=1):
    """Get gaussian masked heatmap.

    Args:
        heatmap (torch.Tensor): Heatmap to be masked.
        center (torch.Tensor): Center coord of the heatmap.
        radius (int): Radius of gausian.
        K (int): Multiple of masked_gaussian. Defaults to 1.

    Returns:
        torch.Tensor: Masked heatmap.
    """
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = torch.from_numpy(
        gaussian[radius - top:radius + bottom,
                 radius - left:radius + right]).to(heatmap.device,
                                                   torch.float32)
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def gaussian_radius(det_size, min_overlap=0.5):
    """Get radius of gaussian.

    Args:
        det_size (tuple[torch.Tensor]): Size of the detection result.
        min_overlap (float): Gaussian_overlap. Defaults to 0.5.

    Returns:
        torch.Tensor: Computed radius.
    """
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = torch.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = torch.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = torch.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


"""
Gaussian Loss 
"""
class GaussianFocalLoss(nn.Module):
    """GaussianFocalLoss is a variant of focal loss.

    More details can be found in the `paper
    <https://arxiv.org/abs/1808.01244>`_
    Code is modified from `kp_utils.py
    <https://github.com/princeton-vl/CornerNet/blob/master/models/py_utils/kp_utils.py#L152>`_  # noqa: E501
    Please notice that the target in GaussianFocalLoss is a gaussian heatmap,
    not 0/1 binary target.

    Args:
        alpha (float): Power of prediction.
        gamma (float): Power of target for negtive samples.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self,
                 alpha=2.0,
                 gamma=4.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(GaussianFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction
                in gaussian distribution.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_reg = self.loss_weight * gaussian_focal_loss(
            pred,
            target,
            weight,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss_reg

import functools

import torch.nn.functional as F


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    avg_factor=None, **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, avg_factor=2)
    tensor(1.5000)
    """

    @functools.wraps(loss_func)
    def wrapper(pred,
                target,
                weight=None,
                reduction='mean',
                avg_factor=None,
                **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper


@weighted_loss
def gaussian_focal_loss(pred, gaussian_target, alpha=2.0, gamma=4.0):
    """`Focal Loss <https://arxiv.org/abs/1708.02002>`_ for targets in gaussian
    distribution.

    Args:
        pred (torch.Tensor): The prediction.
        gaussian_target (torch.Tensor): The learning target of the prediction
            in gaussian distribution.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 2.0.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 4.0.
    """
    eps = 1e-12
    pos_weights = gaussian_target.eq(1)
    neg_weights = (1 - gaussian_target).pow(gamma)
    pos_loss = -(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights
    neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights
    return pos_loss + neg_loss

@weighted_loss
def l1_loss(pred, target):
    """L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.

    Returns:
        torch.Tensor: Calculated loss
    """
    assert pred.size() == target.size() and target.numel() > 0
    loss = torch.abs(pred - target)
    return loss
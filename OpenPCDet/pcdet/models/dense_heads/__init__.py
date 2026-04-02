from .anchor_head_multi import AnchorHeadMulti
from .anchor_head_single import AnchorHeadSingle
from .anchor_head_template import AnchorHeadTemplate
from .point_head_box import PointHeadBox
from .point_head_simple import PointHeadSimple
from .point_intra_part_head import PointIntraPartOffsetHead
from .center_head import CenterHead
from .voxelnext_head import VoxelNeXtHead
from .transfusion_head import TransFusionHead
from .counter_head import CounterHead
from .counter_head_partition import CounterHeadPartition
from .counter_head_partition_multiscale import CounterHeadPartitionMultiScale
from .counter_head_partition_overlap import CounterHeadPartitionOverlap
from .counter_head_partition_exp import CounterHeadPartitionExp
from .center_head_kitti import CenterHeadKitti
from .counter_head_kitti import CounterHeadKitti
from .kitti_counter_head_partition_exp import CounterHeadKittiExp
from .kitti_counter_head_partition_exp import CounterHeadKittiExp
# from .counter_head_partition_waymo import CounterHeadPartitionWaymo
# from .counter_head_partition_overlap_waymo import CounterHeadPartitionOverlapWaymo

from .bev_segmentation_head import BEVSegmentationHead
from .sparse_center_head import SparseCenterHead
from .sparse_transfusion_head import SparseTransFusionHead

__all__ = {
    'AnchorHeadTemplate': AnchorHeadTemplate,
    'AnchorHeadSingle': AnchorHeadSingle,
    'PointIntraPartOffsetHead': PointIntraPartOffsetHead,
    'PointHeadSimple': PointHeadSimple,
    'PointHeadBox': PointHeadBox,
    'AnchorHeadMulti': AnchorHeadMulti,
    'CenterHead': CenterHead,
    'VoxelNeXtHead': VoxelNeXtHead,
    'TransFusionHead': TransFusionHead,
    'CounterHead': CounterHead,
    'CounterHeadPartition': CounterHeadPartition,
    'CounterHeadPartitionMultiScale': CounterHeadPartitionMultiScale,
    'CounterHeadPartitionOverlap': CounterHeadPartitionOverlap,
    'CounterHeadPartitionExp': CounterHeadPartitionExp,
    'CenterHeadKitti': CenterHeadKitti,
    'CounterHeadKitti': CounterHeadKitti,
    'CounterHeadKittiExp': CounterHeadKittiExp,
    # 'CounterHeadPartitionOverlapWaymo': CounterHeadPartitionOverlap,
    'BEVSegmentationHead': BEVSegmentationHead,
    'SparseCenterHead': SparseCenterHead,
    'SparseTransFusionHead': SparseTransFusionHead,
}

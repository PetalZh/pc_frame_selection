Our implementation is based on OpenPCDet(https://github.com/open-mmlab/OpenPCDet/tree/master).

For the evironment setup, please refer to  https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md#

To train the heatmap generator, use the command

```python train.py --cfg_file cfgs/nuscenes_models/heatmap_generator.yaml --epochs=25```

To train the model on different sample rate and sampling method, please run

```python train.py --cfg_file cfgs/nuscenes_models/centerpoint/centerpoint_heatmap.yaml --batch_size=2 --epochs=40  --pretrained_model  /data/checkpoints/your_pretrain_model.pth --sample_rate 0.1 --sample_method heatmap_scene_tree/scene_even_heatmap/heatmap_emd_tree```



To evaluate the result, please run
```bash scripts/dist_test.sh 1 --cfg_file cfgs/nuscenes_models/centerpoint/centerpoint_heatmap.yaml --ckpt  /path_to_model```

Replace the path from nuScenes to Waymo for results of Waymo dataset.

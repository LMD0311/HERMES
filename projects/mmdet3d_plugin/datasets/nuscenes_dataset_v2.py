import copy
from mmdet3d.datasets import NuScenesDataset
import mmcv
from os import path as osp
from mmdet.datasets import DATASETS
import torch
import numpy as np
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from .nuscnes_eval import NuScenesEval_custom
from mmcv.parallel import DataContainer as DC
from collections import defaultdict, OrderedDict
from projects.mmdet3d_plugin.dd3d.datasets.nuscenes import NuscenesDataset as DD3DNuscenesDataset
import copy


@DATASETS.register_module()
class CustomNuScenesDatasetV2(NuScenesDataset):
    def __init__(self, frames=(), mono_cfg=None, overlap_test=False, ego_mask=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frames = frames
        self.queue_length = len(frames)
        self.overlap_test = overlap_test
        self.mono_cfg = mono_cfg
        self.ego_mask = ego_mask
        if not self.test_mode and mono_cfg is not None:
            self.mono_dataset = DD3DNuscenesDataset(**mono_cfg)

    def _mask_points(self, pts):
        assert self.ego_mask is not None
        # remove points belonging to ego vehicle.
        ego_mask = np.logical_and(
            np.logical_and(self.ego_mask[0] <= pts[:, 0],
                           self.ego_mask[2] >= pts[:, 0]),
            np.logical_and(self.ego_mask[1] <= pts[:, 1],
                           self.ego_mask[3] >= pts[:, 1]),
        )
        pts = pts[np.logical_not(ego_mask)]
        return pts

    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        data_queue = OrderedDict()
        input_dict = self.get_data_info(index)
        cur_scene_token = input_dict['scene_token']
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        data_queue[0] = example

        for frame_idx in self.frames:
            chosen_idx = index + frame_idx
            if frame_idx == 0 or chosen_idx < 0 or chosen_idx >= len(self.data_infos):
                continue
            info = self.data_infos[chosen_idx]
            input_dict = self.prepare_input_dict(info)
            # if input_dict['scene_token'] == cur_scene_token:
            #     self.pre_pipeline(input_dict)
            #     example = self.pipeline(input_dict)
            #     data_queue[frame_idx] = example
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)
            data_queue[frame_idx] = example
            if 'MASKED_CAM' in input_dict.keys():
                data_queue[frame_idx]['MASKED_CAM'] = input_dict['MASKED_CAM']

        data_queue = OrderedDict(sorted(data_queue.items()))

        # # TODO: need further check
        # for i in reversed(self.frames):
        #     if (i - 1) not in data_queue and (i - 1) in self.frames:
        #         data_queue[i - 1] = copy.deepcopy(data_queue[i])

        ret = defaultdict(list)
        for i in range(len(data_queue[0]['img'])):
            single_aug_data_queue = {}
            for t in data_queue.keys():
                single_example = {}
                for key, value in data_queue[t].items():
                    single_example[key] = value
                single_aug_data_queue[t] = single_example
            single_aug_data_queue = OrderedDict(sorted(single_aug_data_queue.items()))
            single_aug_sample = self.union2one(single_aug_data_queue)

            for key, value in single_aug_sample.items():
                ret[key].append(value)
        return ret

    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        data_queue = OrderedDict()
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        cur_scene_token = input_dict['scene_token']
        # cur_frame_idx = input_dict['frame_idx']
        ann_info = copy.deepcopy(input_dict['ann_info'])
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        if self.filter_empty_gt:
            if (example is None or ~(example['gt_labels_3d']._data != -1).any()):
                return None
        data_queue[0] = example
        aug_param = copy.deepcopy(example['aug_param']) if 'aug_param' in example else {}

        # frame_idx_to_idx = self.scene_to_frame_idx_to_idx[cur_scene_token]
        for frame_idx in self.frames:
            chosen_idx_ = index + frame_idx
            chosen_idx = max(chosen_idx_, 0)  # the frame before the first frame are set to the first frame
            chosen_idx = min(len(self.data_infos) - 1,
                             chosen_idx)  # the frame after the last frame are set to the last frame
            # if frame_idx == 0 or chosen_idx < 0 or chosen_idx >= len(self.data_infos):
            #     return None
            info = self.data_infos[chosen_idx]
            input_dict = self.prepare_input_dict(info)
            # if input_dict['scene_token'] == cur_scene_token:
            #     input_dict['ann_info'] = copy.deepcopy(ann_info)  # only for pipeline, should never be used
            #     self.pre_pipeline(input_dict)
            #     input_dict['aug_param'] = copy.deepcopy(aug_param)
            #     example = self.pipeline(input_dict)
            #     data_queue[frame_idx] = example
            input_dict['ann_info'] = copy.deepcopy(ann_info)  # only for pipeline, should never be used
            self.pre_pipeline(input_dict)
            input_dict['aug_param'] = copy.deepcopy(aug_param)
            example = self.pipeline(input_dict)
            data_queue[frame_idx] = example

            if 'MASKED_CAM' in input_dict.keys():
                data_queue[frame_idx]['MASKED_CAM'] = input_dict['MASKED_CAM']

        data_queue = OrderedDict(sorted(data_queue.items()))

        # # TODO: need further check
        # for i in reversed(self.frames):
        #     if (i - 1) not in data_queue and (i - 1) in self.frames:
        #         data_queue[i - 1] = copy.deepcopy(data_queue[i])

        return self.union2one(data_queue)

    def union2one(self, queue: dict):
        """
        convert sample queue into one single sample.
        """
        imgs_list = [each['img'].data for each in queue.values()]
        lidar2ego = np.eye(4, dtype=np.float32)
        lidar2ego[:3, :3] = Quaternion(queue[0]['lidar2ego_rotation']).rotation_matrix
        lidar2ego[:3, 3] = queue[0]['lidar2ego_translation']

        egocurr2global = np.eye(4, dtype=np.float32)
        egocurr2global[:3, :3] = Quaternion(queue[0]['ego2global_rotation']).rotation_matrix
        egocurr2global[:3, 3] = queue[0]['ego2global_translation']
        metas_map = {}
        prev_scene_token = None
        prev_pos = None
        prev_angle = None
        for i, each in queue.items():
            metas_map[i] = each['img_metas'].data
            metas_map[i]['timestamp'] = each['timestamp']
            metas_map[i]['MASKED_CAM'] = each['MASKED_CAM']
            if 'aug_param' in each:
                metas_map[i]['aug_param'] = each['aug_param']
            if i == 0:
                metas_map[i]['lidaradj2lidarcurr'] = None
                metas_map[i]['prev_bev_exists'] = False
                start_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                start_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                start_scene_token = metas_map[i]['scene_token']
            else:
                egoadj2global = np.eye(4, dtype=np.float32)
                egoadj2global[:3, :3] = Quaternion(each['ego2global_rotation']).rotation_matrix
                egoadj2global[:3, 3] = each['ego2global_translation']

                lidaradj2lidarcurr = np.linalg.inv(lidar2ego) @ np.linalg.inv(
                    egocurr2global) @ egoadj2global @ lidar2ego
                metas_map[i]['lidaradj2lidarcurr'] = lidaradj2lidarcurr if i < 0 else np.linalg.inv(lidaradj2lidarcurr)
                if i < 0:  # only align the lidar2img matrix for the previous frame
                    for i_cam in range(len(metas_map[i]['lidar2img'])):
                        metas_map[i]['lidar2img'][i_cam] = metas_map[i]['lidar2img'][i_cam] @ np.linalg.inv(
                            lidaradj2lidarcurr)
            # only train the sequence all in the same scene
            if metas_map[i]['scene_token'] != start_scene_token:
                metas_map[i]['prev_bev_exists'] = False
                # prev_scene_token = metas_map[i]['scene_token']
                # prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                # prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][-1] = 0
            else:
                metas_map[i]['prev_bev_exists'] = True
                # tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                # tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] -= start_pos
                metas_map[i]['can_bus'][-1] -= start_angle
                # prev_pos = copy.deepcopy(tmp_pos)
                # prev_angle = copy.deepcopy(tmp_angle)
            if self.ego_mask is not None:
                metas_map[i]["points"] = self._mask_points(metas_map[i]["points"].data)
        queue[0]['img'] = DC(torch.stack(imgs_list), cpu_only=False, stack=True)
        queue[0]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[0]
        return queue

    def prepare_input_dict(self, info):
        # standard protocal modified from SECOND.Pytorch
        # For BEVFormer alignment.
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        lidar2ego[:3, 3] = np.array(info['lidar2ego_translation'])
        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(info['ego2global_rotation']).rotation_matrix
        ego2global[:3, 3] = np.array(info['ego2global_translation'])
        lidar2global_rotation = ego2global[:3, :3] @ lidar2ego[:3, :3]

        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            lidar2ego_translation=info['lidar2ego_translation'],
            lidar2ego_rotation=info['lidar2ego_rotation'],
            lidar2global_rotation=lidar2global_rotation,
            prev=info['prev'],
            next=info['next'],
            scene_token=info['scene_token'],
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp'] / 1e6,
            can_bus=info['can_bus'],
            MASKED_CAM=info['MASKED_CAM'] if 'MASKED_CAM' in info.keys() else -1
        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                                  'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam2img=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))

        rotation = Quaternion(input_dict['ego2global_rotation'])
        translation = input_dict['ego2global_translation']
        can_bus = input_dict['can_bus']
        # Overwrite the canbus pos&rot information.
        can_bus[:3] = translation
        can_bus[3:7] = rotation
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle

        return input_dict

    def filter_crowd_annotations(self, data_dict):
        for ann in data_dict["annotations"]:
            if ann.get("iscrowd", 0) == 0:
                return True
        return False

    def get_data_info(self, index):
        info = self.data_infos[index]
        input_dict = self.prepare_input_dict(info)
        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        if not self.test_mode and self.mono_cfg is not None:
            if input_dict is None:
                return None
            info = self.data_infos[index]
            img_ids = []
            for cam_type, cam_info in info['cams'].items():
                img_ids.append(cam_info['sample_data_token'])

            mono_input_dict, mono_ann_index = [], []
            for i, img_id in enumerate(img_ids):
                tmp_dict = self.mono_dataset.getitem_by_datumtoken(img_id)
                if tmp_dict is not None:
                    if self.filter_crowd_annotations(tmp_dict):
                        mono_input_dict.append(tmp_dict)
                        mono_ann_index.append(i)

            # filter empth annotation
            if len(mono_ann_index) == 0:
                return None

            mono_ann_index = DC(mono_ann_index, cpu_only=True)
            input_dict['mono_input_dict'] = mono_input_dict
            input_dict['mono_ann_idx'] = mono_ann_index
        return input_dict

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        # idx = 2
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:

            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        from nuscenes import NuScenes
        self.nusc = NuScenes(version=self.version, dataroot=self.data_root,
                             verbose=True)

        output_dir = osp.join(*osp.split(result_path)[:-1])

        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        self.nusc_eval = NuScenesEval_custom(
            self.nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=True,
            overlap_test=self.overlap_test,
            data_infos=self.data_infos
        )
        self.nusc_eval.main(plot_examples=0, render_curves=False)
        # record metrics
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        for name in self.CLASSES:
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,
                                      self.ErrNameMapping[k])] = val
        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
        return detail

from mmpose.apis import init_model, inference_topdown
from mmpose.structures import merge_data_samples
import torch
import numpy as np

class PoseEstimator:
    def __init__(self, config_path, checkpoint_path, device='cuda:0'):
        self.model = init_model(config_path, checkpoint_path, device=device)

    def estimate(self, frame, instances):
        """
        对跟踪得到的实例执行姿态估计。

        参数：
            frame: np.ndarray (BGR格式图像)
            instances: List[dict]，每个dict含有：
                - 'bbox': [x1, y1, x2, y2]
                - 'id': track_id

        返回：
            List[dict]，每个dict包含：
                - 'id': track_id
                - 'bbox': bbox原始坐标
                - 'pose': List[List[x, y, score]]
        """
        if not instances:
            return []

        bboxes = np.array([inst['bbox'] for inst in instances])

        # 调用 mmpose 的 inference_topdown，返回一个 PoseDataSample 列表
        pose_results = inference_topdown(self.model, frame, bboxes, bbox_format='xyxy')

        # 合并样本，方便统一处理
        data_sample = merge_data_samples(pose_results)

        keypoints = data_sample.pred_instances.keypoints

        if isinstance(keypoints, torch.Tensor):
            keypoints = keypoints.cpu().numpy()

        # 给实例添加姿态结果
        for i, inst in enumerate(instances):
            inst['pose'] = keypoints[i].tolist()

        return instances

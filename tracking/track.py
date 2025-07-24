import numpy as np 
from deep_sort_realtime.deepsort_tracker import DeepSort 

class DeepSortTracker: 

    def __init__(self, max_age=30, n_init=3): 

        """ 
        初始化 DeepSort 跟踪器 
        Args: 

            max_age (int): 最大丢失帧数，超过则删除track 

            n_init (int): 初始化确认所需连续帧数 

        """ 
        self.deepsort = DeepSort(max_age=max_age, n_init=n_init) 

    def update(self, frame, detections): 

        inputs = [] 

        for det in detections: 

            x1, y1, x2, y2 = det['bbox'] 

            w, h = x2 - x1, y2 - y1 

            conf = det['conf'] 

            # 关键修改：直接将整个det字典作为额外数据传递 

            inputs.append(([x1, y1, w, h], conf, det))  # 第三个参数传递完整det字典 

    

        tracks = self.deepsort.update_tracks(inputs, frame=frame) 

    

        results = [] 

        for track in tracks: 

            if not track.is_confirmed(): 

                continue 

                

            track_id = track.track_id 

            l, t, r, b = track.to_ltrb() 

            

            # 关键修改：直接从track获取原始检测数据 

            det_info = {} 

            if hasattr(track, "last_detection") and track.last_detection: 

                # 确保我们能获取到原始检测数据 

                det_info = track.last_detection[2]  # 第三个元素是我们传递的det字典 

            

            # 构造结果时优先使用原始检测数据中的字段 

            output = { 

                'id': str(track_id), 

                'bbox': [int(l), int(t), int(r), int(b)], 

                # 从原始检测继承所有字段（包括team_id） 

                **{k: v for k, v in det_info.items() if k not in ['bbox', 'conf']} 

            } 

            

            # 确保conf存在 

            if 'conf' in det_info: 

                output['conf'] = det_info['conf'] 

                

            results.append(output) 

    

        return results 
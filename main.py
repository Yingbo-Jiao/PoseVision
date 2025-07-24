def analyze_video(video_path, output_json_path,
                  yolo_weights,
                  pose_config,
                  pose_checkpoint):
    import cv2
    import json
    import os
    from tqdm import tqdm

    from yolo_detection.detector import YOLODetector
    # from court.court import Court
    from tracking.track import DeepSortTracker
    from pose.pose_estimator import PoseEstimator
    from deep_sort_realtime.deepsort_tracker import DeepSort
    from classify.classify import TeamClassifier

    # 打开视频并提取第一帧
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, first_frame = cap.read()
    if not ret:
        print("[ERROR] 无法读取视频第一帧")
        return

    # 保存第一帧作为场地图
    # court_image_path = "temp_court_frame.jpg"
    # cv2.imwrite(court_image_path, first_frame)

    # ===== 初始化模块 =====
    detector = YOLODetector(model_path=yolo_weights)
    # court = Court(reference_image_path=court_image_path)

    team_classifier = TeamClassifier(
        autoencoder_json_path=r'C:\Users\Yingbo.Jiao22\Desktop\PoseVision\classify\project-RG\models\convautoencodermodel_10.json',
        autoencoder_weights_path=r'C:\Users\Yingbo.Jiao22\Desktop\PoseVision\classify\project-RG\models\convautoencodermodel_10.h5',
        sample_video_path=video_path
    )

    player_tracker = DeepSortTracker(max_age=30, n_init=3)
    ball_tracker = DeepSortTracker(max_age=5, n_init=2)

    pose_estimator = PoseEstimator(
        config_path=pose_config,
        checkpoint_path=pose_checkpoint
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置到第一帧
    results = {}
    classified_data = {}  # 新增：用于存储分类数据

    for frame_idx in tqdm(range(total_frames), desc="Processing"):
        ret, frame = cap.read()
        if not ret:
            break

        dets = detector.detect_image(frame)
        ## filtered = court.filter_detections_by_polygon(dets, frame.shape)

        player_dets, ball_dets = [], []
        for det in dets:
            if det[5] == 0:
                player_dets.append({
                    'bbox': [det[0], det[1], det[2], det[3]],
                    'conf': det[4],
                    'class_id': int(det[5])
                })
            elif det[5] == 1:
                ball_dets.append({
                    'bbox': [det[0], det[1], det[2], det[3]],
                    'conf': det[4],
                    'class_id': int(det[5])
                })

        classified_players = team_classifier.classify_frame(frame, player_dets)
        tracked_players = player_tracker.update(frame, classified_players)
        tracked_balls = ball_tracker.update(frame, ball_dets)
        players_with_pose = pose_estimator.estimate(frame, tracked_players)

        # 保存主结果
        results[f'frame_{frame_idx+1}'] = {
            'players': players_with_pose,
            'balls': tracked_balls
        }

        # 新增：保存分类数据
        classified_data[f'frame_{frame_idx+1}'] = {
            'classified_players': classified_players,
            'original_player_dets': player_dets  # 可选：原始检测数据
        }

    cap.release()

    # if os.path.exists(court_image_path):
    #    os.remove(court_image_path)

    # 保存主结果文件
    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] 主分析结果保存至: {output_json_path}")

    # 新增：保存分类数据文件
    base_dir = os.path.dirname(output_json_path)
    classified_json_path = os.path.join(base_dir, "classified_results.json")
    with open(classified_json_path, 'w') as f:
        json.dump(classified_data, f, indent=2)
    print(f"[INFO] 分类数据已另存为: {classified_json_path}")



if __name__ == "__main__":
    analyze_video(
        video_path=r'C:\Users\Yingbo.Jiao22\Desktop\PoseVision\input\sample1.mp4',
        output_json_path=r'C:\Users\Yingbo.Jiao22\Desktop\PoseVision\outputanalysis_results.json',
        yolo_weights=r'C:\Users\Yingbo.Jiao22\Desktop\PoseVision\yolo_detection\best.pt',
        pose_config=r'C:\Users\Yingbo.Jiao22\Desktop\PoseVision\configs\body_2d_keypoint\rtmpose\body8\rtmpose-s_8xb256-420e_body8-256x192.py',
        pose_checkpoint=r'C:\Users\Yingbo.Jiao22\Desktop\PoseVision\configs\rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth'
    )

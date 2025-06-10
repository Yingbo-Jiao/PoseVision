from shapely.geometry import Point, Polygon

def filter_detections_by_roi(detections, roi_points):
    """
    过滤检测框,只保留下边框中点在多边形ROI内的目标

    参数:
        detections (dict): 目标检测结果列表,每个dict含 'bbox' 键，格式为[x1, y1, x2, y2]
        roi_points (tuples): ROI四边形顶点列表 [(x1, y1), (x2, y2), ...]

    返回:
        dict: 过滤后有效的目标列表，如果没有返回空列表
    """
    roi_polygon = Polygon(roi_points)
    valid_detections = []

    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        center_bottom = Point((x1 + x2) / 2, y2)  

        if roi_polygon.contains(center_bottom):
            valid_detections.append(det)

    return valid_detections

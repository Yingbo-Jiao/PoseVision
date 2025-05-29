import json

def parse_labelme(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    shapes = data['shapes']
    corners = []
    for shape in shapes:
        if shape['shape_type'] == 'polygon':
            corners = shape['points']
    return corners



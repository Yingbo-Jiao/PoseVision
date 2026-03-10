import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.patches as patches
import matplotlib.animation as animation

# --------- 1. 输入路径 ---------
video_path = r'C:\Users\Yingbo.Jiao\Desktop\sample1.mp4'
json_path = r"C:\Users\Yingbo.Jiao\Desktop\PoseVision\classified_results.json"
output_json = "transformed_positions_fullcourt.json"
frame_index = 10  # 取第10帧手动标注球场四点

# --------- 2. 抽取视频帧用于选点 ---------
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"无法打开视频文件: {video_path}")
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if frame_index >= total_frames:
    raise ValueError(f"指定帧号超出范围，总帧数为 {total_frames}")
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
ret, frame = cap.read()
if not ret:
    raise ValueError(f"无法读取第 {frame_index} 帧")

# --------- 3. 选取球场四个点 ---------
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.title("请按顺序点击球场四点：\n1.左下角  2.左上角  3.右上角  4.右下角")
src_points = np.array(plt.ginput(4, timeout=0), dtype=np.float32)
plt.close()

print("您选择的点坐标：")
for i, pt in enumerate(src_points):
    print(f"点 {i+1}: ({pt[0]:.1f}, {pt[1]:.1f})")

# --------- 4. 全场目标平面坐标 (mm) ---------
dst_points = np.array([
    [0, 1500],     # 左下角
    [0, 0],        # 左上角
    [2800, 0],     # 右上角
    [2800, 1500]   # 右下角
], dtype=np.float32)

# --------- 5. 计算单应矩阵 H ---------
H = cv2.getPerspectiveTransform(src_points, dst_points)
print("\n单应矩阵 H:")
print(H)

# --------- 6. 加载球员检测数据 ---------
with open(json_path, "r") as f:
    data = json.load(f)

# 转换成 dict 格式 {frame_x: {...}}
if isinstance(data, list):
    data_dict = {}
    for item in data:
        frame_id = f"frame_{item['frame']}"
        data_dict[frame_id] = {"classified_players": []}
        for det in item["players"]:
            data_dict[frame_id]["classified_players"].append({
                "bbox": det["bbox"],
                "team_id": det.get("team", -1),
                "track_id": det.get("track_id", -1)
            })
    data = data_dict

# --------- 7. 映射所有帧到全场坐标 ---------
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()
fps = fps or 60

transformed_data = {
    "court_dimensions": {"width": 2800, "height": 1500},
    "frames": {}
}

frame_ids = sorted(data.keys(), key=lambda x: int(x.split('_')[1]))
for frame_id in frame_ids:
    frame_data = data[frame_id]
    positions = []
    for det in frame_data["classified_players"]:
        x1, y1, x2, y2 = det["bbox"]
        cx = (x1 + x2) / 2
        cy = y2
        original_pt = np.array([[[cx, cy]]], dtype=np.float32)
        mapped_pt = cv2.perspectiveTransform(original_pt, H)[0][0]
        positions.append({
            "player_id": det.get("track_id", -1),
            "team_id": det.get("team_id", -1),
            "court_x": float(mapped_pt[0]),
            "court_y": float(mapped_pt[1]),
            "original_x": float(cx),
            "original_y": float(cy)
        })
    transformed_data["frames"][frame_id] = {"positions": positions}

with open(output_json, "w") as f:
    json.dump(transformed_data, f, indent=2)
print(f"\n✅ 转换后的坐标已保存到: {output_json}")

# =========================================================
# 8. 绘制全场函数
# =========================================================
CBA_LENGTH = 2800
CBA_WIDTH  = 1500
THREE_PT_RADIUS = 6750 / 10
KEY_W      = 4900 / 10
KEY_H      = 5800 / 10
RIM_R      = 450 / 10
RIM_DIST   = 150 / 10
FT_ARC_R   = 1800 / 10

def draw_cba_fullcourt(ax):
    ax.set_facecolor("#f7e1b5")
    ax.set_xlim(0, CBA_LENGTH)
    ax.set_ylim(0, CBA_WIDTH)
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 外边框
    outer = patches.Rectangle((0, 0), CBA_LENGTH, CBA_WIDTH,
                              linewidth=2, edgecolor="black", facecolor="none")
    ax.add_patch(outer)

    # 中线
    ax.plot([CBA_LENGTH/2, CBA_LENGTH/2], [0, CBA_WIDTH], color="black", linewidth=2)

    # 中圈
    center_circle = plt.Circle((CBA_LENGTH/2, CBA_WIDTH/2), FT_ARC_R,
                               fill=False, linewidth=2, color="black")
    ax.add_patch(center_circle)

    # 左右篮筐 + 罚球区 + 三分线
    for side in [0, CBA_LENGTH]:
        if side == 0:  # 左侧
            rim_x = side + RIM_DIST
            key_x = side
            direction = 1
        else:  # 右侧
            rim_x = side - RIM_DIST
            key_x = side - KEY_H
            direction = -1

        # 篮筐
        rim = plt.Circle((rim_x, CBA_WIDTH/2), RIM_R, color="orange", zorder=5)
        ax.add_patch(rim)

        # 罚球区矩形
        paint = patches.Rectangle((key_x, (CBA_WIDTH-KEY_W)/2),
                                  KEY_H, KEY_W, linewidth=2,
                                  edgecolor="black", facecolor="none")
        ax.add_patch(paint)

        # 罚球弧
        ft_arc = patches.Arc((rim_x + direction*(KEY_H-FT_ARC_R), CBA_WIDTH/2),
                             2*FT_ARC_R, 2*FT_ARC_R,
                             angle=0,
                             theta1=270 if side==0 else 90,
                             theta2=90 if side==0 else 270,
                             linewidth=2, color="black")
        ax.add_patch(ft_arc)

        # 三分线直线部分
        corner_y1 = (CBA_WIDTH-KEY_W)/2
        corner_y2 = (CBA_WIDTH+KEY_W)/2
        ax.plot([rim_x, rim_x], [corner_y1, corner_y2], color="black", linewidth=2)

        # 三分弧线
        arc = patches.Arc((rim_x, CBA_WIDTH/2),
                          2*THREE_PT_RADIUS, 2*THREE_PT_RADIUS,
                          angle=0, theta1=22, theta2=338, linewidth=2, color="black")
        ax.add_patch(arc)

def plot_players_on_ax(ax, frame_positions):
    for pos in frame_positions:
        x, y = pos["court_x"], pos["court_y"]
        team = pos.get("team_id", -1)
        if team == 0:
            color = "blue"
        elif team == 1:
            color = "red"
        else:
            color = "gray"
        circle = plt.Circle((x, y), 40, color=color, zorder=10)
        ax.add_patch(circle)
        ax.text(x, y, str(pos["player_id"]),
                color="white", fontsize=8, ha='center', va='center', zorder=11)

# =========================================================
# 9. 合成完整战术板视频
# =========================================================
frame_ids = sorted(transformed_data["frames"].keys(),
                   key=lambda x: int(x.split('_')[1]))

fig, ax = plt.subplots(figsize=(14, 7))
def animate(idx):
    ax.clear()
    frame_id = frame_ids[idx]
    draw_cba_fullcourt(ax)
    plot_players_on_ax(ax, transformed_data["frames"][frame_id]["positions"])
    ax.set_title(f"CBA Full-Court – {frame_id}", fontsize=9)

ani = animation.FuncAnimation(fig, animate,
                              frames=len(frame_ids),
                              interval=1000/fps,
                              blit=False)
ani.save("full_game.mp4", writer='ffmpeg', fps=fps)
plt.close(fig)

print("✅ 完整全场战术板视频已保存为 full_game.mp4")

import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.patches as patches
import matplotlib.animation as animation

# --------- 1. 设置输入路径 ---------
video_path = r'C:\Users\Yingbo.Jiao\Desktop\sample1.mp4'         
json_path = r"C:\Users\Yingbo.Jiao\Desktop\classified_results.json"  # ← 球员检测JSON路径
output_json = "transformed_positions_fixed.json"
frame_index = 10  # 要抽取的视频帧编号（第10帧）

# --------- 2. 读取视频帧 ---------
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
plt.title("请按顺序点击球场四点：\n1.左下角  2.左上角  3.中线上点  4.中线下点")
src_points = np.array(plt.ginput(4, timeout=0), dtype=np.float32)
plt.close()

print("您选择的点坐标：")
for i, pt in enumerate(src_points):
    print(f"点 {i+1}: ({pt[0]:.1f}, {pt[1]:.1f})")

# --------- 4. 半场目标平面坐标（单位 mm） ---------
# 这里只映射半场：宽度1400mm，高度1500mm
dst_points = np.array([
    [0, 1500],    # 左下角
    [0, 0],       # 左上角
    [1400, 0],    # 中线上点
    [1400, 1500]  # 中线下点
], dtype=np.float32)

# --------- 5. 计算单应矩阵 H ---------
H = cv2.getPerspectiveTransform(src_points, dst_points)
print("\n单应矩阵 H:")
print(H)

# --------- 6. 加载球员检测数据 ---------
with open(json_path, "r") as f:
    data = json.load(f)

# --------- 6+. 转换为 dict，key = frame_x ---------
if isinstance(data, list):
    data_dict = {}
    for item in data:
        frame_id = f"frame_{item['frame']}"   # 统一命名，例如 "frame_1"
        data_dict[frame_id] = {"classified_players": []}
        for det in item["players"]:
            data_dict[frame_id]["classified_players"].append({
                "bbox": det["bbox"],
                "team_id": det.get("team", -1),
                "track_id": det.get("track_id", -1)  # 如果没有ID，先给-1
            })
    data = data_dict

# =========================================================
# 7. 处理「全部帧」数据并映射到球场
# =========================================================
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()
fps = fps or 60  # 如果获取失败默认 60

transformed_data = {
    "court_dimensions": {"width": 1400, "height": 1500},
    "frames": {}
}

frame_ids = sorted(data.keys(), key=lambda x: int(x.split('_')[1]))
for frame_id in frame_ids:
    frame_data = data[frame_id]
    positions = []
    for det in frame_data["classified_players"]:
        x1, y1, x2, y2 = det["bbox"]
        cx = (x1 + x2) / 2
        cy = y2  # 底边中点 y

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

# --------- 8. 保存转换后结果 ---------
with open(output_json, "w") as f:
    json.dump(transformed_data, f, indent=2)
print(f"\n✅ 转换后的坐标已保存到: {output_json}")

# =========================================================
# 9. 半场战术板绘图函数
# =========================================================
CBA_HALF_LENGTH = 1400
CBA_HALF_WIDTH  = 1500
THREE_PT_RADIUS = 6750 / 10
KEY_W           = 4900 / 10
KEY_H           = 5800 / 10
RIM_R           = 450 / 10
RIM_C           = (150 / 10, CBA_HALF_WIDTH / 2)
FT_ARC_R        = 1800 / 10

def draw_cba_halfcourt(ax):
    ax.set_facecolor("#f7e1b5")
    ax.set_xlim(0, CBA_HALF_LENGTH)
    ax.set_ylim(0, CBA_HALF_WIDTH)
    ax.set_aspect('equal')
    ax.set_xlabel("Court X (mm)")
    ax.set_ylabel("Court Y (mm)")
    for spine in ax.spines.values():
        spine.set_visible(False)

    paint = patches.Rectangle((0, (CBA_HALF_WIDTH - KEY_W) / 2), KEY_H, KEY_W,
                              facecolor="#f4a582", alpha=0.8, zorder=2)
    ax.add_patch(paint)

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
# 10. 输出全部帧图片 + 视频
# =========================================================
frame_ids = sorted(transformed_data["frames"].keys(),
                   key=lambda x: int(x.split('_')[1]))

# 1) 保存逐帧图片
os.makedirs("frames_img", exist_ok=True)
for frame_id in frame_ids:
    fig, ax = plt.subplots(figsize=(10, 7))
    draw_cba_halfcourt(ax)
    plot_players_on_ax(ax, transformed_data["frames"][frame_id]["positions"])
    ax.set_title(frame_id, fontsize=9)
    fig.savefig(f"frames_img/{frame_id}.jpg", dpi=150)
    plt.close(fig)
print("✅ 全部帧图片已保存到 frames_img/ 文件夹")

# 2) 合成完整视频
fig, ax = plt.subplots(figsize=(10, 7))
def animate(idx):
    ax.clear()
    frame_id = frame_ids[idx]
    draw_cba_halfcourt(ax)
    plot_players_on_ax(ax, transformed_data["frames"][frame_id]["positions"])
    ax.set_title(f"CBA Half-Court – {frame_id}", fontsize=9)

ani = animation.FuncAnimation(fig, animate,
                              frames=len(frame_ids),
                              interval=1000/fps,
                              blit=False)
ani.save("full_game.mp4", writer='ffmpeg', fps=fps)
plt.close(fig)
print("✅ 完整战术板视频已保存为 full_game.mp4")

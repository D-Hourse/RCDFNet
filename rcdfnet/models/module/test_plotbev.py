import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# 3D物体的中心和尺寸
cx, cy = 5, 5  # 物体中心坐标
length, width = 4, 2  # 物体的长宽
theta = np.pi / 6  # 物体绕z轴的旋转角度，30度 = pi/6

# 旋转矩阵（绕z轴旋转）
rotation_matrix = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])

# 原始的长方体角点坐标（相对于物体中心）
corners = np.array([
    [cx - length / 2, cy - width / 2],  # 左下角
    [cx + length / 2, cy - width / 2],  # 右下角
    [cx - length / 2, cy + width / 2],  # 左上角
    [cx + length / 2, cy + width / 2]  # 右上角
])

# 旋转每个角点坐标
rotated_corners = np.dot(corners - np.array([cx, cy]), rotation_matrix) + np.array([cx, cy])

# 输出旋转后的角点
print("旋转后的角点坐标：")
print(rotated_corners)

# 计算旋转后的凸包
hull = ConvexHull(rotated_corners)


# 绘制凸包的边界
def plot_convex_hull(corners, hull, color='blue'):
    for simplex in hull.simplices:
        plt.plot(corners[simplex, 0], corners[simplex, 1], color=color)


# 可视化旋转前后的角点和BEV标注框
plt.figure(figsize=(6, 6))
# 绘制原始矩形
for i in range(len(corners)):
    plt.plot([corners[i, 0], corners[(i + 1) % 4, 0]],
             [corners[i, 1], corners[(i + 1) % 4, 1]], 'r--')  # 红色框表示原始未旋转矩形

# 绘制旋转后的矩形（蓝色框）
plot_convex_hull(rotated_corners, hull, color='blue')  # 蓝色框表示旋转后的凸包


# 获取凸包的最小外接矩形
def get_oriented_bbox(corners, hull):
    # 获取凸包的点
    hull_points = corners[hull.vertices]

    # 计算所有角度
    angles = []
    for i in range(len(hull_points)):
        p1 = hull_points[i]
        p2 = hull_points[(i + 1) % len(hull_points)]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle = np.arctan2(dy, dx)
        angles.append(angle)

    # 计算最小外接矩形
    min_angle = min(angles)
    max_angle = max(angles)

    return min_angle, max_angle


# 获取最小外接矩形的角度
min_angle, max_angle = get_oriented_bbox(rotated_corners, hull)
print(f"最小外接矩形的旋转角度: {min_angle}, {max_angle}")

# 标注说明
plt.text(cx, cy, 'Center', fontsize=12, color='black', ha='center')
plt.text(np.mean(rotated_corners[:, 0]), np.mean(rotated_corners[:, 1]), 'Rotated corners', fontsize=12, color='blue',
         ha='center')

plt.xlim([0, 10])
plt.ylim([0, 10])
plt.gca().set_aspect('equal', adjustable='box')
plt.title("3D Object Projection to BEV View")
plt.grid(True)
plt.show()

import numpy as np
from filterpy.kalman import KalmanFilter
from lib.pipeline.KF_tracker.kf_config import *


class kf_tracker:
    """
    基于卡尔曼滤波器，追踪物体的3D中心位置(x, y, z)。
    状态向量为[x, y, z, vx, vy, vz]，测量值为[x, y, z]。
    """

    def __init__(self, dt=1.0):
        """
        初始化卡尔曼滤波器。
        Args:
            dt (float): 时间步长。
        """
        self.kf = KalmanFilter(dim_x=STATE_DIM, dim_z=MEASUREMENT_DIM)

        # 状态转移矩阵 F
        self.kf.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        # 测量函数 H
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])

        # 过程噪声协方差 Q
        self.kf.Q *= PROCESS_NOISE_COV

        # 测量噪声协方差 R
        self.kf.R *= MEASUREMENT_NOISE_COV

        # 初始状态协方差 P
        self.kf.P *= INITIAL_STATE_COVARIANCE

    def initialize_state(self, initial_xyz):
        """
        用第一个测量值初始化状态向量。
        Args:
            initial_xyz (list or np.ndarray): 目标的初始 [x, y, z] 坐标。
        """
        self.kf.x = np.array([initial_xyz[0], initial_xyz[1], initial_xyz[2], 0, 0, 0])
        print(f"[KF Tracker] 已使用坐标 {initial_xyz} 初始化。")

    def update(self, xyz):
        """
        使用新的测量值更新追踪器状态。
        Args:
            xyz (list or np.ndarray): 新的 [x, y, z] 测量值。
        """
        self.kf.update(np.array(xyz))
        print(f"[KF Tracker] 已使用坐标 {xyz} 更新。")

    def predict(self):
        """
        预测下一个时间步的状态。
        Returns:
            np.ndarray: 预测的状态向量 [x, y, z, vx, vy, vz]。
        """
        self.kf.predict()
        print(f"[KF Tracker] 预测下一位置为 [x,y,z]: {self.kf.x[:3]}")
        return self.kf.x

    def predict_distribution(self, image_shape):
        """
        根据预测的状态和不确定性，给出一个可能的2D图像区域（BBox）。
        这是一个简化的实现，通过在预测的xy位置上增加一个与不确定性相关的边距来创建BBox。

        Args:
            image_shape (tuple): 图像的(height, width)。

        Returns:
            list: 预测的BBox [x1, y1, x2, y2]。
        """
        # 预测的位置不确定性（协方差）
        pos_cov = self.kf.P[:2, :2]
        # 使用标准差作为尺寸的参考（简化处理）
        std_dev_x = np.sqrt(pos_cov[0, 0])
        std_dev_y = np.sqrt(pos_cov[1, 1])

        # 预测的中心点
        center_x, center_y = self.kf.x[0], self.kf.x[1]

        # 基于不确定性估算一个尺寸，这里使用3倍标准差（覆盖约99.7%的概率）
        # 并增加一个基础尺寸，防止不确定性很小时框过小
        width = 6 * std_dev_x + 50
        height = 6 * std_dev_y + 50

        x1 = int(center_x - width / 2)
        y1 = int(center_y - height / 2)
        x2 = int(center_x + width / 2)
        y2 = int(center_y + height / 2)

        # 确保边界框在图像范围内
        h, w = image_shape
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        print(f"[KF Tracker] 预测的2D分布区域 (BBox): {[x1, y1, x2, y2]}")
        return [x1, y1, x2, y2]
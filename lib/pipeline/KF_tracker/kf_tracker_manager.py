from kf_tracker import kf_tracker

class kf_tracker_manager:
    """
    管理kf_tracker的生命周期。
    """
    def __init__(self):
        self.tracker = None
        self.is_initialized = False

    def initialize(self, initial_xyz):
        """
        初始化一个新的kf_tracker实例。
        Args:
            initial_xyz (list or np.ndarray): 目标的初始 [x, y, z] 坐标。
        """
        self.tracker = kf_tracker()
        self.tracker.initialize_state(initial_xyz)
        self.is_initialized = True
        print("[KF Manager] 追踪器已初始化。")

    def process_update(self, xyz):
        """处理更新请求，如果没有初始化则先初始化。"""
        if not self.is_initialized:
            self.initialize(xyz)
        else:
            self.tracker.update(xyz)

    def process_predict(self):
        """处理预测请求。"""
        if self.is_initialized:
            return self.tracker.predict()
        return None

    def get_predicted_bbox(self, image_shape):
        """获取预测的2D分布区域。"""
        if self.is_initialized:
            return self.tracker.predict_distribution(image_shape)
        return None

    def kill(self):
        """
        终止当前的追踪器实例。
        """
        self.tracker = None
        self.is_initialized = False
        print("[KF Manager] 追踪器已被终止。")

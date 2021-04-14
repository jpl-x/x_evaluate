import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from evo.core.metrics import PoseRelation
from evo.core.trajectory import PoseTrajectory3D


class TrajectoryData:
    ape_error_arrays: Dict[PoseRelation, np.ndarray]
    rpe_error_arrays: Dict[PoseRelation, np.ndarray]
    traj_ref: PoseTrajectory3D
    traj_est: PoseTrajectory3D

    def __init__(self):
        self.ape_error_arrays = dict()
        self.rpe_error_arrays = dict()


class PerformanceData:
    """
    Holds computational performance data useful for plots and comparisons
    """
    df_realtime: pd.DataFrame
    rt_factors: np.ndarray


class EKLTPerformanceData:
    events_per_sec: np.ndarray
    events_per_sec_sim: np.ndarray


class EvaluationData:
    name: str
    tags: List[str]
    trajectory_data: Optional[TrajectoryData]
    performance_data: PerformanceData
    eklt_performance_data: Optional[EKLTPerformanceData]

    def __init__(self):
        self.trajectory_data = None
        self.eklt_performance_data = None


class GitInfo:
    branch: str
    last_commit: str
    files_changed: bool

    def __init__(self, branch, last_commit, files_changed):
        self.branch = branch
        self.last_commit = last_commit
        self.files_changed = files_changed


class EvaluationDataSummary:
    data: Dict[str, EvaluationData]
    trajectory_result_table: pd.DataFrame

    x_git_info: GitInfo
    x_vio_ros_git_info: GitInfo

    def __init__(self):
        self.data = dict()

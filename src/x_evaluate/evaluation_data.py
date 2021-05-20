from enum import Enum

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from evo.core.trajectory import PoseTrajectory3D

from x_evaluate.utils import nanrms


class FrontEnd(Enum):
    XVIO = 'XVIO'
    EKLT = 'EKLT'
    EVIO = 'EVIO'

    def __str__(self):
        return self.value


class AlignmentType(Enum):
    Disabled = 1
    PosYaw = 2
    SE3 = 3
    SIM3 = 4


class DistributionSummary:
    N_BINS = 50
    QUANTILES = np.round(np.arange(0, 1.01, 0.05), 2)

    def __init__(self, data: np.ndarray):
        self.n = np.size(data)
        self.mean = np.nanmean(data)
        self.max = np.nanmax(data)
        self.min = np.nanmin(data)
        self.rms = nanrms(data)
        self.std = np.nanstd(data)
        self.nans = np.count_nonzero(np.isnan(data))
        quantiles = np.nanquantile(data, self.QUANTILES)
        # convenience dict
        self.quantiles = dict()
        for i in range(len(self.QUANTILES)):
            self.quantiles[self.QUANTILES[i]] = quantiles[i]
        bins = np.linspace(self.min, self.max, self.N_BINS)
        if issubclass(data.dtype.type, np.integer) and self.max-self.min < self.N_BINS:
            lower = int(self.min)
            upper = int(self.max)
            bins = np.linspace(lower, upper, upper - lower + 1)
        self.hist, self.bins = np.histogram(data, bins=bins)

        if np.all(data > 0):
            bins_log = np.logspace(np.log10(self.min), np.log10(self.max), self.N_BINS)
            self.hist_log, self.bins_log = np.histogram(data, bins_log)


class ErrorType(Enum):
    APE = 1
    RPE = 2


class TrajectoryData:
    traj_ref: PoseTrajectory3D
    traj_est: PoseTrajectory3D
    raw_estimate_t_xyz_wxyz: np.ndarray

    errors: Dict[str, np.ndarray]
    sub_traj_errors: Dict[float, Dict[str, np.ndarray]]

    def __init__(self):
        self.errors = dict()
        self.sub_traj_errors = dict()


class FeatureTrackingData:
    df_x_vio_features: pd.DataFrame

    df_eklt_features: Optional[pd.DataFrame]
    df_eklt_feature_age: Optional[pd.DataFrame]

    def __init__(self):
        self.df_eklt_features = None
        self.df_eklt_feature_age = None


class PerformanceData:
    """
    Holds computational performance data useful for plots and comparisons
    """
    df_realtime: pd.DataFrame
    df_resources: pd.DataFrame
    rt_factors: np.ndarray


class EKLTPerformanceData:
    events_per_sec: np.ndarray
    events_per_sec_sim: np.ndarray
    optimizations_per_sec: np.ndarray
    optimization_iterations: DistributionSummary
    event_processing_times: DistributionSummary


class EvaluationData:
    name: str
    params: Dict
    command: str
    configuration: Dict
    trajectory_data: Optional[TrajectoryData]
    performance_data: PerformanceData
    feature_data: FeatureTrackingData
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
    name: str
    data: Dict[str, EvaluationData]

    trajectory_summary_table: pd.DataFrame

    configuration: Dict
    frontend: FrontEnd

    x_git_info: GitInfo
    x_vio_ros_git_info: GitInfo

    def __init__(self):
        self.data = dict()

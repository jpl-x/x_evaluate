import argparse
import os
from typing import Dict
import cv2
import matplotlib.lines
import numpy as np
import pandas as pd
import tqdm
from matplotlib import pyplot as plt
from x_evaluate.utils import timestamp_to_real_time, timestamp_to_rosbag_time_zero

from x_evaluate.plots import PlotContext, DEFAULT_COLORS
import x_evaluate.plots
from x_evaluate.scriptlets import cache


class SlidingWindowFeatureTracksPlotter:
    TRACK_TYPE_COLOR = {
        "SLAM": DEFAULT_COLORS[0],
        "MSCKF": DEFAULT_COLORS[1],
    }

    def __init__(self, pc: PlotContext, input_folder, input_frames: pd.DataFrame, backend_tracks: pd.DataFrame,
                 df_features: pd.DataFrame, df_realtime: pd.DataFrame, img_width, img_height,
                 sliding_window=0.05, max_tracks=10):
        self._pc = pc
        self._input_frames = input_frames
        self._backend_tracks = backend_tracks
        self._num_features = df_features
        self._num_features['t'] = timestamp_to_rosbag_time_zero(df_features['ts'], df_realtime)
        self._input_folder = input_folder
        self._sliding_window = sliding_window
        # frame_file = os.path.join(input_folder, "frames/" + input_frames.iloc[0]['filename'])
        self._ax = pc.get_axis()
        self._ax.get_xaxis().set_visible(False)
        self._ax.get_yaxis().set_visible(False)

        self._img_width = img_width
        self._img_height = img_height

        # ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1.2, 0.8, 0.8 * float(H) / float(W), 1]))

        # ax.plot_surface(0, X, Y, cstride=1, rstride=1, facecolors=img, linewidth=1, shade=False,
        #                 edgecolor='none')

        # formatting
        # self._ax.set_xlim([0, 0.2])

        # matplotlib artist handles (id -> Artist, t -> Artist)
        self._active_backend_tracks = {}
        self._active_frame = None
        self._active_frame_t = -1.0
        self._text_handle = None

    def plot_till_time(self, t_to):
        t_from = t_to - self._sliding_window

        # df = df[(df['closing_price'] >= 99) & (df['closing_price'] <= 101)]
        current_frame = self._input_frames.loc[self._input_frames.loc[(self._input_frames['t'] <= t_to)]['t'].idxmax()]

        if current_frame['t'] > self._active_frame_t:
            # change the frame
            if self._active_frame is not None:
                self._active_frame.remove()

            self._active_frame_t = current_frame['t']
            frame_file = os.path.join(self._input_folder, "frames/" + current_frame['filename'])
            img = cv2.imread(frame_file)
            img = img.astype(float) / 255 * 2  # // make it brighter
            self._active_frame = self._ax.imshow(img)


        # to_remove = set(self._active_frames.keys()).difference(set(current_frames['t']))
        # to_add = set(current_frames['t']).difference(set(self._active_frames.keys()))
        #
        # for t in to_remove:
        #     self._active_frames[t].remove()
        #     self._active_frames.pop(t)
        # for t in to_add:
        #     frame = current_frames.loc[current_frames['t'] == t].iloc[0]
        #     frame_file = os.path.join(self._input_folder, "frames/" + frame['filename'])
        #     img = cv2.imread(frame_file)
        #     img = img.astype(float) / 255 * 2  # // make it brighter
        #
        #     self._active_frames[t] = handle
        #
        current_backend_tracks = self._backend_tracks.loc[(t_from <= self._backend_tracks['t']) &
                                                          (self._backend_tracks['t'] <= t_to)]

        backend_track_ids = current_backend_tracks.id.unique()
        to_remove = set(self._active_backend_tracks.keys()).difference(set(backend_track_ids))
        to_update = set(self._active_backend_tracks.keys()).intersection(set(backend_track_ids))
        to_add = set(backend_track_ids).difference(set(self._active_backend_tracks.keys()))

        for i in to_remove.union(to_update):
            for e in self._active_backend_tracks[i]:
                e.remove()
            self._active_backend_tracks.pop(i)
        for i in to_update:
            self._active_backend_tracks[i] = self.plot_backend_track(i, t_from, t_to)
        #
        # available_slots = self._max_tracks - len(self._active_backend_tracks)
        # n_new_tracks = min(available_slots, len(to_add))
        #
        for i in to_add:
            self._active_backend_tracks[i] = self.plot_backend_track(i, t_from, t_to)
        #
        # self._ax.set_xlim([t_from, t_to])

        current_row = self._num_features.loc[self._num_features.loc[(self._num_features['t'] <= t_to)]['t'].idxmax()]

        if self._text_handle is not None:
            self._text_handle.remove()

        # for some reason not working properly
        # num_slam_features = int(current_row['num_slam_features'])

        num_slam_features = len(current_backend_tracks.loc[current_backend_tracks.update_type == "SLAM"]['id'].unique())
        slam_features = F"SLAM: {num_slam_features}"



        self._text_handle = self._ax.text(4, 170, slam_features, color=self.TRACK_TYPE_COLOR["SLAM"])

        self._pc.figure.tight_layout()
        self._pc.figure.canvas.draw()


    def plot_backend_track(self, backend_track_id, t_from, t_to):
        margin_back = self._sliding_window
        margin_future = 0.01
        backend_track = self._backend_tracks.loc[(self._backend_tracks.id == backend_track_id) &
                                                 (t_from <= self._backend_tracks['t']) &
                                                 (self._backend_tracks['t'] <= t_to)]

        t, x, y = backend_track[["t", "x_dist", "y_dist"]].to_numpy().T
        color = self.TRACK_TYPE_COLOR[backend_track['update_type'].iloc[0]]
        line_handle = self._ax.plot(x, y, color=color)
        line_handle = line_handle[0]  # flatten from list
        scatter_handle = self._ax.scatter(x[-1], y[-1], color=color)
        return [line_handle, scatter_handle]
        # ax.plot([0, ])


def find_event_track_from_backend_track(df_event_tracks_interpolation, df_backend_tracks, track_id):
    search_key = df_backend_tracks.loc[df_backend_tracks.id == track_id][['t', 'x_dist', 'y_dist']].iloc[1].to_numpy()
    event_tracks = df_event_tracks_interpolation[["interpolated_t", "interpolated_x_dist", "interpolated_y_dist"]]
    diffs = np.linalg.norm((event_tracks - search_key).abs().to_numpy(), axis=1)
    idx = np.argmin(diffs)
    if diffs[idx] > 0.5:
        print("WARNING: closest event tracks seems not to match with backend track")
        print(event_tracks.iloc[idx] - search_key)
    return df_event_tracks_interpolation.loc[idx, "id"]


def main():
    parser = argparse.ArgumentParser(description="Visualizes the output frames provided by ")
    parser.add_argument('--input_folder', required=True, type=str)
    args = parser.parse_args()

    input_folder = args.input_folder

    frames_csv = os.path.join(input_folder, "dumped_frames.csv")
    tracks_csv = os.path.join(input_folder, "xvio_tracks.csv")
    event_tracks_csv = os.path.join(input_folder, "event_tracks.csv")
    event_tracks_interpolation_csv = os.path.join(input_folder, "event_tracks_interpolation.csv")
    if not os.path.exists(frames_csv) or not os.path.exists(tracks_csv):
        print(F"ERROR: no 'dumped_frames.csv' or 'xvio_tracks.csv' found in {input_folder}")
        exit(1)

    df_event_tracks = None
    df_event_tracks_interpolation = None
    if os.path.exists(event_tracks_csv) and os.path.exists(event_tracks_interpolation_csv):
        df_event_tracks = pd.read_csv(event_tracks_csv, delimiter=";")
        df_event_tracks_interpolation = pd.read_csv(event_tracks_interpolation_csv, delimiter=";")

    df_frames_csv = pd.read_csv(frames_csv, delimiter=";")
    df_backend_tracks = pd.read_csv(tracks_csv, delimiter=";")
    df_features = pd.read_csv(os.path.join(input_folder, "features.csv"), delimiter=";")
    df_realtime = pd.read_csv(os.path.join(input_folder, "realtime.csv"), delimiter=";")

    if len(df_frames_csv) <= 0:
        print(F"ERROR: Empty 'dumped_frames.csv' found in {input_folder}")
        exit(1)

    input_frames = df_frames_csv.loc[df_frames_csv.type == 'input_img']

    # Shift time to first frame
    t_0 = input_frames.iloc[0]['t']
    df_backend_tracks['t'] -= t_0
    input_frames['t'] -= t_0

    x_evaluate.plots.use_paper_style_plots = True

    video_writer = cv2.VideoWriter(os.path.join(input_folder, "feature_tracks_2d.avi"), cv2.VideoWriter_fourcc(
        *'DIVX'), 25, (500, 380))

    with PlotContext(base_width_inch=12.8, base_height_inch=7.2) as pc:
        plotter = SlidingWindowFeatureTracksPlotter(pc, input_folder, input_frames, df_backend_tracks,
                                                    df_features, df_realtime, 240, 180, 0.1)
        step = 0.001
        current = 1

        # def press(event):
        #     if event.key == 'escape':
        #         plt.close(pc.figure)
        #     elif event.key == 'right':
        #         press.current += step
        #         plotter.plot_till_time(press.current)
        # press.current = current

        for t in tqdm.tqdm(np.arange(30, 40, step)):
            plotter.plot_till_time(t)
            buffer = np.asarray(pc.figure.canvas.buffer_rgba())
            buffer = buffer.reshape(pc.figure.canvas.get_width_height()[::-1] + (4,))

            # img = cv2.cvtColor(buffer, cv2.COLOR_RGBA cv2.COLOR_RGBA2BGRA)
            # print(img.shape)

            # buffer = cv2.cvtColor(buffer, cv2.COLOR_RGBA2BGRA)
            # buffer = cv2.cvtColor(buffer, cv2.COLOR_BGRA2RGBA)
            buffer = cv2.cvtColor(buffer, cv2.COLOR_RGB2BGR)
            # print(buffer.shape)
            # cv2.imshow("plot", buffer)
            # cv2.waitKey(1)
            video_writer.write(buffer)

    print("Saving video...")
    video_writer.release()


def create_event_tracks_correspondences(df_backend_tracks, df_event_tracks_interpolation):
    backend_to_event_track_id_map = {}
    for i in tqdm.tqdm(df_backend_tracks['id'].unique()):
        j = find_event_track_from_backend_track(df_event_tracks_interpolation, df_backend_tracks, i)
        backend_to_event_track_id_map[i] = j
    return backend_to_event_track_id_map


if __name__ == '__main__':
    main()

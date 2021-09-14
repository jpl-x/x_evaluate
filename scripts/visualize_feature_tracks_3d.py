import argparse
import os
from typing import Dict

import cv2
import numpy as np
import pandas as pd
import tqdm
from matplotlib import pyplot as plt
from x_evaluate.plots import PlotContext, DEFAULT_COLORS
import x_evaluate.plots
from x_evaluate.scriptlets import cache


class SlidingWindowFeatureTracksPlotter:
    TRACK_TYPE_COLOR = {
        "SLAM": DEFAULT_COLORS[0],
        "MSCKF": DEFAULT_COLORS[1],
    }

    def __init__(self, pc: PlotContext, input_folder, input_frames: pd.DataFrame, backend_tracks: pd.DataFrame,
                 event_tracks: pd.DataFrame, backend_to_frontend_ids: Dict, img_width, img_height,
                 sliding_window=0.05, max_tracks=10):
        self._pc = pc
        self._input_frames = input_frames
        self._backend_tracks = backend_tracks
        self._event_tracks = event_tracks
        self._input_folder = input_folder
        self._backend_to_frontend_ids = backend_to_frontend_ids
        # frame_file = os.path.join(input_folder, "frames/" + input_frames.iloc[0]['filename'])
        self._ax = pc.get_axis(projection="3d")

        self._img_width = img_width
        self._img_height = img_height

        surf_x = np.arange(0, img_width)
        surf_y = np.arange(0, img_height)
        self._surf_x, self._surf_y = np.meshgrid(surf_x, surf_y)

        # ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1.2, 0.8, 0.8 * float(H) / float(W), 1]))

        # ax.plot_surface(0, X, Y, cstride=1, rstride=1, facecolors=img, linewidth=1, shade=False,
        #                 edgecolor='none')

        # formatting
        self._ax.view_init(elev=10, azim=-80)
        self._ax.invert_zaxis()
        self._ax.zaxis.set_major_locator(plt.NullLocator())
        self._ax.yaxis.set_major_locator(plt.NullLocator())
        self._ax.set_xlabel("t [s]")
        self._ax.set_ylim([0, img_width - 1])
        self._ax.set_zlim([img_height - 1, 0])
        # self._ax.set_xlim([0, 0.2])

        # matplotlib artist handles (id -> Artist, t -> Artist)
        self._active_backend_tracks = {}
        self._active_event_tracks = {}
        self._active_frames = {}
        self._sliding_window = sliding_window
        self._max_tracks = max_tracks
        self._active_ekf_update_marks = {}

    def plot_till_time(self, t_to):
        t_from = t_to - self._sliding_window

        # df = df[(df['closing_price'] >= 99) & (df['closing_price'] <= 101)]
        current_frames = self._input_frames.loc[(t_from <= self._input_frames['t']) & (self._input_frames['t'] <= t_to)]

        to_remove = set(self._active_frames.keys()).difference(set(current_frames['t']))
        to_add = set(current_frames['t']).difference(set(self._active_frames.keys()))

        for t in to_remove:
            self._active_frames[t].remove()
            self._active_frames.pop(t)
        for t in to_add:
            frame = current_frames.loc[current_frames['t'] == t].iloc[0]
            frame_file = os.path.join(self._input_folder, "frames/" + frame['filename'])

            img = cv2.imread(frame_file)
            img = img.astype(float) / 255 * 2  # // make it brighter

            handle = self._ax.plot_surface(t, self._surf_x, self._surf_y, cstride=1, rstride=1, facecolors=img,
                                           linewidth=1, shade=False, edgecolor='none', alpha=0.5)
            self._active_frames[t] = handle

        # meaning that we draw event tracks and highlight backend tracks:
        if self._event_tracks is not None:
            current_backend_tracks = self._backend_tracks.loc[(t_from <= self._backend_tracks['t']) &
                                                              (self._backend_tracks['t'] <= t_to)]

            backend_track_ids = current_backend_tracks.id.unique()

            to_remove = set(self._active_backend_tracks.keys()).difference(set(backend_track_ids))
            to_update = set(self._active_backend_tracks.keys()).intersection(set(backend_track_ids))
            to_add = set(backend_track_ids).difference(set(self._active_backend_tracks.keys()))

            for i in to_remove.union(to_update):
                j = self._backend_to_frontend_ids[i]
                self._active_backend_tracks[i].remove()
                self._active_backend_tracks.pop(i)
                for e in self._active_event_tracks[j]:
                    e.remove()
                self._active_event_tracks.pop(j)

            for i in to_update:
                j = self._backend_to_frontend_ids[i]
                self._active_backend_tracks[i], self._active_event_tracks[j] = self.plot_event_track(i, j, t_from, t_to)

            available_slots = self._max_tracks - len(self._active_backend_tracks)
            n_new_tracks = min(available_slots, len(to_add))

            for i in np.random.choice(list(to_add), n_new_tracks, replace=False):
                j = self._backend_to_frontend_ids[i]
                self._active_backend_tracks[i], self._active_event_tracks[j] = self.plot_event_track(i, j, t_from, t_to)

            tracks = self._backend_tracks.loc[self._backend_tracks['id'].isin(list(self._active_backend_tracks.keys()))]
            ekf_times = tracks['t'].sort_values().unique()
            ekf_times = ekf_times[ekf_times <= t_to]
            ekf_times = ekf_times[ekf_times >= t_from]

            to_remove = set(self._active_ekf_update_marks.keys()).difference(set(ekf_times))
            to_add = set(ekf_times).difference(set(self._active_ekf_update_marks.keys()))

            for t in to_remove:
                for u in self._active_ekf_update_marks[t]:
                    u.remove()
                self._active_ekf_update_marks.pop(t)
            for t in to_add:

                handle = self._ax.plot([t, t, t, t], [0, self._img_width, self._img_width, 0],
                              [0, 0, self._img_height, self._img_height], color="green", linestyle="--")
                #
                # handle = self._ax.plot_surface(t, X, Y, cstride=1, rstride=1, color="green", alpha=0.1, linewidth=1,
                #                 shade=False, edgecolor='none')
                self._active_ekf_update_marks[t] = handle

            #
            # for t in ekf_times[ekf_times < 0.2]:
            #     ax.plot_surface(t, X, Y, cstride=1, rstride=1, color="green", alpha=0.1, linewidth=1,
            #                     shade=False, edgecolor='none')


        else:
            print("Not implemented")

        self._ax.set_xlim([t_from, t_to])
        self._pc.figure.tight_layout()
        self._pc.figure.canvas.draw()

    def plot_event_track(self, backend_track_id, event_track_id, t_from, t_to):
        margin_back = self._sliding_window
        margin_future = 0.01
        backend_track = self._backend_tracks.loc[(self._backend_tracks.id == backend_track_id) &
                                                 (t_from <= self._backend_tracks['t']) &
                                                 (self._backend_tracks['t'] <= t_to)]

        frontend_track = self._event_tracks.loc[(self._event_tracks.id == event_track_id) &
                                                (t_from - margin_back <= self._event_tracks['t']) &
                                                (self._event_tracks['t'] <= t_to + margin_future)]

        # frontend_track = self._event_tracks.loc[self._event_tracks.id == event_track_id]

        t, x, y = frontend_track[["t", "center_x", "center_y"]].to_numpy().T
        event_track_handle = self._ax.plot(t, x, y, marker="o", color=self.TRACK_TYPE_COLOR[backend_track.iloc[0][
            'update_type']], linewidth=1, markersize=2)

        t, x, y = backend_track[["t", "x_dist", "y_dist"]].to_numpy().T
        backend_track_handle = self._ax.scatter(t, x, y, marker="o", facecolor="green", linewidth=2)
        return backend_track_handle, event_track_handle
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

    if len(df_frames_csv) <= 0:
        print(F"ERROR: Empty 'dumped_frames.csv' found in {input_folder}")
        exit(1)

    input_frames = df_frames_csv.loc[df_frames_csv.type == 'input_img']

    backend_to_event_track_id_map = {}

    if df_event_tracks is not None:
        def create_mapping():
            print("Creating backend -> event tracks correspondences")
            return create_event_tracks_correspondences(df_backend_tracks, df_event_tracks_interpolation)

        backend_to_event_track_id_map = cache(os.path.join(input_folder, "tracks_mapping.pickle"), create_mapping)

    # Shift time to first frame
    t_0 = input_frames.iloc[0]['t']
    df_backend_tracks['t'] -= t_0
    input_frames['t'] -= t_0
    if df_event_tracks is not None:
        df_event_tracks['t'] -= t_0

    frame_file = os.path.join(input_folder, "frames/" + input_frames.iloc[0]['filename'])

    x_evaluate.plots.use_paper_style_plots = True

    video_writer = cv2.VideoWriter("feature_tracks.avi", cv2.VideoWriter_fourcc(*'DIVX'), 25, (1280, 720))

    with PlotContext(base_width_inch=12.8, base_height_inch=7.2) as pc:
        plotter = SlidingWindowFeatureTracksPlotter(pc, input_folder, input_frames, df_backend_tracks, df_event_tracks,
                                                    backend_to_event_track_id_map, 240, 180)
        step = 0.001
        current = 1

        # def press(event):
        #     if event.key == 'escape':
        #         plt.close(pc.figure)
        #     elif event.key == 'right':
        #         press.current += step
        #         plotter.plot_till_time(press.current)
        # press.current = current

        for t in tqdm.tqdm(np.arange(30, 38, step)):
            plotter.plot_till_time(t)
            buffer = np.asarray(pc.figure.canvas.buffer_rgba())
            buffer = buffer.reshape(pc.figure.canvas.get_width_height()[::-1] + (4,))

            # img = cv2.cvtColor(buffer, cv2.COLOR_RGBA cv2.COLOR_RGBA2BGRA)
            # print(img.shape)

            # buffer = cv2.cvtColor(buffer, cv2.COLOR_RGBA2BGRA)
            # buffer = cv2.cvtColor(buffer, cv2.COLOR_BGRA2RGBA)
            buffer = cv2.cvtColor(buffer, cv2.COLOR_RGB2BGR)
            # cv2.imshow("plot", buffer)
            # cv2.waitKey(1)
            video_writer.write(buffer)

        # pc.figure.canvas.mpl_connect('key_press_event', press)

        # plotter.plot_till_time(current)
        #
        # ax = pc.get_axis(projection="3d")
        # img = cv2.imread(frame_file)
        # img = img.astype(float) / 255
        #
        # H, W, _ = img.shape
        # X = np.arange(0, W)
        # Y = np.arange(0, H)
        # X, Y = np.meshgrid(X, Y)
        #
        # # ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1.2, 0.8, 0.8 * float(H) / float(W), 1]))
        #
        # ax.plot_surface(0, X, Y, cstride=1, rstride=1, facecolors=img, linewidth=1, shade=False,
        #                 edgecolor='none')
        #
        # # formatting
        # ax.view_init(elev=10, azim=-80)
        # ax.invert_zaxis()
        # ax.zaxis.set_major_locator(plt.NullLocator())
        # ax.yaxis.set_major_locator(plt.NullLocator())
        # ax.set_xlabel("t [s]")
        # ax.set_ylim([0, W - 1])
        # ax.set_zlim([H - 1, 0])
        # ax.set_xlim([0, 0.2])
        #
        # track_ids = df_backend_tracks['id'].unique()[:5]
        #
        # for i in track_ids:
        #     backend_track = df_backend_tracks.loc[df_backend_tracks.id == i]
        #     if df_event_tracks is not None:
        #         frontend_track = df_event_tracks.loc[df_event_tracks.id == backend_to_event_track_id_map[i]]
        #         t, x, y = frontend_track[["t", "center_x", "center_y"]].to_numpy().T
        #         ax.plot(t, x, y, marker="o", color=feature_type_to_color[backend_track.iloc[0]['update_type']],
        #                 linewidth=1, markersize=2)
        #
        #     t, x, y = backend_track[["t", "x_dist", "y_dist"]].to_numpy().T
        #     ax.scatter(t, x, y, marker="o", facecolor="green", linewidth=2)
        #     # ax.plot([0, ])
        #
        # ekf_times = df_backend_tracks['t'].sort_values().unique()
        #
        # for t in ekf_times[ekf_times < 0.2]:
        #     ax.plot_surface(t, X, Y, cstride=1, rstride=1, color="green", alpha=0.1, linewidth=1,
        #                     shade=False, edgecolor='none')
        #
        # # take first three plots
        #
        #
        # # ax.plot([0, 1], [0, 1], [0, 1])
        # #
        # # ax.imshow(img, aspect="auto")

    print("Saving video...")
    video_writer.release()



    # image_type_tables = dict()
    # master_image_types = dict()
    #
    # print(F"Considering {input_folder}")
    #
    # image_type_tables[input_folder] = {image_type: t.loc[t.type == image_type] for image_type in t['type'].unique()}
    #
    # # sort them by time and use time as index
    # image_type_tables[input_folder] = {k: v.sort_values('t') for k, v in image_type_tables[input_folder].items()}
    # image_type_tables[input_folder] = {k: v.drop_duplicates('t') for k, v in
    #                                    image_type_tables[input_folder].items()}
    # image_type_tables[input_folder] = {k: v.set_index('t') for k, v in image_type_tables[input_folder].items()}
    # lengths = [len(table) for table in image_type_tables[input_folder].values()]
    # master_image_types[input_folder] = list(image_type_tables[input_folder].keys())[np.argmax(lengths)]
    #
    # print(
    #     F'Found following image types {image_type_tables[input_folder].keys()} with this amount of entries: {lengths}')
    #
    # master_folder = input_folder
    #
    # master_time = image_type_tables[master_folder][master_image_types[master_folder]].index
    #
    # master_time = np.arange(np.min(master_time)+0.1, np.max(master_time)-0.1, 0.001)
    #
    # master_time = master_time[35000:40000]
    #
    # for input_folder in input_folders:
    #
    #     file_lists = {k: [] for k in image_type_tables[input_folder].keys()}
    #
    #     # for loop for now:
    #     for t in tqdm.tqdm(master_time):
    #         for image_type, table in image_type_tables[input_folder].items():
    #             row = table.index.get_loc(t, 'ffill')
    #             file_lists[image_type].append(table.iloc[row]['filename'])
    #
    #     # see https://stackoverflow.com/a/52115793
    #
    #     name = os.path.basename(os.path.normpath(os.path.join(input_folder, "..")))
    #
    #     frames_folder = os.path.join(input_folder, "frames")
    #     renderer[name] = {
    #         image_type: RgbFrameRenderer(F"{name}".upper(), file_lists[image_type], frames_folder) for image_type in
    #         image_type_tables[input_folder].keys()
    #     }
    # #
    # # file_list = t.loc[t.type == 'input_img']['filename']
    # #
    # # test_renderer = RgbFrameRenderer("TEST", file_list, frames_folder)
    #
    # render_list = [renderer['eklt']['feature_img'], renderer['eklt']['tracker_img'],
    #                renderer['haste']['feature_img'], BlankRenderer(len(renderer['haste']['feature_img'].file_lists[0]), (240, 180)),
    #                renderer['xvio']['feature_img'], renderer['xvio']['tracker_img']]
    #
    # dataset_player = DatasetPlayer(render_list, 100, scale=2, grid_size=(2, 3), row_first=False)
    #                                # output_video_file="/tmp/out.avi")
    # dataset_player.run()


def create_event_tracks_correspondences(df_backend_tracks, df_event_tracks_interpolation):
    backend_to_event_track_id_map = {}
    for i in tqdm.tqdm(df_backend_tracks['id'].unique()):
        j = find_event_track_from_backend_track(df_event_tracks_interpolation, df_backend_tracks, i)
        backend_to_event_track_id_map[i] = j
    return backend_to_event_track_id_map


if __name__ == '__main__':
    main()

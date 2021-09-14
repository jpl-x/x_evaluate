import argparse
import os

import cv2
import numpy as np
import pandas as pd
import tqdm
from x_evaluate.plots import PlotContext

from x_evaluate.utils import read_x_evaluate_gt_csv, convert_to_evo_trajectory

from x_evaluate.visualizer.dataset_player import DatasetPlayer
from x_evaluate.visualizer.renderer import RgbFrameRenderer, BlankRenderer


def main():
    parser = argparse.ArgumentParser(description="Visualizes trajectories as animation")
    parser.add_argument('--input_folders', nargs='+', required=True, type=str)

    args = parser.parse_args()

    input_folders = []
    ref_trajectory = None
    trajectories = []

    for f in args.input_folders:

        if ref_trajectory is None:
            ref_trajectory = read_x_evaluate_gt_csv(os.path.join(f, "gt.csv"))

        df_poses = pd.read_csv(os.path.join(f, "pose.csv"), delimiter=";")
        traj_est, _ = convert_to_evo_trajectory(df_poses, prefix="estimated_")

        traj_est.timestamps -= ref_trajectory.timestamps[0]
        trajectories.append(traj_est)

        input_folders.append(f)

    print(trajectories)

    video_writer = cv2.VideoWriter(os.path.join(input_folder, "feature_tracks_2d.avi"), cv2.VideoWriter_fourcc(
        *'DIVX'), 25, (1280, 720))

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


    # renderer = dict()
    #
    # image_type_tables = dict()
    # master_image_types = dict()
    #
    # for i in range(len(input_folders)):
    #     input_folder = input_folders[i]
    #     t = frame_tables[i]
    #
    #     print(F"Considering {input_folder}")
    #
    #     image_type_tables[input_folder] = {image_type: t.loc[t.type == image_type] for image_type in t['type'].unique()}
    #
    #     # sort them by time and use time as index
    #     image_type_tables[input_folder] = {k: v.sort_values('t') for k, v in image_type_tables[input_folder].items()}
    #     image_type_tables[input_folder] = {k: v.drop_duplicates('t') for k, v in
    #                                        image_type_tables[input_folder].items()}
    #     image_type_tables[input_folder] = {k: v.set_index('t') for k, v in image_type_tables[input_folder].items()}
    #     lengths = [len(table) for table in image_type_tables[input_folder].values()]
    #     master_image_types[input_folder] = list(image_type_tables[input_folder].keys())[np.argmax(lengths)]
    #
    #     print(
    #         F'Found following image types {image_type_tables[input_folder].keys()} with this amount of entries: {lengths}')
    #
    # master_folder = input_folders[0]
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


if __name__ == '__main__':
    main()

import argparse
import os

import numpy as np
import pandas as pd
import tqdm

from x_evaluate.visualizer.dataset_player import DatasetPlayer
from x_evaluate.visualizer.renderer import RgbFrameRenderer, BlankRenderer


def main():
    parser = argparse.ArgumentParser(description="Visualizes the output frames provided by ")
    parser.add_argument('--input_folders', nargs='+', required=True, type=str)

    args = parser.parse_args()

    frame_tables = []
    input_folders = []

    for f in args.input_folders:

        frames_csv = os.path.join(f, "dumped_frames.csv")
        if not os.path.exists(frames_csv):
            print(F"Warning no 'dumped_frames.csv' found in {f}")
            continue

        df_frames_csv = pd.read_csv(frames_csv, delimiter=";")

        if len(df_frames_csv) <= 0:
            print(F"Empty 'dumped_frames.csv' found in {f}")
            continue

        frame_tables.append(df_frames_csv)
        input_folders.append(f)

    renderer = dict()

    image_type_tables = dict()
    master_image_types = dict()

    for i in range(len(input_folders)):
        input_folder = input_folders[i]
        t = frame_tables[i]

        print(F"Considering {input_folder}")

        image_type_tables[input_folder] = {image_type: t.loc[t.type == image_type] for image_type in t['type'].unique()}

        # sort them by time and use time as index
        image_type_tables[input_folder] = {k: v.sort_values('t') for k, v in image_type_tables[input_folder].items()}
        image_type_tables[input_folder] = {k: v.drop_duplicates('t') for k, v in
                                           image_type_tables[input_folder].items()}
        image_type_tables[input_folder] = {k: v.set_index('t') for k, v in image_type_tables[input_folder].items()}
        lengths = [len(table) for table in image_type_tables[input_folder].values()]
        master_image_types[input_folder] = list(image_type_tables[input_folder].keys())[np.argmax(lengths)]

        print(
            F'Found following image types {image_type_tables[input_folder].keys()} with this amount of entries: {lengths}')

    master_folder = input_folders[0]

    master_time = image_type_tables[master_folder][master_image_types[master_folder]].index

    # master_time = np.arange(np.min(master_time)+0.5, np.max(master_time)-0.1, 0.001)

    # master_time = master_time[:10000]

    master_time = np.arange(20, 30, 0.001)

    for input_folder in input_folders:

        file_lists = {k: [] for k in image_type_tables[input_folder].keys()}

        # for loop for now:
        for t in tqdm.tqdm(master_time):
            for image_type, table in image_type_tables[input_folder].items():
                row = table.index.get_loc(t, 'ffill')
                file_lists[image_type].append(table.iloc[row]['filename'])

        # see https://stackoverflow.com/a/52115793

        name = os.path.basename(os.path.normpath(os.path.join(input_folder, "..")))

        frames_folder = os.path.join(input_folder, "frames")
        renderer[name] = {
            image_type: RgbFrameRenderer(F"{name}".upper(), file_lists[image_type], frames_folder) for image_type in
            image_type_tables[input_folder].keys()
        }
    #
    # file_list = t.loc[t.type == 'input_img']['filename']
    #
    # test_renderer = RgbFrameRenderer("TEST", file_list, frames_folder)

    render_list = [renderer['xvio']['feature_img'], renderer['eklt']['feature_img'], renderer['haste']['feature_img']]
                   # renderer['haste']['feature_img'], BlankRenderer(len(renderer['haste']['feature_img'].file_lists[0]), (240, 180)),
                   # renderer['xvio']['feature_img'], renderer['xvio']['tracker_img']]

    dataset_player = DatasetPlayer(render_list, 100, scale=2,  #)
                                   output_video_file="/tmp/out.avi")
    dataset_player.run()


if __name__ == '__main__':
    main()

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from x_evaluate.utils import timestamp_to_rosbag_time_zero


def main():
    parser = argparse.ArgumentParser(description='Angle correlation analyzer')
    parser.add_argument('--output_folder', type=str, required=True)
    parser.add_argument('--input_folder', type=str, required=True)
    args = parser.parse_args()
    tracks_file = os.path.join(args.input_folder, 'tracks.csv')
    time_file = os.path.join(args.input_folder, 'realtime.csv')
    print(F"opening {tracks_file}")
    tracks = pd.read_csv(tracks_file, delimiter=";")
    rt = pd.read_csv(time_file, delimiter=";")

    analyzed_tracks = 0
    cur_id = 0

    while analyzed_tracks < 100:
        cur_id += 1
        track = tracks.loc[tracks.id == cur_id]

        if len(track) > 10:
            track_times = track['patch_t_current'].to_numpy() - track.iloc[0]['patch_t_current']
            delta_t = track_times[-1] - track_times[0]

            if delta_t < 0.1:
                print(F"Delta t too small: {delta_t}")
                continue

            print(F"Analyzing track with {len(track)} updates, tracked over {delta_t:.1f}s")
            print(F"    had the following updates: {track['update_type'].unique()}")

            last_row_is_lost = track.loc[track.update_type == 'Lost'].index[0] == track.iloc[-1:].index[0]
            assert last_row_is_lost, "Assuming only last update to be 'lost'"

            features_pos = track[['center_x', 'center_y']].to_numpy()
            diff = features_pos[1:, :] - features_pos[:-1, :]

            angle_estimates = np.arctan2(diff[:, 1], diff[:, 0])
            angle_estimates = np.mod(angle_estimates, 2 * np.pi)
            angle = track['flow_angle'].to_numpy()[1:]

            print(F"    flow_angle in [{np.min(angle)}, {np.max(angle)}]"
                  F" estimate in [{np.min(angle_estimates)},{np.max(angle_estimates)}]")

            plt.figure()
            plt.plot(track_times[1:-1], np.rad2deg(angle_estimates[:-1]), label="Differential feature position angle")
            plt.plot(track_times[1:-1], np.rad2deg(angle[:-1]), label="Optimized angle")
            plt.ylabel("angle [deg]")
            plt.legend()

            filename = F"track_{cur_id}.svg"
            file = os.path.join(args.output_folder, filename)
            plt.savefig(file)
            plt.clf()
            plt.close()

            analyzed_tracks += 1


if __name__ == '__main__':
    main()

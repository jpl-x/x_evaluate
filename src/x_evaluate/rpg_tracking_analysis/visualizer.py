from os.path import isfile
import os
import cv2
from os.path import join
import numpy as np
import tqdm

from x_evaluate.rpg_tracking_analysis.tracker_utils import filter_tracks, get_track_data


class FeatureTracksVisualizer:
    def __init__(self, file, dataset, params):
        self.params = params

        self.dataset = dataset
        self.tracks, self.feature_ids, self.colors = self.load_feature_tracks(file)

        self.min_time_between_screen_refresh_ms = 5
        self.max_time_between_screen_refresh_ms = 100

        self.is_paused = False
        self.is_looped = False

        self.marker = params["marker"]

        self.compute_min_max_speed()
        self.update_display_rate()

        self.times = np.linspace(self.min_stamp, self.max_stamp,
                                 int(self.params['framerate'] * (self.max_stamp - self.min_stamp)))

        self.time_index = 0

        self.cv2_window_name = 'tracks'

        cv2.namedWindow(self.cv2_window_name, cv2.WINDOW_NORMAL)

    @staticmethod
    def crop_gt(gt, predictions):
        gt = {i: g for i, g in gt.items() if i in predictions}
        predictions = {i: p for i, p in predictions.items() if i in gt}

        for i, gt_track in gt.items():
            prediction_track = predictions[i]

            t_max = prediction_track[-1, 0]
            gt_track = gt_track[gt_track[:, 0] <= t_max]

            gt[i] = gt_track

        return gt

    @staticmethod
    def discard_on_threshold(predictions, gt, thresh):
        assert set(gt.keys()) == set(predictions.keys())

        for i, gt_track in gt.items():
            pred_track = predictions[i]
            x_p_interp = np.interp(gt_track[:, 0], pred_track[:, 0], pred_track[:, 1])
            y_p_interp = np.interp(gt_track[:, 0], pred_track[:, 0], pred_track[:, 2])
            error = np.sqrt((x_p_interp - gt_track[:, 1]) ** 2 + (y_p_interp - gt_track[:, 2]) ** 2)
            idxs = np.where(error > thresh)[0]
            if len(idxs) == 0:
                continue
            t_max = gt_track[idxs[0], 0]

            gt[i] = gt_track[:idxs[0]]
            predictions[i] = pred_track[pred_track[:, 0] < t_max]

        return predictions, gt

    def load_feature_tracks(self, file, method="estimation", color=[0, 255, 0], gt_color=[255, 0, 255]):
        tracks = {}
        colors = {}

        tracks_dir = os.path.dirname(file)
        color = [r for r in reversed(color)]
        colors[method] = color

        # load tracks
        data = np.genfromtxt(file, delimiter=" ")
        first_len_tracks = len(data)

        valid_ids, data = filter_tracks(data, filter_too_short=True)
        track_data = {i: data[data[:, 0] == i, 1:] for i in valid_ids}
        tracks[method] = track_data

        if len(track_data) < first_len_tracks:
            print("WARNING: This package only supports evaluation of tracks which have been initialized at the same"
                  "time. All tracks except the first have been discarded.")

        # load gt
        tracks_csv = join(tracks_dir, "tracks.txt.gt.txt")
        if isfile(tracks_csv):
            gt = get_track_data(tracks_csv)
            colors["gt"] = gt_color

            # if true, crop all tracks from gt to have the same length as the predictions.
            if self.params["crop_to_predictions"]:
                gt = self.crop_gt(gt, tracks[method])

            if self.params["error_threshold"] > 0:
                tracks[method], gt = self.discard_on_threshold(tracks[method], gt, self.params["error_threshold"])

            tracks["gt"] = gt

        feature_ids = {label: list(tracks_dict.keys()) for label, tracks_dict in tracks.items()}

        max_stamp = -1
        min_stamp = 10 ** 1000

        for label, tracks_dict in tracks.items():
            for i, track in tracks_dict.items():
                min_stamp = min([min_stamp, min(track[:, 0])])
                max_stamp = max([max_stamp, max(track[:, 0])])

        self.min_stamp = min_stamp
        self.max_stamp = max_stamp

        return tracks, feature_ids, colors

    def pause(self):
        self.is_paused = True

    def unpause(self):
        self.is_paused = False

    def toggle_pause(self):
        self.is_paused = not self.is_paused

    def toggle_loop(self):
        self.is_looped = not self.is_looped

    def forward(self, num_timesteps=1):
        if self.is_looped:
            self.time_index = (self.time_index + 1) % len(self.times)
        else:
            self.time_index = min(self.time_index + num_timesteps, len(self.times) - 1)

    def backward(self, num_timesteps=1):
        self.time_index = max(self.time_index - num_timesteps, 0)

    def go_to_begin(self):
        self.time_index = 0

    def go_to_end(self):
        self.time_index = len(self.times) - 1

    def increase_track_history_length(self):
        self.params['track_history_length'] = self.params['track_history_length'] * 1.25

    def decrease_track_history_length(self):
        self.params['track_history_length'] = self.params['track_history_length'] / 1.25

    def compute_min_max_speed(self):
        self.max_speed = 1000.0 / (self.min_time_between_screen_refresh_ms * self.params['framerate'])
        self.min_speed = 1000.0 / (self.max_time_between_screen_refresh_ms * self.params['framerate'])

    def update_display_rate(self):

        self.params['speed'] = np.clip(self.params['speed'], self.min_speed, self.max_speed)
        self.time_between_screen_refresh_ms = int(1000.0 / (self.params['speed'] * self.params['framerate']))

    def write_to_video_file(self, f):
        height, width, _ = self.dataset.images[0].shape
        height, width = int(self.params["scale"] * height), int(self.params["scale"] * width)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.video_writer = cv2.VideoWriter(f, fourcc, self.params['framerate'] * self.params["speed"], (width, height))

        for t in tqdm.tqdm(self.times):
            image_to_display = self.update(t)
            self.video_writer.write(image_to_display)

        self.cleanup()

    def visualization_loop(self):

        while True:
            t = self.times[self.time_index]
            image_to_display = self.update(t)
            cv2.imshow(self.cv2_window_name, image_to_display)

            if not self.is_paused:
                self.forward(1)

            c = cv2.waitKey(self.time_between_screen_refresh_ms)
            key = chr(c & 255)

            if c == 27:  # 'q' or 'Esc': Quit
                break
            elif key == 'r':  # 'r': Reset
                self.go_to_begin()
                self.unpause()
            elif key == 'p' or c == 32:  # 'p' or 'Space': Toggle play/pause
                self.toggle_pause()
            elif key == "a":  # 'Left arrow': Go backward
                self.backward(1)
                self.pause()
            elif key == "d":  # 'Right arrow': Go forward
                self.forward(1)
                self.pause()
            elif key == "s":  # 'Down arrow': Go to beginning
                self.go_to_begin()
                self.pause()
            elif key == "w":  # 'Up arrow': Go to end
                self.go_to_end()
                self.pause()
            elif key == "e":
                self.increase_track_history_length()
            elif key == "q":
                self.decrease_track_history_length()
            elif key == 'l':  # 'l': Toggle looping
                self.toggle_loop()

        self.cleanup()

    def cleanup(self):
        cv2.destroyAllWindows()

        if hasattr(self, 'video_writer'):
            self.video_writer.release()

    def update(self, t, track_history_length=None):
        if track_history_length == None:
            track_history_length = self.params['track_history_length']

        return self.plot_between(t - track_history_length, t)

    def get_image_closest_to(self, t):
        image_index = np.searchsorted(self.dataset.times, t, side="left") - 1
        return self.dataset.images[image_index]

    def draw_marker(self, img, x, y, color):
        c = int(3 * self.params["scale"])
        t = int(1 * self.params["scale"])
        if self.marker == "cross":
            cv2.line(img, (x - c, y), (x + c, y), color, thickness=t)
            cv2.line(img, (x, y - c), (x, y + c), color, thickness=t)
        elif self.marker == "circle":
            cv2.circle(img, center=(x, y), radius=c, color=color, thickness=t)

    def draw_legend(self, image, legend, size):
        s = self.params["scale"]

        off_x = int(size[1])
        t = int(10 * s)
        n = int(70 * s)

        for label, color in legend.items():
            if label == "gt":
                label = "ground truth"
            cv2.putText(image, label, (off_x - n, t), cv2.FONT_HERSHEY_COMPLEX, int(s / 4), color)
            t += int(10 * s)

        return image

    def plot_between(self, t0, t1):
        image = self.get_image_closest_to(t1).copy()

        # resize
        h, w, _ = image.shape
        s = self.params["scale"]
        image = cv2.resize(image, dsize=(int(w * s), int(h * s)))

        image = self.draw_legend(image, self.colors, image.shape[:2])

        for label, tracks_dict in self.tracks.items():
            for feature_id, track in tracks_dict.items():
                t = track[:, 0]
                track_segment = track[(t <= t1) & (t >= t0)]

                if len(track_segment) > 0:
                    for point in track_segment[:-1]:
                        _, x, y = (s * point).astype(int)
                        trail_marker = "cross" if label == "gt" else "dot"
                        self.draw_trail(image, x, y, self.colors[label], marker=trail_marker)

                    _, x, y = (s * track_segment[-1]).astype(int)
                    self.draw_marker(image, x, y, self.colors[label])

        return image

    def draw_trail(self, img, x, y, entry, marker="dot"):
        c = 0 * self.params["scale"]
        if marker == "dot":
            x_min = int(max([x - c, 0]))
            x_max = int(min([x + c + self.params["scale"], img.shape[1]]))
            y_min = int(max([y - c, 0]))
            y_max = int(min([y + c + self.params["scale"], img.shape[0]]))

            img[y_min:y_max, x_min:x_max, :] = np.array(entry)

        elif marker == "cross":
            c = int(2 * self.params['scale'])
            t = int(.5 * self.params["scale"])

            x_min = max([x - c, 0])
            x_max = int(min([x + c + self.params["scale"], img.shape[1]]))
            y_min = max([y - c, 0])
            y_max = int(min([y + c + self.params["scale"], img.shape[0]]))

            xmi = int(x - t / 2)
            ymi = int(y - t / 2)
            xma = int(x + t / 2)
            yma = int(y + t / 2)

            if x < 0 or x > img.shape[1] - 1 or y < 0 or x > img.shape[0] - 1:
                return

            img[y_min:y_max, xmi:xma, :] = np.array(entry)
            img[ymi:yma, x_min:x_max, :] = np.array(entry)

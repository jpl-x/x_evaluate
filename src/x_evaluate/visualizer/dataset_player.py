import glob

import cv2
import pygame
from enum import Enum
from typing import List
import numpy as np

from x_evaluate.visualizer.utils import get_pygame_font
from x_evaluate.visualizer.renderer import AbstractFrameRenderer


class Action(Enum):
    NO_ACTION = 0
    QUIT = 1
    TOGGLE_PAUSE = 2
    LEFT = 3
    RIGHT = 4
    TOGGLE_REVERSE = 5


class DatasetPlayer:

    def __init__(self, frame_renderer: List[AbstractFrameRenderer], fps=25, scale=1, grid_size=None,
                 output_video_file=None, row_first=True):
        self._frame_renderer = frame_renderer
        self._fps = fps
        self._file_lists = []
        self._scale = scale
        self._action = Action.NO_ACTION
        self._frame_height = None
        self._frame_width = None
        self._row_first = row_first
        self._number_of_tiles = sum([x.number_of_outputs for x in self._frame_renderer])
        if grid_size is not None:
            self._grid_height, self._grid_width = grid_size
            assert(self._grid_width * self._grid_height >= self._number_of_tiles)
        else:
            self.calculate_grid_size()

        # getting font after pygame is initialized
        self._font = None
        self.number_of_frames = self.prepare_filelists_and_frame_size()
        self._empty_first_frame = self.prepare_empty_first_frame(self._frame_width, self._frame_height)

        if output_video_file is not None:
            self._video_writer = cv2.VideoWriter(output_video_file, cv2.VideoWriter_fourcc(*'DIVX'), 25,
                                                 (self._grid_width*self._frame_width,
                                                  self._grid_height*self._frame_height))
        else:
            self._video_writer = None

    def update_action(self):
        self._action = Action.NO_ACTION

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._action = Action.QUIT
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_ESCAPE:
                    self._action = Action.QUIT
                elif event.key == pygame.K_SPACE:
                    self._action = Action.TOGGLE_PAUSE
                elif event.key == pygame.K_LEFT:
                    self._action = Action.LEFT
                elif event.key == pygame.K_RIGHT:
                    self._action = Action.RIGHT
                elif event.key == pygame.K_r:
                    self._action = Action.TOGGLE_REVERSE

    def calculate_frame_size(self):
        # do a test render
        surf = self._frame_renderer[0].render([self._file_lists[0][i][0] for i in range(len(self._file_lists[0]))])
        self._frame_height = int(surf[0].get_height()*self._scale)
        self._frame_width = int(surf[0].get_width()*self._scale)

    def calculate_grid_size(self) -> (int, int):
        w = 1
        h = 1
        while self._number_of_tiles > w * h:
            if w - h < 2:  # this number should adapt, but works fine up to ~30
                w = w + 1
            else:
                w = w - 1
                h = h + 1
        self._grid_height, self._grid_width = h, w

    def run(self):
        pygame.init()
        display = pygame.display.set_mode(
            (self._grid_width*self._frame_width, self._grid_height*self._frame_height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        clock = pygame.time.Clock()
        self._font = get_pygame_font()
        self._empty_first_frame = self.prepare_empty_first_frame(self._frame_width, self._frame_height)

        increment = 1
        current_frame = -1
        is_replaying = True

        while True:
            self.update_action()

            if self._action == Action.QUIT:
                break
            elif self._action == Action.TOGGLE_PAUSE:
                is_replaying = not is_replaying
                # print(F"Is replaying changed: {is_replaying}")
            elif self._action == Action.TOGGLE_REVERSE:
                increment = -increment

            is_replaying = is_replaying and current_frame+increment in range(self.number_of_frames)

            frame_has_changed = True

            if is_replaying:
                current_frame = current_frame + increment
            elif self._action == Action.RIGHT and current_frame + 1 < self.number_of_frames:
                current_frame = current_frame + 1
            elif self._action == Action.LEFT and current_frame > 0:
                current_frame = current_frame - 1
            else:
                frame_has_changed = False

            clock.tick(self._fps)

            if frame_has_changed:
                self.render_all_current_frames(current_frame, display)

                # if self._font is not None:
                #     display.blit(
                #         self._font.render('%2d FPS (target: %2d FPS)' % (clock.get_fps(), self._fps),
                #                           True, (250, 250, 250)),
                #         (self._frame_width * 0.05, self._frame_height * 0.05))
                # Draw the display.
                # draw_image(display, image)
                pygame.display.flip()

        if self._video_writer is not None:
            print("Saving video...")
            self._video_writer.release()

        pygame.quit()

    def prepare_filelists_and_frame_size(self):
        lengths = []
        for idx, r in enumerate(self._frame_renderer):
            self._file_lists.append(r.file_lists)
            lengths += [len(f) for f in r.file_lists]
            # for m in r.file_matcher:
            #     file_list = glob.glob(m)
            #     file_list.sort()
            #     files_found = len(file_list)
            #     lengths.append(files_found)
            #     self._file_lists[idx].append(file_list)
            #
            #     if files_found == 0:
            #         print(F"Warning: no files using the matcher {r.file_matcher}")
        number_of_frames = min(lengths)
        assert number_of_frames > 0, "Expected RGB frames to exist, but no matching file found"
        self.calculate_frame_size()
        return number_of_frames

    def render_all_current_frames(self, current_frame, display):
        grid_i, grid_j = 0, 0
        for idx, r in enumerate(self._frame_renderer):
            file_list = [self._file_lists[idx][i][current_frame]
                         for i in range(len(self._file_lists[idx]))]

            frames = []

            if r.requires_prev_files:
                if current_frame > 0:
                    # append previous files
                    for i in range(len(self._file_lists[idx])):
                        file_list.append(self._file_lists[idx][i][current_frame - 1])

                    frames = r.render(file_list)
                else:
                    frames = [self._empty_first_frame for _ in range(r.number_of_outputs)]
            else:
                frames = r.render(file_list)

            assert len(frames) == r.number_of_outputs, F"Expecting {r.number_of_outputs} outputs from renderer {r}, " \
                                                       F"but got {len(frames)}"

            for i, frame in enumerate(frames):

                if frame.get_width() != self._frame_width or frame.get_width() != self._frame_height:
                    # when up-scaling, scaling into the same surface is not sufficient. To generalize allocate new one.
                    # passing 'frame' makes sure to use same format
                    scaled = pygame.Surface((self._frame_width, self._frame_height), 0, frame)
                    pygame.transform.smoothscale(frame, (self._frame_width, self._frame_height), scaled)
                    frame = scaled

                full_filename = self._file_lists[idx][0][current_frame]
                base_filename = full_filename.split('/')[-1]

                if self._font is not None:
                    # text = self._font.render('%s [%s]' % (base_filename, r.names[i]), True, (250, 250, 250))
                    text = self._font.render('%s' % (r.names[i]), True, (250, 250, 250))
                    frame.blit(text, (self._frame_width / 2 - text.get_rect().width / 2, self._frame_height * 0.1))
                display.blit(frame, (self._frame_width * grid_j, self._frame_height * grid_i))

                if self._row_first:
                    if grid_j >= self._grid_width - 1:
                        grid_i, grid_j = grid_i + 1, 0
                    else:
                        grid_j = grid_j + 1
                else:
                    if grid_i >= self._grid_height - 1:
                        grid_j, grid_i = grid_j + 1, 0
                    else:
                        grid_i = grid_i + 1

        if self._video_writer is not None:
            img = pygame.surfarray.array3d(display)
            # img = img.swapaxes(0, 1)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = np.fliplr(img)
            img = np.rot90(img)
            self._video_writer.write(img)

    def prepare_empty_first_frame(self, width, height):
        frame = pygame.Surface((width, height))
        frame.fill((0, 0, 255))
        if self._font is not None:
            text = self._font.render("Not available on first frame", True, (0, 255, 0))
            frame.blit(text, (width / 2 - text.get_rect().width / 2, height * 0.5))
        return frame


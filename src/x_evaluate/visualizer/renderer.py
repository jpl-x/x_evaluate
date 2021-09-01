import os
from abc import ABCMeta, abstractmethod
from typing import List

import cv2
import numpy as np
import pygame


class AbstractFrameRenderer(metaclass=ABCMeta):

    names: List[str]
    file_lists: List[List[str]]
    requires_prev_files: bool
    number_of_outputs: int

    def __init__(self, name: List[str], file_lists: List[List[str]], requires_prev_files: bool = False,
                 number_of_output_surfaces: int = 1):
        self.names = name
        self.file_lists = file_lists
        self.requires_prev_files = requires_prev_files
        self.number_of_outputs = number_of_output_surfaces

    @abstractmethod
    def render(self, files: List[str]) -> List[pygame.SurfaceType]:
        pass


class BlankRenderer(AbstractFrameRenderer):

    def __init__(self,  file_list_length: int, frame_size):

        file_list = [""] * file_list_length
        self.frame_size = frame_size

        super().__init__([""], [file_list])

    def render(self, files: List[str]) -> List[pygame.SurfaceType]:
        return [pygame.Surface(self.frame_size)]


class RgbFrameRenderer(AbstractFrameRenderer):

    def __init__(self, name: str, file_list: List[str], root_folder=None):
        # path = os.path.join(dataset_path, F"{sensor}/{sub_folder}")

        if root_folder:
            file_list = [os.path.join(root_folder, f) for f in file_list]

        super().__init__([name], [file_list])

    def render(self, files: List[str]) -> List[pygame.SurfaceType]:
        # THIS SHOULD BE UNIT TESTED IN ABSTRACT FRAME RENDERER: assert len(files) == 1
        img = cv2.imread(files[0])
        im_surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))
        return [im_surface]

#
# class DepthFrameRenderer(RgbFrameRenderer):
#
#     def __init__(self, dataset_path, sensor="depth", sub_folder="frames"):
#         super().__init__(dataset_path, sensor, sub_folder=sub_folder)
#
#     def render(self, files: List[str]) -> List[pygame.SurfaceType]:
#         img = cv2.imread(files[0], cv2.COLOR_RGB2GRAY)
#         # im_surface = pygame.image.load(files[0])
#         # img = img[:, :, :3]
#         # img = img[:, :, ::-1]
#         im_surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))
#         return [im_surface]
#
#
# class RgbFrameProjectionRenderer(RgbFrameRenderer):
#
#     def update_points_to_project(self, points_3d):
#         assert np.shape(points_3d)[0] == 3, "expecting 3xn 3d points here"
#         self._points_3d = points_3d
#
#     def __init__(self, dataset_path, projection_matrix, sensor="rgb", sub_folder="frames"):
#         super().__init__(dataset_path, sensor, sub_folder=sub_folder)
#         self._points_3d = np.array([[], [], []])
#         self.projection_matrix = projection_matrix
#
#     def render(self, files: List[str]) -> List[pygame.SurfaceType]:
#         img = cv2.imread(files[0])
#
#         # im_surface = pygame.image.load(files[0])
#
#         xy = np.matmul(self.projection_matrix, self._points_3d)
#         uv = xy[0:2, :] / xy[2, :]  # the flip is taken care of later
#
#         img = img[:, :, :3]
#         img = img[:, :, ::-1]
#         img = np.ascontiguousarray(img, dtype=np.uint8)  # avoids issue with cv2.drawMarker
#
#         for i in range(np.shape(uv)[1]):
#             cv2.drawMarker(img, (int(round(uv[0, i])), int(round(uv[1, i]))), (255, 0, 0), markerType=cv2.MARKER_STAR,
#                            markerSize=10, thickness=2, line_type=cv2.LINE_AA)
#
#         im_surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))
#         return [im_surface]

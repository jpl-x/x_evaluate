import cv2
import numpy as np
import pygame


def draw_arrow_with_cv2(img_out, p, vel):
    vel_norm = np.linalg.norm(vel)
    if vel_norm < 1:
        return
    spin_size = 0.3*np.linalg.norm(vel_norm)
    p2 = [sum(x) for x in zip(p, vel)]
    delta = [x2 - x for x2, x in zip(p2, p)]
    p2 = tuple(np.round(p2).astype(int))
    # cv2.line()
    cv2.line(img_out, p, p2, (220, 220, 220), thickness=1, lineType=cv2.LINE_AA)
    # cvLine(resultDenseOpticalFlow, p, p2, CV_RGB(220, 220, 220), 1, CV_AA);
    angle = np.arctan2(delta[1], delta[0])
    p = (p2[0] - spin_size * np.cos(angle + np.pi / 4),
         p2[1] - spin_size * np.sin(angle + np.pi / 4))
    p = tuple(np.round(p).astype(int))
    cv2.line(img_out, p, p2, (220, 220, 220), thickness=1, lineType=cv2.LINE_AA)
    p = (p2[0] - spin_size * np.cos(angle - np.pi / 4),
         p2[1] - spin_size * np.sin(angle - np.pi / 4))
    p = tuple(np.round(p).astype(int))
    cv2.line(img_out, p, p2, (220, 220, 220), thickness=1, lineType=cv2.LINE_AA)


def get_pygame_font(desired_font='ubuntumono'):
    fonts = [x for x in pygame.font.get_fonts()]
    font = desired_font if desired_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 24)


def render_optical_flow_data(data):
    intensity = np.linalg.norm(data, axis=2)
    angle = np.arctan2(data[:, :, 0], data[:, :, 1])
    max_intensity = 100
    # N.B.: an intensity of exactly 1.0 makes the output black (perhaps showing the over-saturation), so keep it < 1
    intensity = np.clip(intensity, 0, max_intensity - 1) / max_intensity
    # log scaling
    basis = 30
    intensity = np.log1p((basis - 1) * intensity) / np.log1p(basis - 1)
    # for the angle they use 360° scale, see https://stackoverflow.com/a/57203497/14467327
    angle = (np.pi + angle) * 360 / (2 * np.pi)
    # print(F"Ranges, angle: [{np.min(angle)}, {np.max(angle)}], "
    #       F"intensity: [{np.min(intensity)}, {np.max(intensity)}]")
    intensity = intensity[:, :, np.newaxis]
    angle = angle[:, :, np.newaxis]
    hsv_img = np.concatenate((angle, np.ones_like(intensity), intensity), axis=2)
    img_out = np.array(cv2.cvtColor(np.array(hsv_img, dtype=np.float32), cv2.COLOR_HSV2RGB) * 256,
                       dtype=np.dtype("uint8"))
    for y in range(12, len(data), 25):
        for x in range(12, len(data[0]), 25):
            vel = (data[y][x][0], data[y][x][1])

            p = (x, y)
            draw_arrow_with_cv2(img_out, p, vel)
    return img_out



import time

import cv2
import numpy as np
import pandas as pd
import tqdm
from scipy import interpolate
from unrealcv import client
import matplotlib.pyplot as plt


def main():
    # print('Connecting to UnrealCV client')
    # client.connect()
    #
    # unreal_trajectory = pd.DataFrame(columns=["t", "unreal_p_x", "unreal_p_y", "unreal_p_z", "unreal_rot_x",
    #                                           "unreal_rot_y", "unreal_rot_z"])
    #
    # t_0 = time.time()
    #
    # if not client.isconnected():  # Check if the connection is successfully established
    #     print('UnrealCV server is not running. Run the game from http://unrealcv.github.io first.')
    # else:
    #     time.sleep(0.03)
    #     response = client.request('vset /camera/1/fov 70')
    #     print(response)
    #     time.sleep(0.03)
    #     response = client.request('vrun r.setres 720x560')
    #     print(response)
    #     time.sleep(0.03)
    #     file = client.request('vget /camera/1/lit png')
    #     image = np.frombuffer(file, np.uint8)
    #     image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    #
    #     print(F"Got image {image.shape}")
    #
    #     t = time.time() - t_0
    #
    #     while t < 60:
    #         t = time.time() - t_0
    #         location = client.request("vget /camera/0/location")
    #         rotation = client.request("vget /camera/0/rotation")
    #         new_row = eval(F"[{location.replace(' ', ', ')}, " F"{rotation.replace(' ', ', ')}]")
    #         new_row = [t] + new_row
    #         print(new_row)
    #         unreal_trajectory.loc[len(unreal_trajectory)] = new_row
    #         print(response)
    #         time.sleep(1)
    #
    #
    #     cv2.imshow("REPLY", image)
    #     cv2.waitKey()
    #
    #     print(F"image saved to {file}")
    #     print
    #     'Image is saved to %s' % filename
    #     for gt_type in ['normal', 'object_mask']:
    #         filename = client.request('vget /camera/0/%s' % gt_type)
    #         print
    #         '%s is saved to %s' % (gt_type, filename)
    #     filename = client.request('vget /camera/0/depth depth.exr')
    #     print
    #     'depth is saved to %s' % filename
    #     # Depth needs to be saved to HDR image to ensure numerical accuracy
    # unreal_trajectory.to_csv("unreal_panoramic_tour.csv")




    # READ, INTERPOLATE, RENDER

    unrealcv_trajectory = pd.read_csv("unreal_panoramic_tour.csv")

    # plt.figure()
    # plt.plot(unrealcv_trajectory['t'], unrealcv_trajectory['unreal_rot_y'])
    # plt.show()
    fps = 25

    # supposes it chnages only once
    changing_pair = np.flatnonzero(np.array(abs(unrealcv_trajectory['unreal_rot_y'].diff()) > 300))
    unrealcv_trajectory.loc[changing_pair[0]:(changing_pair[1]-1), 'unreal_rot_y'] += 360

    t = np.arange(0, 60.0, 1.0/fps)

    rot_x = normalize_unreal_angle(interpolate_column('unreal_rot_x', t, unrealcv_trajectory))
    rot_y = normalize_unreal_angle(interpolate_column('unreal_rot_y', t, unrealcv_trajectory))
    p_x = interpolate_column('unreal_p_x', t, unrealcv_trajectory)
    p_y = interpolate_column('unreal_p_y', t, unrealcv_trajectory)
    p_z = interpolate_column('unreal_p_z', t, unrealcv_trajectory) + 100
    plt.figure()
    plt.plot(t, rot_y)
    plt.show()
    plt.figure()
    plt.plot(t, rot_x)
    plt.show()

    video_writer = cv2.VideoWriter("mars.avi", cv2.VideoWriter_fourcc(*'DIVX'), fps, (1920, 1080))

    print('Connecting to UnrealCV client')
    client.connect()

    if not client.isconnected():  # Check if the connection is successfully established
        print('UnrealCV server is not running. Run the game from http://unrealcv.github.io first.')
    else:
        time.sleep(0.03)
        response = client.request('vset /camera/1/fov 70')
        print(response)
        time.sleep(0.03)
        response = client.request('vrun r.setres 720x560')
        print(response)

        for i in tqdm.tqdm(range(len(p_x))):
            response = client.request(F"vset /camera/1/location {p_x[i]} {p_y[i]} {p_z[i]}")
            # print(response)
            time.sleep(0.03)
            response = client.request(F"vset /camera/1/rotation {rot_x[i]} {rot_y[i]} {0.0}")
            # print(response)

            time.sleep(0.03)
            file = client.request('vget /camera/1/lit png')
            image = np.frombuffer(file, np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)

            # print(F"Got image {image.shape}")

            video_writer.write(image)

    print("Saving video...")
    video_writer.release()

    print("Done")


def interpolate_column(column, time, unrealcv_trajectory):
    tck = interpolate.splrep(unrealcv_trajectory['t'][::5], unrealcv_trajectory[column][::5], s=0)
    ynew = interpolate.splev(time, tck, der=0)
    return ynew


def normalize_unreal_angle(angle):
    angle = np.fmod(angle, 360)
    angle[angle > 180] -= 360
    return angle


if __name__ == '__main__':
    main()


import argparse
import pandas as pd
from evo.core.trajectory import PoseTrajectory3D
from scipy.spatial.transform import Rotation as R

from x_evaluate.utils import convert_to_evo_trajectory


def main():
    print("Converting NeuroBEM to ESIM trajectory")
    parser = argparse.ArgumentParser(description='Converting NeuroBEM drone trajectories to ESIM compatible csv')

    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    input_traj = pd.read_csv(args.input)

    # zero align
    input_traj['t'] -= input_traj['t'][0]

    # convert_to_evo_trajectory()
    #
    # PoseTrajectory3D(t_xyz_wxyz[:, 1:4], t_xyz_wxyz[:, 4:8], t_xyz_wxyz[:, 0]), raw_t_xyz_wxyz


    # to ns
    input_traj['t'] *= 1e9

    # normalize quaternions: ESIM checks for this and it seems NeuroBEM contains non-normalized quaternions
    rotations = R.from_quat(input_traj[["quat x", "quat y", "quat z", "quat w"]])
    rotations.as_quat()

    data = [input_traj["t"], input_traj["pos x"], input_traj["pos y"], input_traj["pos z"],
            rotations.as_quat()[:, 0], rotations.as_quat()[:, 1], rotations.as_quat()[:, 2], rotations.as_quat()[:, 3]]

    headers = ["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"]
    # output_traj = pd.concat(data, axis=1, keys=headers)

    output_traj = pd.DataFrame.from_dict(dict(zip(headers, data)))

    print(output_traj)

    with open(args.output, 'w') as f:
        # ESIM checks for this header comment
        f.write("# timestamp, x, y, z, qx, qy, qz, qw\n")
        output_traj.to_csv(f, header=False, index=False)


if __name__ == '__main__':
    main()

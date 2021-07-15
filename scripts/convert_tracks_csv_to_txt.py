import argparse
import os

from x_evaluate.utils import convert_to_tracks_txt, read_eklt_output_files


def main():
    parser = argparse.ArgumentParser(description='Converting tracks.csv to EKLT compatible txt file')

    parser.add_argument('--input_folder', type=str, required=True)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.input_folder, "tracks.txt")

    _, _, df_tracks = read_eklt_output_files(args.input_folder)

    convert_to_tracks_txt(df_tracks, args.output)


if __name__ == '__main__':
    main()

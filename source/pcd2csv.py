import open3d as o3d
import numpy as np
import pandas as pd
import argparse
import os

def convert_pcd_to_csv(input_file, output_file):
    if not output_file:
        base_name = os.path.splitext(input_file)[0]
        output_file = base_name + '.csv'

    pcd = o3d.io.read_point_cloud(input_file)
    points = np.asarray(pcd.points)
    df = pd.DataFrame(points, columns=["x", "y", "z"])
    df.to_csv(output_file, index=False, header=False)
    print(f"Converted {input_file} to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Convert PCD files to CSV format.')
    parser.add_argument('input_files', nargs='+', help='Input PCD file paths')
    parser.add_argument('-o', '--output', help='Output CSV file path')
    args = parser.parse_args()

    for input_file in args.input_files:
        convert_pcd_to_csv(input_file, args.output)

if __name__ == "__main__":
    main()

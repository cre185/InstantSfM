import os
from argparse import ArgumentParser
import subprocess

from instantsfm.controllers.data_reader import ReadData, PathInfo

def run_gaussian_splatting():
    parser = ArgumentParser()
    parser.add_argument('--data_path', required=True, help='Path to the data folder')
    gs_args = parser.parse_args()

    path_info = ReadData(gs_args.data_path)
    if not path_info:
        print('Invalid data path, please check the provided path')
        return

    image_folder_name = path_info.image_path.split(os.path.sep)[-1]

    result_path = os.path.join(gs_args.data_path, 'gsplat')
    # python instantsfm/vis/gsplat_trainer.py default --data_dir $source_path --data_factor 1 --result_dir $source_path/gsplat --close_viewer_after_training
    gs_cmd = f'python instantsfm/vis/gsplat_trainer.py default --data_dir {gs_args.data_path} --image_folder_name {image_folder_name} --data_factor 1 --result_dir {result_path} --close_viewer_after_training'
    # call the command
    subprocess.run(gs_cmd, shell=True)

def entrypoint():
    # Entry point for pyproject.toml
    run_gaussian_splatting()

if __name__ == '__main__':

    entrypoint()

import os
import time
from argparse import ArgumentParser

from instantsfm.controllers.data_reader import ReadData
from instantsfm.controllers.reconstruction_visualizer import OfflinePlayer

def run_vis():
    parser = ArgumentParser()
    parser.add_argument('--data_path', required=True, help='Path to the data folder')
    parser.add_argument('--record', default='', help='Path to the record folder if different from default folder')
    viser_args = parser.parse_args()

    path_info = ReadData(viser_args.data_path)
    if not path_info:
        print('Invalid data path, please check the provided path')
        return
    
    if viser_args.record:
        record_path = viser_args.record
    else:
        record_base_path = path_info.record_path
        if not record_base_path:
            print('No record path specified and no default record path found in data folder.')
            return
        # Find the latest record folder
        record_folders = [d for d in os.listdir(record_base_path) if os.path.isdir(os.path.join(record_base_path, d))]
        if not record_folders:
            print('No record folders found in:', record_base_path)
            return
        record_path = os.path.join(record_base_path, sorted(record_folders)[-1])
    
    if not os.path.exists(record_path):
        print('Record path does not exist:', record_path)
        return

    player = OfflinePlayer(record_path, os.path.join(path_info.output_path, '0'))
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Visualization server terminated by user.")

def entrypoint():
    # Entry point for pyproject.toml
    run_vis()
    
if __name__ == '__main__':
    entrypoint()

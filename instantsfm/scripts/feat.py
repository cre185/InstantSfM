import time
from argparse import ArgumentParser

from instantsfm.controllers.config import Config
from instantsfm.controllers.data_reader import ReadData
from instantsfm.controllers.feature_handler import GenerateDatabase

def run_feature_handler():
    handler_types = ['colmap', 'dedode', 'disk+lightglue', 'superpoint+lightglue', 'sift']
    parser = ArgumentParser()
    parser.add_argument('--data_path', required=True, help='Path to the data folder')
    parser.add_argument('--manual_config_name', help='Name of the manual configuration file')
    parser.add_argument('--feature_handler', choices=handler_types, default='colmap', help='Feature handling method to use')
    parser.add_argument('--single_camera', action='store_true', help='Assume all images are from a single camera')
    handler_args = parser.parse_args()

    path_info = ReadData(handler_args.data_path)
    if not path_info:
        print('Invalid data path, please check the provided path')
        return
    if path_info.database_exists:
        print('Database path already exists')
        return

    start_time = time.time()
    config = Config(handler_args.feature_handler, handler_args.manual_config_name)
    GenerateDatabase(path_info.image_path, path_info.database_path, handler_args.feature_handler, config, single_camera=handler_args.single_camera)
    print('Feature extraction done in', time.time() - start_time, 'seconds')

def entrypoint():
    # Entry point for pyproject.toml
    run_feature_handler()
    
if __name__ == '__main__':
    entrypoint()

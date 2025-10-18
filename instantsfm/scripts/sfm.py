import time
from argparse import ArgumentParser

from instantsfm.controllers.config import Config
from instantsfm.controllers.data_reader import ReadData, ReadColmapDatabase, ReadDepthsIntoFeatures
from instantsfm.controllers.global_mapper import SolveGlobalMapper
from instantsfm.controllers.reconstruction_writer import WriteGlomapReconstruction
from instantsfm.controllers.reconstruction_visualizer import ReconstructionVisualizer

def run_sfm():
    parser = ArgumentParser()
    parser.add_argument('--data_path', required=True, help='Path to the data folder')
    parser.add_argument('--enable_gui', action='store_true', help='Enable GUI for visualization')
    parser.add_argument('--record_recon', action='store_true', help='Save reconstruction data at each step')
    parser.add_argument('--record_path', default=None, help='Path to save the recorded reconstruction data')
    parser.add_argument('--disable_depths', action='store_true', help='Disable the use of depths if available')
    parser.add_argument('--export_txt', action='store_true', help='Export the reconstruction in binary format')
    parser.add_argument('--manual_config_name', help='Name of the manual configuration file')
    mapper_args = parser.parse_args()

    path_info = ReadData(mapper_args.data_path)
    if not path_info:
        print('Invalid data path, please check the provided path')
        return
    
    view_graph, cameras, images, feature_name = ReadColmapDatabase(path_info.database_path)
    if view_graph is None or cameras is None or images is None:
        return
    if path_info.depth_path and not mapper_args.disable_depths:
        depths = ReadDepthsIntoFeatures(path_info.depth_path, cameras, images)
    else:
        depths = None

    # enable different configs for different feature handlers and image numbers
    start_time = time.time()
    config = Config(feature_name, mapper_args.manual_config_name)
    if mapper_args.enable_gui or mapper_args.record_recon:
        visualizer = ReconstructionVisualizer(save_data=mapper_args.record_recon, 
                                              save_dir=mapper_args.record_path if mapper_args.record_path else path_info.record_path)
        cameras, images, tracks = SolveGlobalMapper(view_graph, cameras, images, config, depths=depths, visualizer=visualizer)
    else:
        cameras, images, tracks = SolveGlobalMapper(view_graph, cameras, images, config, depths=depths)
    print('Reconstruction done in', time.time() - start_time, 'seconds')
    WriteGlomapReconstruction(path_info.output_path, cameras, images, tracks, path_info.image_path, export_txt=mapper_args.export_txt)
    print('Reconstruction written to', path_info.output_path)

    if mapper_args.enable_gui:
        # block until the GUI is closed
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Visualization server terminated by user.")

def entrypoint():
    # Entry point for pyproject.toml
    run_sfm()
    
if __name__ == '__main__':

    entrypoint()

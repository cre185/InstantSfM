# Usage for colmap_eval module  
This module is used to evaluate the results of several SfM pipelines compared to given ground truth. Only AUC is tested for now.  
## 1. Install dependencies  
```bash
conda create -n eval python=3.12
conda activate eval
pip install -r instantsfm/eval/colmap_eval/requirements.txt
```
## 2. Prepare the results  
Place reconstruction result of several methods to specified place (by default `dataset/`). The results should be organized in a specific way as described below. Currently you need to manually place the results in the `dataset` directory.  
The `dataset` directory should be organized as follows:  

- Each dataset is placed in the `dataset` directory and named after the dataset's name.  
- Inside each dataset folder, there are fixed categories, and each category is named accordingly.  
- Within each category folder, there are multiple scene folders. Each scene folder follows the COLMAP storage format, which includes:  
  - An `images` folder containing the images for the scene.  
  - One or more `sparse` folders from different reconstruction methods used for computing metrics.  

Ensure that the directory structure adheres to this format for proper evaluation.  
There are different names for supported datasets, and we will show their default folder structure one by one. Note that the name of categories and scenes are not specified and you can add any number of them, while the dataset names are currently hardcoded. Extra files in scene folders are also allowed (usually files like database.db or other files included in the original dataset), so there's no need to delete them.  
- **ETH3D** (dataset name: `eth3d`):  
```
- eth3d
  - dslr
    - botanical_garden
      - dslr_calibration_undistorted
      - images
      - sparse_colmap
      - sparse_glomap
      - sparse
    - ...(similarly)
  - rig
    - ...(similarly)
```
- **Tanks and Temples** (dataset name: `tt`):  
```
- tt
  - Advanced
    - Auditorium
      - cams_1
      - images
      - sparse_colmap
      - sparse_glomap
      - sparse
    - ...(similarly)
  - Intermediate
    - ...(similarly)
```
- **DTU** (dataset name: `dtu`):  
```
- dtu
  - dtu_testing
    - scan1
      - cams
      - images
      - sparse_colmap
      - sparse_glomap
      - sparse
    - ...(similarly)
  - dtu_training
    - ...(similarly)
```
- **BlendedMVS** (dataset name: `blended_mvs`):  
```
- blended-mvs
  - BlendedMVS
    - 5a0271884e62597cdee0d0eb
      - blended_images
      - cams
      - sparse_colmap
      - sparse_glomap
      - sparse
    - ...(similarly)
```
- **IMC2023** (dataset name: `imc2023`):  
```
- imc2023
  - <category>
    - <scene>
      - images
      - sfm
      - sparse_colmap
      - sparse_glomap
      - sparse
    - ...(similarly)
```
- **IMC2024** (dataset name: `imc2024`):  
```
- imc2024
  - train
    - all
      - <scene>
        - images
        - sfm
        - sparse_colmap
        - sparse_glomap
        - sparse
      - ...(similarly)
```
## 3. Run evaluation  
To run the evaluation, use the following command from the `instantsfm/eval/colmap_eval/` directory:  
```bash
cd instantsfm/eval/colmap_eval
python evaluate.py --data_path /path/to/dataset --datasets eth3d tt dtu
```
You can specify multiple datasets separated by spaces. The available dataset names are: `eth3d`, `tt`, `dtu`, `blended_mvs`, `imc2023`, `imc2024`.  
To restrict evaluation to specific categories or scenes, use `--categories` and `--scenes`:  
```bash
python evaluate.py --data_path /path/to/dataset --datasets eth3d --categories dslr --scenes botanical_garden
```
By default, relative pairwise pose errors are computed (AUC at 1°, 3°, 5°, 10°). To compute absolute pose errors instead, use `--error_type absolute`.  
The results are saved as a CSV file in the `--run_path` directory (defaults to `dataset/`). The report name can be specified with `--report_name`.

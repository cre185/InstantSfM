import gradio as gr
import os
import numpy as np
import plotly.graph_objects as go
from typing import Optional
from pathlib import Path
from datetime import datetime
import shutil
import subprocess

from instantsfm.utils.read_write_model import read_images_binary, read_images_text, read_points3D_binary, read_points3D_text, read_cameras_binary, read_cameras_text

def run_sfm_from_images(images: list[str]) -> Optional[dict]:
    # Function to run SfM reconstruction from given images
    # After copying images to a temporary directory, run the run_sfm function for result
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir = Path("demo_tmp") / f"run_{datetime_str}"
    img_dir = temp_dir / "images"
    img_dir.mkdir(parents=True)
    for img in images:
        img_path = Path(img)
        if img_path.exists():
            img_dest = img_dir / img_path.name
            shutil.copy(img_path, img_dest)
    
    return run_sfm(str(temp_dir))

def run_sfm(input_path: str) -> Optional[dict]:
    # Function to run SfM reconstruction
    images_dir = Path(input_path) / "images"
    if not images_dir.exists():
        return None
    
    # run reconstruction
    database_path = Path(input_path) / "database.db"
    if not database_path.exists():
        subprocess.run(["ins-feat", "--data_path", f"{input_path}"])
    subprocess.run(["ins-sfm", "--data_path", f"{input_path}"])
    
    # real data from sparse reconstruction result
    sparse_dir = Path(input_path) / "sparse" / "0"
    if not sparse_dir.exists():
        return None
    
    if os.path.exists(sparse_dir / "images.txt"):
        images = read_images_text(sparse_dir / "images.txt")
    elif os.path.exists(sparse_dir / "images.bin"):
        images = read_images_binary(sparse_dir / "images.bin")
    else:
        print("No images found in the sparse directory.")
        print(f"Tried dir {sparse_dir / 'images.txt'} and {sparse_dir / 'images.bin'}")
        return None
    
    if os.path.exists(sparse_dir / "cameras.txt"):
        cameras = read_cameras_text(sparse_dir / "cameras.txt")
    elif os.path.exists(sparse_dir / "cameras.bin"):
        cameras = read_cameras_binary(sparse_dir / "cameras.bin")
    else:
        print("No cameras found in the sparse directory.")
        print(f"Tried dir {sparse_dir / 'cameras.txt'} and {sparse_dir / 'cameras.bin'}")
        return None
    
    if os.path.exists(sparse_dir / "points3D.txt"):
        points3D = read_points3D_text(sparse_dir / "points3D.txt")
    elif os.path.exists(sparse_dir / "points3D.bin"):
        points3D = read_points3D_binary(sparse_dir / "points3D.bin")
    else:
        print("No points3D found in the sparse directory.")
        print(f"Tried dir {sparse_dir / 'points3D.txt'} and {sparse_dir / 'points3D.bin'}")
        return None

    return {
        "images": images,
        "cameras": cameras,
        "points3D": points3D,
    }

def create_3d_visualization(sfm_data: dict) -> go.Figure:
    fig = go.Figure()

    points3D = sfm_data["points3D"].values()
    points_xyz = np.array([p.xyz for p in points3D])
    points_rgb = np.array([p.rgb for p in points3D])
    points_id = np.array([p.id for p in points3D])
    
    rgb_colors = [f'rgb({r},{g},{b})' for r, g, b in points_rgb]

    hover_texts = [f'Point3D ID: {point_id}<br>XYZ: [{x:.2f}, {y:.2f}, {z:.2f}]<br>RGB: {rgb}' 
                  for point_id, (x, y, z), rgb in zip(points_id, points_xyz, rgb_colors)]
    
    fig.add_trace(go.Scatter3d(
        x=points_xyz[:, 0],
        y=points_xyz[:, 1],
        z=points_xyz[:, 2],
        mode='markers',
        marker=dict(
            size=0.5,
            color=rgb_colors,
            opacity=0.8
        ),
        text=hover_texts,
        hoverinfo='text',
        name='3D Points'
    ))

    images = sfm_data["images"].values()
    cameras = sfm_data["cameras"]
    cam_positions = []

    for img in images:
        qvec = img.qvec
        qw, qx, qy, qz = qvec
        
        R = np.array([
            [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
        ])

        camera_position = -R.T @ img.tvec
        cam_positions.append(camera_position)

        # Define camera intrinsic parameters
        cam = cameras[img.camera_id]
        focal_length = cam.params[0]  # Focal length, assuming it's the first parameter (sometimes it's fx)
        sensor_width = cam.width
        sensor_height = cam.height

        # Calculate image plane corners in camera space
        half_width = (sensor_width / 2) / focal_length
        half_height = (sensor_height / 2) / focal_length
        # scale down the frustum to a fixed smaller size for better visualization
        FRUSTUM_SCALE = 0.4
        corners_camera_space = np.array([
            [-half_width, -half_height, 1],
            [half_width, -half_height, 1],
            [half_width, half_height, 1],
            [-half_width, half_height, 1]
        ]) * FRUSTUM_SCALE

        # Transform corners to world space
        corners_world_space = (R.T @ corners_camera_space.T).T + camera_position

        # Draw edges of the view frustum
        edges = [
            (camera_position, corners_world_space[0]),
            (camera_position, corners_world_space[1]),
            (camera_position, corners_world_space[2]),
            (camera_position, corners_world_space[3]),
            (corners_world_space[0], corners_world_space[1]),
            (corners_world_space[1], corners_world_space[2]),
            (corners_world_space[2], corners_world_space[3]),
            (corners_world_space[3], corners_world_space[0]),
        ]

        for edge in edges:
            fig.add_trace(go.Scatter3d(
                x=[edge[0][0], edge[1][0]],
                y=[edge[0][1], edge[1][1]],
                z=[edge[0][2], edge[1][2]],
                mode='lines',
                line=dict(color='red', width=0.5),
                name=None
            ))
    
    cam_positions = np.array(cam_positions)
    all_points = np.vstack([points_xyz, cam_positions])
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)
    
    # calculate scene size and center
    center = (min_vals + max_vals) / 2
    size = np.max(max_vals - min_vals) / 2
    
    x_min, y_min, z_min = center - size
    x_max, y_max, z_max = center + size

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title='X',
                range=[x_min, x_max],
                autorange=False,
                visible=False
            ),
            yaxis=dict(
                title='Y',
                range=[y_min, y_max],
                autorange=False,
                visible=False
            ),
            zaxis=dict(
                title='Z',
                range=[z_min, z_max],
                autorange=False,
                visible=False
            ),
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        legend=dict(yanchor="top"),
        height=800,
        width=1200,
    )
    
    return fig

def process_images(images: list[str]):
    if not images:
        raise gr.Error("No images uploaded!")
    
    sfm_data = run_sfm_from_images(images)
    if sfm_data is None:
        raise gr.Error("SfM reconstruction from images failed!")
    
    return create_3d_visualization(sfm_data)

def process_folder(input_path: str):
    if not os.path.isdir(input_path):
        raise gr.Error("Invalid input path!")
    
    images_path = os.path.join(input_path, "images")
    if not os.path.exists(images_path):
        raise gr.Error("Images folder not found!")
    
    sfm_data = run_sfm(input_path)
    if sfm_data is None:
        raise gr.Error("SfM reconstruction failed!")
    
    return create_3d_visualization(sfm_data)


with gr.Blocks(title="InstantSfM") as demo:
    gr.Markdown("# InstantSfM Reconstruction Demo")
    
    with gr.Row():
        with gr.Column(scale=3):
            # from images
            image_uploader = gr.File(
                label="Upload images",
                file_types=["image"],
                file_count="multiple",
                height=400,
            )
            
            submit_btn_images = gr.Button("Start Reconstruction from Images", variant="primary")

            gr.Markdown(
                "### **OR**",
                elem_id="or-text",
                visible=True
            )

            # from folder
            input_dir = gr.Textbox(
                label="Input folder path",
                placeholder="input path here",
                type="text",
            )
            
            submit_btn_folder = gr.Button("Start Reconstruction from Folder", variant="primary")
            
        with gr.Column(scale=7):
            # create default figure
            fig = go.Figure()
            points = np.random.rand(100, 3) * 2 - 1
            fig.add_trace(go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=0.5,
                    color='white',
                    opacity=0.8
                ),
                hoverinfo='text',
                name='3D Points'
            ))
            fig.update_layout(
                scene=dict(
                    xaxis=dict(
                        title='X',
                        range=[-1, 1],
                        autorange=False,
                        visible=False
                    ),
                    yaxis=dict(
                        title='Y',
                        range=[-1, 1],
                        autorange=False,
                        visible=False
                    ),
                    zaxis=dict(
                        title='Z',
                        range=[-1, 1],
                        autorange=False,
                        visible=False
                    ),
                    aspectmode='cube',
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    )
                ),
                legend=dict(yanchor="top"),
                height=800,
                width=1200,
            )

            plot_output = gr.Plot(
                label="Reconstruction result",
                value=fig,
            )
    
    submit_btn_images.click(
        fn=process_images,
        inputs=image_uploader,
        outputs=plot_output,
        api_name="sfm_reconstruction"
    )
    submit_btn_folder.click(
        fn=process_folder,
        inputs=input_dir,
        outputs=plot_output,
        api_name="sfm_reconstruction"
    )

if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)
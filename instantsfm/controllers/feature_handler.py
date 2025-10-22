import numpy as np
import os
import time
import tqdm
import concurrent.futures

from instantsfm.utils import database
from instantsfm.scene.defs import Ids2PairId, PairId2Ids

def GenerateDatabase(image_path, database_path, feature_handler_name, config, single_camera=False, camera_per_folder=False):
    # colmap support from command line. ensure colmap is installed
    if feature_handler_name == 'colmap':
        import subprocess
        feature_extractor_cmd = [
            'colmap', 'feature_extractor',
            '--image_path', image_path,
            '--database_path', database_path,
            '--ImageReader.camera_model', 'SIMPLE_RADIAL',
            '--ImageReader.single_camera', '1' if single_camera else '0',
            '--ImageReader.single_camera_per_folder', '1' if (not single_camera and camera_per_folder) else '0'
        ]
        exhaustive_matcher_cmd = [
            'colmap', 'exhaustive_matcher',
            '--database_path', database_path
        ]
        sequential_matcher_cmd = [
            'colmap', 'sequential_matcher',
            '--database_path', database_path
        ]
        use_exhaustive = True
        matcher_cmd = exhaustive_matcher_cmd if use_exhaustive else sequential_matcher_cmd

        try:
            print(f"Feature extraction started for {image_path} by {feature_extractor_cmd}")
            subprocess.run(feature_extractor_cmd, check=True)
            print(f"Feature extraction completed for {image_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error during feature extraction: {e}")
        try:
            subprocess.run(matcher_cmd, check=True)
            print(f"Exhaustive matching completed for {database_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error during exhaustive matching: {e}")
        return
    if feature_handler_name == 'dedode':
        import cv2
        import kornia as K
        import kornia.feature as KF
        import torch
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        start_time = time.time()
        image_name_list = os.listdir(image_path)
        image_list = [K.io.load_image(os.path.join(image_path, image_name), K.io.ImageLoadType.RGB32)[None, ...] for image_name in image_name_list]
        images = torch.cat(image_list, dim=0).to(torch.float16).to(device)
        if config.OPTIONS['uniform_camera']:
            dedode = KF.DeDoDe.from_pretrained(detector_weights="L-upright", descriptor_weights="B-upright", amp_dtype=torch.float16).to(device)
            # images = N x 3 x H x W, treat images in batch
            batch_size = 4 
            progress_bar = tqdm.tqdm(total=len(image_list))
            num_remaining_images = len(image_list) % batch_size
            B = 4096
            if num_remaining_images > 0:
                image_batch = images[:num_remaining_images]
                keypoints, scores, descriptions = dedode(image_batch, n=B)
                progress_bar.update(num_remaining_images)
            else: 
                keypoints = torch.tensor([]).to(device)
                descriptions = torch.tensor([]).to(device)
            for i in range(num_remaining_images, len(image_list), batch_size):
                image_batch = images[i:i+batch_size]
                keypoints_batch, scores_batch, descriptions_batch = dedode(image_batch, n=B)
                keypoints = torch.cat([keypoints, keypoints_batch], dim=0) if keypoints is not None else keypoints_batch
                descriptions = torch.cat([descriptions, descriptions_batch], dim=0) if descriptions is not None else descriptions_batch
                progress_bar.update(batch_size)
            progress_bar.close()
            del dedode
            torch.cuda.empty_cache()

            height, width = images.shape[2:]
            focal_length = min(width, height) * 1.2
            cx, cy = width / 2, height / 2
            params = np.array([focal_length, cx, cy])
            print(f"Feature extraction done in {time.time() - start_time} seconds")

            progress_bar = tqdm.tqdm(total=((len(image_list)-1)*len(image_list)) // 2)
            matches = {}
            matcher = KF.DescriptorMatcher(match_mode='snn', th=0.92).to(device)
            for idx1 in range(len(image_list)):
                for idx2 in range(idx1):
                    des1, des2 = descriptions[idx1], descriptions[idx2]
                    pair_matches = matcher(des1, des2)
                    matches[Ids2PairId(idx1, idx2)] = pair_matches[1].cpu().numpy().astype(np.uint32)
                    progress_bar.update(1)
            
            '''matcher = KF.DescriptorMatcher(match_mode='snn', th=0.94).to(device)
            batch_size = 16
            for idx1 in range(len(image_list)):
                for idx2 in range(0, idx1, batch_size):
                    end_idx = min(idx2 + batch_size, idx1)
                    des1 = descriptions[idx1]
                    des2_batch = descriptions[idx2:end_idx].view(-1, descriptions.shape[-1])
                    pair_matches_batch = matcher(des1, des2_batch)[1]
                    for j in range(end_idx - idx2):
                        matches_for_image = pair_matches_batch[(pair_matches_batch[:, 1] >= j * B) 
                                                               & (pair_matches_batch[:, 1] < (j + 1) * B)]
                        matches_for_image[:, 1] = matches_for_image[:, 1] % B
                        matches[Ids2PairId(idx1, idx2 + j)] = matches_for_image.cpu().numpy().astype(np.uint32)
                    progress_bar.update(end_idx - idx2)'''
            
            progress_bar.close()
            del matcher
            torch.cuda.empty_cache()
            print(f"Matching done in {time.time() - start_time} seconds")

            keypoints = keypoints.cpu().numpy()
            descriptions = descriptions.cpu().numpy()
        else:
            dedode = KF.DeDoDe.from_pretrained(detector_weights="L-upright", descriptor_weights="B-upright", amp_dtype=torch.float16).to(device)
            # treat images one by one
            progress_bar = tqdm.tqdm(total=len(image_list))
            keypoints = torch.tensor([]).to(device)
            descriptions = torch.tensor([]).to(device)
            for i in range(len(image_list)):
                image = images[i:i+1]
                keypoint, score, description = dedode(image, n=4096)
                keypoints = torch.cat([keypoints, keypoint], dim=0) if keypoints is not None else keypoint
                descriptions = torch.cat([descriptions, description], dim=0) if descriptions is not None else description
                progress_bar.update(1)
            progress_bar.close()
            del dedode
            torch.cuda.empty_cache()

            height, width = images.shape[2:]
            focal_length = min(width, height) * 1.2
            cx, cy = width / 2, height / 2
            params = np.array([focal_length, cx, cy])
            print(f"Feature extraction done in {time.time() - start_time} seconds")

            progress_bar = tqdm.tqdm(total=((len(image_list)-1)*len(image_list)) // 2)
            matches = {}
            matcher = KF.DescriptorMatcher(match_mode='snn', th=0.92).to(device)
            for idx1 in range(len(image_list)):
                for idx2 in range(idx1):
                    des1, des2 = descriptions[idx1], descriptions[idx2]
                    pair_matches = matcher(des1, des2)
                    matches[Ids2PairId(idx1, idx2)] = pair_matches[1].cpu().numpy().astype(np.uint32)
                    progress_bar.update(1)
            
            progress_bar.close()
            del matcher
            torch.cuda.empty_cache()
            print(f"Matching done in {time.time() - start_time} seconds")

            keypoints = keypoints.cpu().numpy()
            descriptions = descriptions.cpu().numpy()
    elif feature_handler_name == 'superpoint+lightglue' or feature_handler_name == 'disk+lightglue':
        import cv2
        import kornia as K
        import kornia.feature as KF
        import torch
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        start_time = time.time()
        image_name_list = os.listdir(image_path)
        image_list = [K.io.load_image(os.path.join(image_path, image_name), K.io.ImageLoadType.RGB32)[None, ...] for image_name in image_name_list]
        images = torch.cat(image_list, dim=0).to(device)
        # this method assumes all images have the same size
        matcher = KF.DISK().to(device) if feature_handler_name == 'disk+lightglue' else KF.SuperPoint().to(device)
        batch_size = 8
        progress_bar = tqdm.tqdm(total=len(image_list))
        keypoints, descriptions = [], []
        for i in range(0, len(image_list), batch_size):
            image_batch = images[i:i+batch_size] if i+batch_size < len(image_list) else images[i:]
            with torch.no_grad():
                features = matcher(image_batch, n=4096, pad_if_not_divisible=True)
            for feature in features:
                keypoints.append(feature.keypoints)
                descriptions.append(feature.descriptors)
            progress_bar.update(image_batch.shape[0])
        
        progress_bar.close()
        del matcher

        height, width = images.shape[2:]
        focal_length = min(width, height) * 1.2
        cx, cy = width / 2, height / 2
        params = np.array([focal_length, cx, cy])
        print(f"Feature extraction done in {time.time() - start_time} seconds")

        matches = {}
        matcher = KF.LightGlue(features='disk' if feature_handler_name == 'disk+lightglue' else 'superpoint').to(device)
        image_size = torch.tensor([width, height], device=device).view(1, 2)

        def get_data(idx1, idx2):
            kpts1, des1 = keypoints[idx1].unsqueeze(0), descriptions[idx1].unsqueeze(0)
            kpts2, des2 = keypoints[idx2].unsqueeze(0), descriptions[idx2].unsqueeze(0)
            return {'image0': {'keypoints': kpts1, 'descriptors': des1, 'image_size': image_size},
                    'image1': {'keypoints': kpts2, 'descriptors': des2, 'image_size': image_size}}

        progress_bar = tqdm.tqdm(total=((len(image_list)-1)*len(image_list)) // 2)
        for idx1 in range(len(image_list)):
            for idx2 in range(idx1):
                match = matcher(get_data(idx1, idx2))['matches']
                match = match[0].cpu().numpy().astype(np.uint32)
                matches[Ids2PairId(idx1, idx2)] = match
                progress_bar.update(1)

        progress_bar.close()
        del matcher
        print(f"Matching done in {time.time() - start_time} seconds")

        keypoints = [kpt.cpu().numpy() for kpt in keypoints]
        descriptions = [des.cpu().numpy() for des in descriptions]
    elif feature_handler_name == 'sift':
        import cv2
        import kornia as K
        import kornia.feature as KF
        import torch
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        start_time = time.time()
        image_name_list = os.listdir(image_path)
        image_list = [K.io.load_image(os.path.join(image_path, image_name), K.io.ImageLoadType.RGB32)[None, ...] for image_name in image_name_list]
        images = torch.cat(image_list, dim=0).to(device)
        images = K.color.rgb_to_grayscale(images)

        # this method assumes all images have the same size
        SIFT = KF.SIFTFeatureScaleSpace(num_features=4096).to(device)
        progress_bar = tqdm.tqdm(total=len(image_list))
        lafs, keypoints, descriptors = [], [], []
        for image in images:
            with torch.no_grad():
                feature = SIFT(image.unsqueeze(0))
            lafs.append(feature[0])
            keypoints.append(feature[0].squeeze(0)[:, :, -1])
            descriptors.append(feature[2].squeeze(0))
            progress_bar.update(1)
        
        progress_bar.close()
        del SIFT

        '''import matplotlib.pyplot as plt
        idx = 0
        image = image_list[idx][0].cpu().numpy().transpose(1, 2, 0)
        kpts = keypoints[idx].cpu().numpy().astype(np.int32)

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(image)
        ax.plot(kpts[:, 0], kpts[:, 1], 'ro', markersize=2)
        plt.title('Keypoints in the first image')
        plt.show()'''

        height, width = images.shape[2:]
        focal_length = min(width, height) * 1.2
        cx, cy = width / 2, height / 2
        params = np.array([focal_length, cx, cy])
        print(f"Feature extraction done in {time.time() - start_time} seconds")

        matches = {}
        matcher = KF.DescriptorMatcher(match_mode='snn', th=0.9).to(device)

        progress_bar = tqdm.tqdm(total=((len(image_list)-1)*len(image_list)) // 2)
        for idx1 in range(len(image_list)):
            for idx2 in range(idx1):
                desc1, desc2 = descriptors[idx1], descriptors[idx2]
                match = matcher(desc1, desc2)
                match = match[1].cpu().numpy().astype(np.uint32)
                matches[Ids2PairId(idx1, idx2)] = match
                progress_bar.update(1)

        progress_bar.close()
        del matcher

        '''idx1, idx2 = 0, 1
        image1 = image_list[idx1][0].cpu().numpy().transpose(1, 2, 0)
        image2 = image_list[idx2][0].cpu().numpy().transpose(1, 2, 0)
        kpts1 = keypoints[idx1].cpu().numpy()
        kpts2 = keypoints[idx2].cpu().numpy()
        match = matches[Ids2PairId(idx1, idx2)]

        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        img_combined = np.hstack((image1, image2))
        ax.imshow(img_combined)
        for m in match:
            pt1 = kpts1[m[0]]
            pt2 = kpts2[m[1]] + np.array([image1.shape[1], 0])
            ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'r-', linewidth=0.5)
            ax.plot(pt1[0], pt1[1], 'bo', markersize=2)
            ax.plot(pt2[0], pt2[1], 'bo', markersize=2)
        plt.show()'''
        print(f"Matching done in {time.time() - start_time} seconds")

        keypoints = [laf.squeeze(0)[:, :, -1].cpu().numpy().astype(np.int32) for laf in lafs]
        descriptions = [descriptor.cpu().numpy() for descriptor in descriptors]

    db = database.COLMAPDatabase.connect(database_path)
    db.create_tables()

    image_id_list = []
    camera_data = []
    image_data = []
    keypoints_data = []
    descriptors_data = []

    if config.OPTIONS['uniform_camera']:
        camera_data.append((0, width, height, params))
        for idx in range(len(image_list)):
            image_data.append((image_name_list[idx], 1))
            image_id = idx + 1
            image_id_list.append(image_id)
            kpt, des = keypoints[idx], descriptions[idx]
            keypoints_data.append((image_id, kpt))
            descriptors_data.append((image_id, des))
    else:
        for idx in range(len(image_list)):
            camera_data.append((0, width, height, params))
            cam_id = idx + 1
            image_data.append((image_name_list[idx], cam_id))
            image_id = idx + 1
            image_id_list.append(image_id)
            kpt, des = keypoints[idx], descriptions[idx]
            keypoints_data.append((image_id, kpt))
            descriptors_data.append((image_id, des))
    db.add_cameras_batch(camera_data)
    db.add_images_batch(image_data)
    db.add_keypoints_batch(keypoints_data)
    db.add_descriptors_batch(descriptors_data)
    print(f"Images added to database in {time.time() - start_time} seconds")

    match_data = []
    geometry_data = []
    def match_handling(pair_id, match):
        idx1, idx2 = PairId2Ids(pair_id)
        image_id1, image_id2 = image_id_list[idx1], image_id_list[idx2]
        match_data.append((image_id1, image_id2, match))
        if len(match) >= config.FEATURE_HANDLER_OPTIONS['min_num_matches']:
            geometry_data.append((image_id1, image_id2, match))
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(match_handling, pair_id, match) for pair_id, match in matches.items()]
    db.add_matches_batch(match_data)
    db.add_two_view_geometries_batch(geometry_data)
    print(f"Two view geometries added to database in {time.time() - start_time} seconds")

    db.add_feature_name(feature_handler_name)
    db.commit()
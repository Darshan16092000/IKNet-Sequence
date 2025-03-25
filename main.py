import os
import sys
import numpy as np
import time
import cv2

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from datasets.hot3d import HOT3DSequence, HOT3DDataset
# from models.sapiens.classes_and_palettes import get_class_names_and_colors


def main():
    dataset_path = '/netscratch/millerdurai/Datasets/HOT3D/dataset' 

    sequence_name = "P0001_10a27bf7"
    # sequence_name = 'P0001_15c4300c'
    # sequence_name = 'P0001_23fa0ee8'

    dataset = HOT3DDataset(dataset_path, 'train')
    n_items = len(dataset)

    # dataset = HOT3DSequence(dataset_path, sequence_name)
    for i in range(0, 400):
        i = np.random.randint(0, n_items)
        # try:
        t = time.time()
        data = dataset[i]
        t = time.time() - t
        print('Time taken for getting data:', t)

    #         rgb = data['extras']['rgb_np']
    #         undistorted_rgb = data['extras']['undistorted_rgb_np']

    #         raw_hand_mask = data['raw_hand_mask'].permute(1, 2, 0).numpy().astype(dtype=np.uint8) 
    #         undistorted_hand_mask = data['undistorted_hand_mask'].permute(1, 2, 0).numpy().astype(dtype=np.uint8) 

    #         raw_arm_mask = data['raw_arm_mask'].permute(1, 2, 0).numpy().astype(dtype=np.uint8) 
    #         undistorted_arm_mask = data['undistorted_arm_mask'].permute(1, 2, 0).numpy().astype(dtype=np.uint8) 

    #         colors = [
    #             [
    #                 [128, 200, 255],
    #                 [255, 0, 109],
    #             ],
    #             [
    #                 [0, 255, 36],
    #                 [189, 0, 204],
    #             ]
    #             ]
    #         opacity = 0.3
            
    #         mask = np.zeros_like(rgb)
    #         for idx, _ in enumerate(colors):
    #             mask[raw_hand_mask[:, :, idx] == 1, :] = colors[idx][0]
    #             mask[raw_arm_mask[:, :, idx] == 1, :] = colors[idx][1]
    #         rgb = (rgb * (1 - opacity) + mask * opacity).astype(np.uint8)


    #         mask = np.zeros_like(undistorted_rgb)
    #         for idx, _ in enumerate(colors):
    #             mask[undistorted_hand_mask[:, :, idx] == 1, :] = colors[idx][0]
    #             mask[undistorted_arm_mask[:, :, idx] == 1, :] = colors[idx][1]
    #         undistorted_rgb = (undistorted_rgb * (1 - opacity) + mask * opacity).astype(np.uint8)

    #         video.write(np.hstack((rgb, undistorted_rgb))[:, :, ::-1])

    #         # video_fisheye.write(rgb[:, :, ::-1])
    #         # video_undistort.write(undistorted_rgb[:, :, ::-1])

    #     except Exception as e:
    #         print('Error:', e)

    #         break

    # video.release()
    # # video_undistort.release()


if __name__  == '__main__':
    main()

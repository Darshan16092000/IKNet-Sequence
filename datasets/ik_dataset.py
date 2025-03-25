import os; import sys; 
ROOT_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(ROOT_DIR)

import torch
import os
import numpy as np
import cv2
import random

from collections import OrderedDict
from torch.utils.data import Dataset
from dis_est_utils.metrics import compute_3d_errors_batch


class IKDataset(Dataset):
    def __init__(self, cfg, target_datset, split='train', sequence_length=30):
        self.cfg = cfg
        self.split = split
        self.sequence_length = sequence_length

        assert self.split in ['train', 'val', 'test'], "Invalid split. Must be one of ['train', 'val', 'test']"

        self.is_train = self.split == 'train'
        self.target_dataset = target_datset

        self.num_joints = cfg.NUM_JOINTS
        self.length = len(self.target_dataset)

    def __len__(self):
        return self.length
    
    # def __getitem__(self, index):
    #     # Ensure the index is valid for the sequence length
    #     # print(f"Original index: {index}")
    #     # print(f"Sequence Length: {self.sequence_length}")
    #     # print(f"Dataset Length: {self.length}")
    #     # print(f"Sum of Index + Sequence Length: {index + self.sequence_length}")

    #     if index + self.sequence_length > self.length:
    #         index = self.length - self.sequence_length
    #         # print(f"Adjusted index for sequence length: {index}")

    #     data_list = []  # frames of data
    #     meta = []  # frames of meta
    #     for i in range(self.sequence_length):
    #         # print(f"Processing frame {i+1} in the sequence (index: {index + i})")

    #         frame_data = self.target_dataset[index + i]
    #         fisheye624_params = frame_data['fisheye624_params']
    #         sequence_name = frame_data['extras']['sequence_name']
    #         annotation_key = frame_data['extras']['annotation_key']

    #         hand_params = frame_data['hand_params']
    #         hand_type = random.choice(list(hand_params.keys()))
    #         # print(f"Frame data: {frame_data}")
    #         # print(f"Fisheye624 parameters dtype: {type(fisheye624_params)}")
    #         # print(f"Sequence name dtype: {type(sequence_name)}")
    #         # print(f"Hand parameters dtype: {type(hand_params)}")
    #         # print(f"Selected hand type: {hand_type}")

    #         hand_param = hand_params[hand_type]
    #         hand_pose_pca = hand_param['hand_pose_pca']
    #         hand_pose_aa = hand_param['hand_pose_aa']
    #         global_orient = hand_param['camera_global_orient']
    #         transl = hand_param['camera_transl']
    #         betas = hand_param['betas']
    #         j3d = hand_param['camera_j3D']
    #         hand_pose_aa = torch.tensor(hand_pose_aa).float()
    #         global_orient = torch.tensor(global_orient).float()

    #         # print(f"Hand pose PCA dtype: {type(hand_pose_pca)}, shape: {np.array(hand_pose_pca).shape}")
    #         # print(f"Hand pose axis-angle dtype: {type(hand_pose_aa)}, shape: {np.array(hand_pose_aa).shape}")
    #         # print(f"Global orientation dtype: {type(global_orient)}, shape: {np.array(global_orient).shape}")
    #         # print(f"Translation dtype: {type(transl)}, shape: {np.array(transl).shape}")
    #         # print(f"Betas dtype: {type(betas)}, shape: {np.array(betas).shape}")
    #         # print(f"3D joint positions dtype: {type(j3d)}, shape: {np.array(j3d).shape}")
    #         # print(f"Hand pose axis-angle tensor dtype: {hand_pose_aa.dtype}, shape: {hand_pose_aa.shape}")
    #         # print(f"Global orientation tensor dtype: {global_orient.dtype}, shape: {global_orient.shape}")

    #         frame_meta = {
    #             'sequence_name': sequence_name,
    #             'annotation_key': annotation_key,
    #             'fisheye624_params': fisheye624_params,
    #             'j3d': j3d,
    #             'hand_pose_pca': hand_pose_pca,
    #             'hand_type': hand_type,
    #             'betas': betas,
    #             'transl': transl,
    #         }
    #         # print(f"Frame meta keys: {frame_meta.keys()}")

    #         palm_length = np.linalg.norm(j3d[9] - j3d[0])
    #         root_relative_j3d = j3d - j3d[0]
    #         palm_normalized_root_relative_j3d = root_relative_j3d / palm_length
    #         palm_normalized_root_relative_j3d = torch.tensor(palm_normalized_root_relative_j3d).float()

    #         # print(f"Palm length dtype: {type(palm_length)}")
    #         # print(f"Root-relative 3D joint positions dtype: {type(root_relative_j3d)}, shape: {root_relative_j3d.shape}")
    #         # print(f"Palm-normalized root-relative 3D joint positions dtype: {type(palm_normalized_root_relative_j3d)}, shape: {palm_normalized_root_relative_j3d.shape}")
    #         # print(f"Palm-normalized root-relative 3D joint positions tensor dtype: {palm_normalized_root_relative_j3d.dtype}, shape: {palm_normalized_root_relative_j3d.shape}")

    #         frame_data = {
    #             'pr_j3d': palm_normalized_root_relative_j3d,
    #             'global_orient': global_orient,
    #             'hand_pose_aa': hand_pose_aa,
    #         }
    #         # print(f"Frame data (processed) keys: {frame_data.keys()}")

    #         data_list.append(frame_data)
    #         meta.append(frame_meta)
    #         # print(f"Data list length: {len(data_list)}")
    #         # print(f"Meta list length: {len(meta)}")


    #     # Combine tensors for each frame of data
    #     combined_pr_j3d = torch.stack([frame['pr_j3d'] for frame in data_list])
    #     combined_global_orient = torch.stack([frame['global_orient'] for frame in data_list])
    #     combined_hand_pose_aa = torch.stack([frame['hand_pose_aa'] for frame in data_list])

    #     data = {
    #         'pr_j3d': combined_pr_j3d,
    #         'global_orient': combined_global_orient,
    #         'hand_pose_aa': combined_hand_pose_aa,
    #     }
    #     # print(data['pr_j3d'].shape)
    #     # print(f"Combined data keys: {data.keys()}")
    #     # print(f"Combined pr_j3d shape: {combined_pr_j3d.shape}")
    #     # print(f"Combined global_orient shape: {combined_global_orient.shape}")
    #     # print(f"Combined hand_pose_aa shape: {combined_hand_pose_aa.shape}")

    #     # Combine meta data (assuming meta data is not tensor and should be kept as list)
    # #     combined_meta = {
    # #     'sequence_name': [m['sequence_name'] for m in meta],
    # #     'annotation_key': [m['annotation_key'] for m in meta],
    # #     'hand_type': [m['hand_type'] for m in meta],
    # #     'fisheye624_params': [m['fisheye624_params'] for m in meta],
    # #     'j3d': [m['j3d'] for m in meta],
    # #     'hand_pose_pca': [m['hand_pose_pca'] for m in meta],
    # #     'betas': [m['betas'] for m in meta],
    # #     'transl': [m['transl'] for m in meta],
    # # }
        
    #     # print(f"Meta Type: {type(meta)}")  # Should be a list
    #     # print(f"Meta Length: {len(meta)}")  # Should be 30 (sequence length)
    #     # print(f"Meta[0] Type: {type(meta[0])}")  # Should be a dictionary
    #     # print(f"Meta[0] Keys: {meta[0].keys()}")  # Should show valid keys
    #     # print(f"Meta[0] Sample: {meta[0]}")  # Print one sample for verification

    #     # combined_meta = {key: [] for key in meta[0]}  # Initialize empty lists for each key

    #     # for sequence in meta:  # Loop over batch size (should be 32 sequences)
    #     #     for key in sequence:
    #     #         combined_meta[key].append(sequence[key])

 
    #     # combined_meta = {key: [frame[key] for frame in meta] for key in meta[0].keys()}
    #     combined_meta = {
    #     'hand_type': [m['hand_type'] for m in meta],  # List of 30 hand_types
    #     'sequence_name': [m['sequence_name'] for m in meta],
    #     'annotation_key': [m['annotation_key'] for m in meta],
    #     'j3d': [m['j3d'] for m in meta],
    #     # Add other keys as needed...
    #     }
        
    #     # combined_meta = meta


    #     # print("\n" + "="*40 + " COMBINED_META STRUCTURE " + "="*40)
    #     # print(f"Total keys in combined_meta: {len(combined_meta.keys())}")
        
    #     # for key in combined_meta:
    #     #     print("\n" + "-"*30)
    #     #     print(f"Key: {key}")
    #     #     print(f"Number of items: {len(combined_meta[key])} (should be {self.sequence_length})")
    #     #     print(f"First 3 items:")
            
    #     #     # Print first 3 elements with type info
    #     #     for i, item in enumerate(combined_meta[key][:3]):
    #     #         print(f"[{i}] {type(item)}: {str(item)[:50]}...")  # Truncate long values
                
    #     # print("="*90 + "\n")

    #     # for frame_meta in combined_meta:
    #     #     print(f"Sequence Name: {frame_meta['sequence_name']}")
    #     #     print(f"Hand Type: {frame_meta['hand_type']}")
    #     #     print(f"Annotation Key: {frame_meta['annotation_key']}")
    #     #     print("-" * 50)  # Separator for readability
    #     # exit()

    #     # # print(f"Total frames processed: {len(data_list)}")
    #     # print(f"Combined meta length: {len(combined_meta)}")
    #     # exit()
        
        

    #     return data, combined_meta
    
    def __getitem__(self, index):
        data_list = []  # processed frames data
        meta = []       # corresponding metadata

        i = index
        while len(data_list) < self.sequence_length:
            if i >= self.length:
                # Instead of raising an error, pad the sequence with the last valid frame.
                if not data_list:
                    # No valid frames found at all.
                    raise IndexError(f"No valid left-hand frames found starting from index {index}.")
                while len(data_list) < self.sequence_length:
                    data_list.append(data_list[-1])
                    meta.append(meta[-1])
                break

            frame_data = self.target_dataset[i]
            i += 1  # move to the next frame

            # Skip frame if left-hand data is not present.
            if 'left' not in frame_data['hand_params']:
                continue

            fisheye624_params = frame_data['fisheye624_params']
            sequence_name = frame_data['extras']['sequence_name']
            annotation_key = frame_data['extras']['annotation_key']

            hand_params = frame_data['hand_params']
            # Use left-hand only
            hand_type = 'left'
            hand_param = hand_params[hand_type]
            hand_pose_pca = hand_param['hand_pose_pca']
            hand_pose_aa = hand_param['hand_pose_aa']
            global_orient = hand_param['camera_global_orient']
            transl = hand_param['camera_transl']
            betas = hand_param['betas']
            j3d = hand_param['camera_j3D']

            # Convert to tensors
            hand_pose_aa = torch.tensor(hand_pose_aa).float()
            global_orient = torch.tensor(global_orient).float()

            frame_meta = {
                'sequence_name': sequence_name,
                'annotation_key': annotation_key,
                'fisheye624_params': fisheye624_params,
                'j3d': j3d,
                'hand_pose_pca': hand_pose_pca,
                'hand_type': hand_type,
                'betas': betas,
                'transl': transl,
            }

            # Process the 3D joints: compute palm length and normalize.
            palm_length = np.linalg.norm(j3d[9] - j3d[0])
            root_relative_j3d = j3d - j3d[0]
            palm_normalized_root_relative_j3d = root_relative_j3d / palm_length
            palm_normalized_root_relative_j3d = torch.tensor(palm_normalized_root_relative_j3d).float()

            frame_processed = {
                'pr_j3d': palm_normalized_root_relative_j3d,
                'global_orient': global_orient,
                'hand_pose_aa': hand_pose_aa,
            }

            data_list.append(frame_processed)
            meta.append(frame_meta)

        # Combine the sequence frames into tensors
        combined_pr_j3d = torch.stack([frame['pr_j3d'] for frame in data_list])
        combined_global_orient = torch.stack([frame['global_orient'] for frame in data_list])
        combined_hand_pose_aa = torch.stack([frame['hand_pose_aa'] for frame in data_list])

        data = {
            'pr_j3d': combined_pr_j3d,
            'global_orient': combined_global_orient,
            'hand_pose_aa': combined_hand_pose_aa,
        }

        combined_meta = {
            'hand_type': [m['hand_type'] for m in meta],
            'sequence_name': [m['sequence_name'] for m in meta],
            'annotation_key': [m['annotation_key'] for m in meta],
            'j3d': [m['j3d'] for m in meta],
        }

        return data, combined_meta


    def evaluate_joints(cls, cfg, all_gt_j3ds, all_preds_j3d, all_vis_j3d):
        errors, errors_pa = compute_3d_errors_batch(all_gt_j3ds, all_preds_j3d, all_vis_j3d)
        
        MPJPE = np.mean(errors) 
        PAMPJPE = np.mean(errors_pa)

        name_values = []

        for i in range(cfg.NUM_JOINTS):
            name_values.append((f'Joint_{i}_MPJPE', errors[i]))
        name_values.append(('MPJPE', MPJPE))
            
        for i in range(cfg.NUM_JOINTS):
            name_values.append((f'Joint_{i}_PAMPJPE', errors_pa[i]))
        name_values.append(('PAMPJPE', PAMPJPE))


        # heatmap_sequence = ["Head", # 0
        #                     "Neck", # 1
        #                     "Right_shoulder", # 2 
        #                     "Right_elbow", # 3
        #                     "Right_wrist", # 4
        #                     "Left_shoulder", # 5
        #                     "Left_elbow", # 6
        #                     "Left_wrist", # 7
        #                     "Right_hip", # 8
        #                     "Right_knee", # 9
        #                     "Right_ankle", # 10
        #                     "Right_foot", # 11
        #                     "Left_hip", # 12 
        #                     "Left_knee", # 13
        #                     "Left_ankle", #14
        #                     "Left_foot"] # 15

        # for i, joint_name in enumerate(heatmap_sequence):
        #     name_values.append((f'{joint_name}_MPJPE', errors[i]))
        # name_values.append(('MPJPE', MPJPE))

        # for i, joint_name in enumerate(heatmap_sequence):
        #     name_values.append((f'{joint_name}_PAMPJPE', errors_pa[i]))
        # name_values.append(('PAMPJPE', PAMPJPE))

        name_values = OrderedDict(name_values)

        return name_values, MPJPE


def main():
    from settings import config as cfg
    from datasets import HOT3DLoader 

    train_dataset = IKDataset(cfg, HOT3DLoader(cfg.DATASET.ROOT, get_camera=True, split='train'))

    for i in range(len(train_dataset)):
        data, meta = train_dataset[i]
        print(data.keys())
        print(meta.keys())
        break


if __name__ == '__main__':
    main()
import os
import numpy as np
import cv2
import platform
import torch
import tarfile
import json
import sys

from torch.utils.data import Dataset
from tqdm import tqdm, trange
from concurrent.futures import ThreadPoolExecutor

if platform.system() == 'Windows':
    PATH_OF_HOT3D_LIB = 'F:/Datasets/HOT3D/hot3d/hot3d'
else:
    PATH_OF_HOT3D_LIB = '/netscratch/millerdurai/Datasets/HOT3D/hot3d/hot3d'

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PATH_OF_HOT3D_LIB)
sys.path.append(ROOT_DIR)



from ik_net.datasets.ik_data_utils import PCATOAA
from settings import config as cfg

from datapipes.base_pipeline import BasePipelineCreator
from datapipes.decoders.image_decoder import ImageDecoder

    
def build_tar_index(tar_path):
    index = {}
    with tarfile.open(tar_path, 'r') as tar:
        for idx, tarinfo in enumerate(tqdm(tar, desc=f"Building tar index")):
            
            # key = f"{tarinfo.name}"
            # c = input(key)
            # if c == 'q': return index
            #if idx > 100: break

            index[tarinfo.name] = (tar_path, tarinfo.offset_data, tarinfo.size)
    
    return index


class SamplerLoader:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tarfiles_dict = dict()
        self.image_decoder = ImageDecoder('rgb8') 

    def load(self, tar_info, type):
        tar_path, offset_data, data_size = tar_info

        if tar_path not in self.tarfiles_dict:            
            file_obj = open(tar_path, 'rb')
            tar_size = os.path.getsize(tar_path)
            # mmap_obj = mmap.mmap(file_obj.fileno(), tar_size, access=mmap.ACCESS_READ)

            self.tarfiles_dict[tar_path] = {
                                            'file_obj': file_obj, 
                                            # 'mmap_obj': mmap_obj
                                            }
 
        tar_file_data = self.tarfiles_dict[tar_path]

        # file_data = tar_file_data['mmap_obj'][offset_data:offset_data + data_size]

        tar_file_data['file_obj'].seek(offset_data)
        file_data = tar_file_data['file_obj'].read(data_size)

        if 'png' in type or 'jpg' in type:
            data = self.image_decoder("file_path", file_data)
        elif 'json' in type:
            data = json.loads(file_data)

        return data
    

class HOT3DLoader(Dataset):
    def __init__(self, data_root, split, get_camera=False) -> None:
        self.data_root = data_root
        self.split = split

        assert self.split in ['train', 'val', 'test']

        self.factory = BasePipelineCreator(data_root)

        tar_index_map = dict()
        tar_file_paths = self.factory.get_tar_files_for_subsets(self.split, 
                                                                component_groups=["annotations"],)

        with ThreadPoolExecutor() as executor:
            for component_name, tar_paths in tar_file_paths.items():
                tar_index_map[component_name] = dict()

                for tar_index in executor.map(build_tar_index, tar_paths):
                    tar_index_map[component_name].update(tar_index)
        
        self.sample_loader = SamplerLoader()

        self.tar_index_map = tar_index_map
        self.anno_keys = list(tar_index_map['annotations'].keys())
        self.n_samples = len(self.anno_keys)

        self.pca_to_aa = PCATOAA(cfg)

        print('Number of samples:', self.n_samples)

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        annotation_key = self.anno_keys[index]

        sequence_name = annotation_key.split('.')[0]        
        annotation_info = self.tar_index_map['annotations'][annotation_key]
        annotation = self.sample_loader.load(annotation_info, type='json')

        camera_params = annotation['rgb_camera_params']

        hand_params = dict()
        for hand_type, hand_param in annotation['rgb_hand_params'].items():
            # if hand_type == 'left':  # Only process the left hand
                hand_pose_pca = np.array(hand_param['hand_pose'])
                hand_params[hand_type] = {
                    'hand_pose_pca': hand_pose_pca,
                    'hand_pose_aa': self.pca_to_aa(hand_type, hand_pose_pca),
                    'camera_transl': np.array(hand_param['camera_transl']),
                    'betas': np.array(hand_param['betas']),
                    'camera_j3D': np.array(hand_param['camera_j3D']),
                'camera_global_orient': np.array(hand_param['camera_global_orient']),
                }

        data = dict()
        data['hand_params'] = hand_params
        data['fisheye624_params'] = camera_params['fisheye624_params']
        data['extras'] = dict()
        data['extras']['sequence_name'] = sequence_name
        data['extras']['annotation_key'] = annotation_key

        return data 



def main():
    pinhole_params = {'focal_length': torch.tensor([480., 480.], dtype=torch.float32), 'principal_point': torch.tensor([703.5000, 703.5000], dtype=torch.float32)}

    if platform.system() == 'Windows':
        data_root = "F:/Datasets/HOT3D/dataset/ariaSampleDS"
    else:
        data_root = "/netscratch/millerdurai/Datasets/HOT3D/dataset/ariaSampleDS"
    
    dataset = HOT3DLoader(data_root, 'train')

    for i in trange(len(dataset)):
        data = dataset[i]
        
        # print(data[0].shape)


    factory = BasePipelineCreator(data_root)

    for subset in factory.get_subsets():
        groups = factory.get_component_groups(subset)
        stats = factory.get_component_groups_stats(subset)
        shard_len = factory.get_average_shard_sample_count(subset)
        print(f"Subset: {subset}")
        print(f" avg shard len: {shard_len}")
        for group in groups:
            print(f" component group {group}: {stats[group]['all_components']}")


    subset = "train"
    tar_index_map = dict()
    tar_file_paths = factory.get_tar_files_for_subsets(subset, 
                                                       component_groups=["rgb", "annotations", "segmentation", "slam"],)

    with ThreadPoolExecutor() as executor:
        for component_name, tar_paths in tar_file_paths.items():
            tar_index_map[component_name] = dict()

            for tar_index in executor.map(build_tar_index, tar_paths):
                tar_index_map[component_name].update(tar_index)
    
    sample_loader = SamplerLoader()
    

    for tarinfo_name, tar_info in tar_index_map['rgb'].items():
        annotation_key = tarinfo_name.replace('rgb', 'annotation').replace('.jpg', '.json')
        arm_mask_key = tarinfo_name.replace('rgb', 'arm_mask').replace('.jpg', '.png')
        hand_mask_key = tarinfo_name.replace('rgb', 'hand_mask').replace('.jpg', '.png')
        slam_left_key = tarinfo_name.replace('rgb', 'left_mono')
        slam_right_key = tarinfo_name.replace('rgb', 'right_mono')

        rgb_info = tar_info
        annotation_info = tar_index_map['annotations'][annotation_key]
        arm_mask_info = tar_index_map['segmentation'][arm_mask_key]
        hand_mask_info = tar_index_map['segmentation'][hand_mask_key]
        slam_left_info = tar_index_map['slam'][slam_left_key]
        slam_right_info = tar_index_map['slam'][slam_right_key]

        print(tarinfo_name)

        rgb = sample_loader.load(rgb_info, type='jpg')
        slam_left = sample_loader.load(slam_left_info, type='jpg')
        slam_right = sample_loader.load(slam_right_info, type='jpg')
        
        annotation = sample_loader.load(annotation_info, type='json')
        arm_mask = sample_loader.load(arm_mask_info, type='png')
        arm_mask = cv2.cvtColor(arm_mask, cv2.COLOR_BGR2RGB)
        
        
        print(annotation['hand_params'].keys())
        

        for j2D in get_j2d_from_params(annotation['hand_params'], annotation['rgb_camera_params'], rgb):
            cv2.circle(rgb, (int(j2D[0]), int(j2D[1])), 2, (255, 0, 0), -1)

        for j2D in get_j2d_from_params(annotation['hand_params'], annotation['slam_left_camera_params'], slam_left):
            cv2.circle(slam_left, (int(j2D[0]), int(j2D[1])), 2, (255, 0, 0), -1)

        for j2D in get_j2d_from_params(annotation['hand_params'], annotation['slam_right_camera_params'], slam_right):
            cv2.circle(slam_right, (int(j2D[0]), int(j2D[1])), 2, (255, 0, 0), -1)

        print(rgb.shape)
        print(arm_mask.shape)
        print(np.unique(arm_mask[:, :, 0]))

        cv2.imshow('rgb', cv2.resize(rgb[:, :, ::-1], (640, 640)))
        cv2.imshow('slam_left', slam_left)
        cv2.imshow('slam_right', slam_right)

        cv2.imshow('arm_mask', cv2.resize(arm_mask, (640, 640)))

        cv2.waitKey(0)

        # break


if __name__ == "__main__":  
    main()

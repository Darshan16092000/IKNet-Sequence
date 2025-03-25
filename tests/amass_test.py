import sys
import os
import torch
import smplx
import numpy as np
import pyrender
import platform
import trimesh
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from glob import glob
from default_config import cfg
from geometry_utils import axis_to_rot6D, rot6D_to_axis

if platform.system() == 'Windows':
    SMPL_H_PATH = 'F:/Datasets/AMASS/SMPL_H'
else:
    SMPL_H_PATH = '/netscratch/millerdurai/Datasets/AMASS/SMPL_H'


SEQ = 'ACCAD'

def get_smpl_h_files(seq_name):
    return glob(f'{SMPL_H_PATH}/{seq_name}/*/*.npz', recursive=True)


def main():
    smpl_h_files = get_smpl_h_files(SEQ)
    
    print(cfg.mano_path)
    
    mano_left = smplx.MANO(cfg.mano_path,
                           is_rhand=False,
                           ext='pkl',
                           use_pca=False
                           )

    mano_right = smplx.MANO(cfg.mano_path,
                            is_rhand=True,
                            ext='pkl',
                            use_pca=False
                            )

    smplh_body = smplx.SMPLH(
                        os.path.dirname(cfg.mano_path) + '/smplh',
                         gender='male',
                         ext='pkl',
                         use_pca=False,
                         num_betas=16,
    )

    scene = pyrender.Scene()


    for smpl_h_file in smpl_h_files:
        data = np.load(smpl_h_file)
        
        betas = data['betas'][None, :][:, :10]
        betas = torch.tensor(betas).float()
        

        n_frames = data['poses'].shape[0]
        
        for i in np.random.randint(0, n_frames, 5):    
            print(f'Processing {i}/{n_frames}')

            print(data.files)

            trans = data['trans'][i][None, :]
            trans = torch.tensor(trans).float()

            poses = data['poses'][i][None, :]
            poses = torch.tensor(poses).float()

            root_orient = poses[:, :3]
            pose_body = poses[:, 3:66]
            pose_hand = poses[:, 66:]

            # print(pose_hand.shape)
            # pose_hand_rot6d = axis_to_rot6D(pose_hand.view(-1, 3))
            # rot6D_to_axis(pose_hand_rot6d.view(-1, 6))
            # print(pose_hand_rot6d.shape)

            with torch.no_grad():
                smplh_body_output = smplh_body(betas=betas, body_pose=pose_body, return_verts=True,
                                               global_orient=root_orient,
                                               transl=trans,
                                               left_hand_pose=pose_hand[:, :45], 
                                               right_hand_pose=pose_hand[:, 45:])
                
                lt = torch.tensor([[1.0, 0.0, 0.0]])
                rt = torch.tensor([[-1.0, 0.0, 0.0]])
                mano_left_output = mano_left(hand_pose=pose_hand[:, :45], transl=lt, return_verts=True)
                mano_right_output = mano_right(hand_pose=pose_hand[:, 45:], transl=rt, return_verts=True)

            smplh_mesh = pyrender.Mesh.from_trimesh(trimesh.Trimesh(vertices=smplh_body_output.vertices[0], faces=smplh_body.faces ))
            smplh_node = scene.add(smplh_mesh)
            
            mano_left_mesh = pyrender.Mesh.from_trimesh(trimesh.Trimesh(vertices=mano_left_output.vertices[0], faces=mano_left.faces ))
            mano_left_node = scene.add(mano_left_mesh)

            mano_right_mesh = pyrender.Mesh.from_trimesh(trimesh.Trimesh(vertices=mano_right_output.vertices[0], faces=mano_right.faces ))
            mano_right_node = scene.add(mano_right_mesh)
            
            pyrender.Viewer(scene, use_raymond_lighting=True)       

            scene.remove_node(smplh_node)
            scene.remove_node(mano_left_node)
            scene.remove_node(mano_right_node)

            # print(smplh_body_output.vertices.shape)
            # print(smplh_body_output.joints.shape)


            # print(betas.shape)    
            # print(poses.shape)
            # print(root_orient.shape)
            # print(pose_body.shape)  
            # print(pose_hand.shape)
            # print(faces.shape)

            # break
        # smplh_body_output = smplh_body(betas=torch.tensor(betas).float(), body_pose=torch.tensor(poses).float(), return_verts=True)

        # print(smplh_body_output.vertices.shape)
        # print(smplh_body_output.joints.shape)


        # print(betas.shape)    
        # print(poses.shape)
        
        # break

if __name__ == '__main__':
    main()

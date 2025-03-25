import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys  
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

ROOT_DIR = os.path.dirname(ROOT_DIR)
sys.path.append(ROOT_DIR)


from ik_net.datasets import IKDataset, HOT3DLoader as TargetDataset
from ik_net.settings import config as cfg
from default_config import cfg as config
from data_loaders.mano_layer import MANOHandModel


mano_hand_model_pca = MANOHandModel(config.mano_path, joint_mapper=config.mano_mapping, use_pose_pca=True)
mano_hand_model_aa = MANOHandModel(config.mano_path, joint_mapper=config.mano_mapping, use_pose_pca=False)

def get_mesh_from_mano(handedness, betas, global_orient, transl, hand_pose, type='pca'):
    if type == 'pca':
        mano_hand_model = mano_hand_model_pca
    else:
        mano_hand_model = mano_hand_model_aa

    device = mano_hand_model.device

    global_orient = global_orient.float().to(device).unsqueeze(0)
    betas = torch.tensor(betas).float().to(device)
    hand_pose = torch.tensor(hand_pose).float().to(device).unsqueeze(0)
    transl = torch.tensor(transl).float().to(device).unsqueeze(0)

    global_xfrom = torch.cat([global_orient, transl], dim=1)

    if handedness == 'left':
        is_right_hand = torch.tensor([False]).to(device)
        faces = mano_hand_model.mano_layer_left.faces
    else:
        is_right_hand = torch.tensor([True]).to(device)
        faces = mano_hand_model.mano_layer_right.faces

    vertices, joints = mano_hand_model.forward_kinematics(shape_params=betas, joint_angles=hand_pose, global_xfrom=global_xfrom, is_right_hand=is_right_hand)
    vertices = vertices[0].cpu().numpy()
    joints = joints[0].cpu().numpy()

    return vertices, faces, joints

def plot_hand_mesh(ax, vertices, faces, color, label):
    """Plot a hand mesh on the given Axes3D object."""
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    ax.plot_trisurf(x, y, faces, z, color=color, alpha=0.6, edgecolor='none', label=label)


def main():

    train_dataset = IKDataset(cfg, TargetDataset(cfg.DATASET.ROOT, get_camera=True, split='train'))

    indices = list(range(len(train_dataset)))
    random.shuffle(indices)

    for i in indices:
        data, meta = train_dataset[i]

        global_orient = data['global_orient']
        hand_pose_aa = data['hand_pose_aa']
            #axisangle
        hand_pose_pca = meta['hand_pose_pca']
        hand_type = meta['hand_type']
        betas = meta['betas']
        j3d = meta['j3d']
        transl = meta['transl']

        pca_vertices, faces, pca_j3d = get_mesh_from_mano(hand_type, betas, global_orient, transl, hand_pose_pca)

        aa_vertices, faces, aa_j3d = get_mesh_from_mano(hand_type, betas, global_orient, transl, hand_pose_aa, type='aa')   

        transl_zero = torch.zeros(3)
        w_t_vertices, faces, w_t_j3d = get_mesh_from_mano(hand_type, betas, global_orient, transl_zero, hand_pose_aa, type='aa')   
        
        print('pca_j3d[0]:  ', pca_j3d[0])
        print('aa_j3d[0]:  ', aa_j3d[0])
        print('j3d:  ', j3d[0])

        w_t_vertices -= w_t_j3d[0]
        w_t_j3d -= w_t_j3d[0]

        w_t_vertices += j3d[0]
        w_t_j3d += j3d[0]

        print('w_t_j3d[0]:  ', w_t_j3d[0])


        # Plot with Matplotlib
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the PCA and AA meshes
        plot_hand_mesh(ax, pca_vertices, faces, color='red', label='PCA Mesh')
        plot_hand_mesh(ax, aa_vertices, faces, color='green', label='AA Mesh')
        plot_hand_mesh(ax, w_t_vertices, faces, color='blue', label='Wrist Transl Mesh')

        # Plot the joints (j3d points)
        ax.scatter(j3d[:, 0], j3d[:, 1], j3d[:, 2], color='blue', s=20, label='Joints (j3d)')

        # Add joint index labels
        for idx, joint in enumerate(j3d):
            ax.text(joint[0], joint[1], joint[2], str(idx), color='black', fontsize=8)

        all_vertices = np.concatenate([pca_vertices, aa_vertices, w_t_vertices, j3d], axis=0)
        # Set equal scaling using all vertices from all meshes
        max_range = np.array([all_vertices[:, 0].max()-all_vertices[:, 0].min(), 
                            all_vertices[:, 1].max()-all_vertices[:, 1].min(), 
                            all_vertices[:, 2].max()-all_vertices[:, 2].min()]).max() / 2.0

        mid_x = (all_vertices[:, 0].max() + all_vertices[:, 0].min()) * 0.5
        mid_y = (all_vertices[:, 1].max() + all_vertices[:, 1].min()) * 0.5
        mid_z = (all_vertices[:, 2].max() + all_vertices[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)


        # Set plot details
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        plt.title("PCA and AA Hand Meshes with Joint Indices")
        plt.show()


if __name__ == '__main__':
    main()

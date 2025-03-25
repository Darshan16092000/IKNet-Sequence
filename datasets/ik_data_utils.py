import os
import os.path as osp
import pickle
import numpy as np
from smplx.utils import Struct


class PCATOAA:
    def load_mano_hand_components(self):
        model_path = self.cfg.MANO_PATH + '/mano'
        ext = 'pkl'
        
        hand_components = dict()
        for hand_type, is_rhand in [('right', True), ('left', False)]:
            if osp.isdir(model_path):
                model_fn = 'MANO_{}.{ext}'.format(
                    'RIGHT' if is_rhand else 'LEFT', ext=ext)
                mano_path = os.path.join(model_path, model_fn)
        
            if ext == 'pkl':
                with open(mano_path, 'rb') as mano_file:
                    model_data = pickle.load(mano_file, encoding='latin1')
            elif ext == 'npz':
                model_data = np.load(mano_path, allow_pickle=True)
            else:
                raise ValueError('Unknown extension: {}'.format(ext))

            data_struct = Struct(**model_data)
            hand_components[hand_type] = np.array(data_struct.hands_components, dtype=np.float32)
            
        return hand_components

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.hand_components = self.load_mano_hand_components()

    def __call__(self, hand_type, pca_pose):
        is_np = isinstance(pca_pose, np.ndarray)

        if not is_np:
            pca_pose = np.array(pca_pose)

        num_pca_components = pca_pose.shape[0]
        basis_subset = self.hand_components[hand_type][:, :num_pca_components]

        full_hand_pose = pca_pose @ basis_subset.T

        if is_np:
            return full_hand_pose
        
        return full_hand_pose.tolist()


class AAtoPCA:        
    def load_mano_inverse_hand_components(self):
        model_path = self.cfg.MANO_PATH + '/mano'
        ext = 'pkl'
        
        hand_components = dict()
        for hand_type, is_rhand in [('right', True), ('left', False)]:
            if osp.isdir(model_path):
                model_fn = 'MANO_{}.{ext}'.format(
                    'RIGHT' if is_rhand else 'LEFT', ext=ext)
                mano_path = os.path.join(model_path, model_fn)
        
            if ext == 'pkl':
                with open(mano_path, 'rb') as mano_file:
                    model_data = pickle.load(mano_file, encoding='latin1')
            elif ext == 'npz':
                model_data = np.load(mano_path, allow_pickle=True)
            else:
                raise ValueError('Unknown extension: {}'.format(ext))

            data_struct = Struct(**model_data)
            hand_components[hand_type] = np.linalg.inv(np.array(data_struct.hands_components, dtype=np.float32))
            
        return hand_components

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.inverse_hand_components = self.load_mano_inverse_hand_components()

    def __call__(self, hand_type, hand_pose):
        is_np = isinstance(hand_pose, np.ndarray)

        if not is_np:
            hand_pose = np.array(hand_pose)        
    
        pca_hand_pose = hand_pose @ self.inverse_hand_components[hand_type]

        if is_np:
            return pca_hand_pose
        
        return pca_hand_pose.tolist()

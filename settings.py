import os
import yaml
import platform
import json

import numpy as np
from easydict import EasyDict as edict


config = edict()

# debug
config.DEBUG = edict()
config.DEBUG.NO_MP = False
config.DEBUG.DEBUG = False
config.DEBUG.TRAIN_SET_ONLY = False
config.DEBUG.SAVE_BATCH_IMAGES_GT = True
config.DEBUG.SAVE_BATCH_IMAGES_PRED = True
config.DEBUG.SAVE_HEATMAPS_GT = True
config.DEBUG.SAVE_HEATMAPS_PRED = True
config.DEBUG.SAVE_BATCH_J3D_GT = False

config.BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))
config.N_GPUS = len(os.getenv('CUDA_VISIBLE_DEVICES', '0').split(','))
config.PRINT_FREQ = 50
config.TEST_ITERATIONS_PER_EPOCH = 2000

config.MODEL = edict()
config.MODEL.FOV_H = 110
config.MODEL.ASPECT_RATIO = 1.0

config.MODEL.INPUT_CHANNEL = 3
config.MODEL.IMAGE_SIZE = [640, 640] 
config.MODEL.HEAT_MAP_SCALE = 4
config.MODEL.HEATMAP_SIZE = [config.MODEL.IMAGE_SIZE[0] // config.MODEL.HEAT_MAP_SCALE, config.MODEL.IMAGE_SIZE[1] // config.MODEL.HEAT_MAP_SCALE]  # width * height, ex: 24 * 32
config.MODEL.TARGET_TYPE = 'gaussian'
config.MODEL.SIGMA = 2

config.MODEL.UNCERTAINTY_DEPTH_PREDICTION = True
config.MODEL.vMF = True
config.MODEL.USE_SPHERICAL_COORDS = False
config.MODEL.vMF_LATENT = False

config.MODEL.CHECKPOINT_PATH = os.getenv('CHECKPOINT_PATH', '')        
# with open(os.path.join(os.path.dirname(__file__), 'intrinsics.json'), 'r') as f:
#     config.ocam_model = json.load(f)
# config.MODEL.CHECKPOINT_PATH = 'HOT3DLoader_2025-03-18-10-09/model_best.pth'        
# # with open(os.path.join(os.path.dirname(__file__), 'intrinsics.json'), 'r') as f:
# #     config.ocam_model = json.load(f)


# DATASET related params
config.DATASET = edict()

config.DATASET.TEMPORAL_STEPS = 20 
config.TRAIN_ITERATIONS_PER_EPOCH = 2500 

config.DATASET.TYPE = 'Synthetic' # 'Real' or 'Synthetic' or 'Wild'
config.DATASET.BG_AUG = False

if platform.system() == 'Windows':
    # config.DATASET.ROOT = "F:/Datasets/HOT3D/dataset/ariaSampleDS"
    config.DATASET.ROOT = "F:/Datasets/HOT3D/dataset/ariaDS"

    config.MANO_PATH = "F:/models/mano_v1_2/models"
    # OUTPUT_DIR = 'F:/Experiments/Aria/ik_net'
    OUTPUT_DIR = 'F:/Experiments/Aria/ik_net_develop'

else:
    # config.DATASET.ROOT = "/netscratch/millerdurai/Datasets/HOT3D/dataset/ariaSampleDS"
    config.DATASET.ROOT = "/netscratch/millerdurai/Datasets/HOT3D/dataset/ariaDS"

    config.MANO_PATH = "/netscratch/millerdurai/models/mano_v1_2/models"
    # OUTPUT_DIR = '/netscratch/millerdurai/Experiments/Aria/ik_net'
    OUTPUT_DIR = '/netscratch/asodariya/Experiments/Aria/ik_net_develop'


config.DATASET.TEST_ON = 'Synthetic' # Synthetic or 'Real' or 'Wild'
config.DATASET.FINETUNE_ON = 'Synthetic'

config.DATASET.REAL_ROOT = 'Specifiy the path to the real dataset (EE3D-R FOLDER) in Line 54 of settings.py'
config.DATASET.SYN_ROOT = 'Specifiy the path to the synthetic dataset(EE3D-S FOLDER) in Line 55 of settings.py'
config.DATASET.SYN_TEST_ROOT = 'Specifiy the path to the synthetic test dataset(EE3D-S-Test FOLDER) in Line 56 of settings.py'
config.DATASET.WILD_ROOT = 'Specifiy the path to the in-the-wild dataset (EE3D-W FOLDER) in Line 57 of settings.py'
config.DATASET.BACKGROUND_DATASET_ROOT = 'Specifiy the path to the background dataset (Background_Dataset FOLDER) in Line 58 of settings.py'

config.DATASET.REAL = edict()
config.DATASET.SYNTHETIC = edict()

config.DATASET.SCALE_FACTOR = 0.2
config.DATASET.FLIP = True
config.DATASET.ROT_FACTOR = 3

config.DATASET.SEQUENCE_LENGTH = 30


config.DATASET.ENSEMBLE_DATASETS = [
 [config.DATASET.REAL_ROOT, 'Real', 0.6], 
 [config.DATASET.REAL_ROOT, 'Wild', 0.6], 
 [config.DATASET.SYN_ROOT, 'Synthetic', 1.0],
]


config.DATASET.REPRESENTATION = 'LNES'

config.DATASET.EVENT_BATCH_SIZE = 8192 
config.DATASET.REAL.MAX_FRAME_TIME_IN_MS = 33
config.DATASET.SYNTHETIC.MAX_FRAME_TIME_IN_MS = 20
config.DATASET.SYNTHETIC.RETURN_RGB = False

config.DATASET.LNES = edict()
config.DATASET.LNES.WINDOWS_TIME_MS = max(config.DATASET.REAL.MAX_FRAME_TIME_IN_MS, config.DATASET.SYNTHETIC.MAX_FRAME_TIME_IN_MS)

config.DATASET.EROS = edict()
config.DATASET.EROS.KERNEL_SIZE = 3
config.DATASET.EROS.DECAY_BASE = 0.7



config.NUM_JOINTS = 21



config.OUTPUT_DIR = f'{OUTPUT_DIR}/logs/output'
config.LOG_DIR = f'{OUTPUT_DIR}/logs/tensorboard'

config.GPUS = '0'
config.WORKERS = 4

# Cudnn related params
config.CUDNN = edict()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True

# common params for NETWORK
config.MODEL.NAME = 'EgoHPE'
config.MODEL.INIT_WEIGHTS = True
config.MODEL.PRETRAINED = ''

config.MODEL.STYLE = 'pytorch'

config.LOSS = edict()
config.LOSS.USE_TARGET_WEIGHT = True


# train
config.TRAIN = edict()

config.TRAIN.LR_FACTOR = 0.1
config.TRAIN.LR_STEP = [50, 90]
config.TRAIN.LR = 0.00003

config.TRAIN.OPTIMIZER = 'adam'
config.TRAIN.MOMENTUM = 0.9
config.TRAIN.WD = 0.0001
config.TRAIN.NESTEROV = False
config.TRAIN.GAMMA1 = 0.99
config.TRAIN.GAMMA2 = 0.0

config.TRAIN.BEGIN_EPOCH = 0
config.TRAIN.END_EPOCH = 1400

config.TRAIN.RESUME = False
config.TRAIN.CHECKPOINT = ''

config.TRAIN.BATCH_SIZE = 32
config.TRAIN.SHUFFLE = True

# testing
config.TEST = edict()

# size of images for each device
config.TEST.BATCH_SIZE = 32
# Test Model Epoch
config.TEST.FLIP_TEST = False
config.TEST.POST_PROCESS = True
config.TEST.SHIFT_HEATMAP = True

config.TEST.USE_GT_BBOX = False
# nms
config.TEST.OKS_THRE = 0.5
config.TEST.IN_VIS_THRE = 0.0
config.TEST.COCO_BBOX_FILE = ''
config.TEST.BBOX_THRE = 1.0
config.TEST.MODEL_FILE = ''
config.TEST.IMAGE_THRE = 0.0
config.TEST.NMS_THRE = 1.0



def _update_dict(k, v):
    if k == 'DATASET':
        if 'MEAN' in v and v['MEAN']:
            v['MEAN'] = np.array([eval(x) if isinstance(x, str) else x
                                  for x in v['MEAN']])
        if 'STD' in v and v['STD']:
            v['STD'] = np.array([eval(x) if isinstance(x, str) else x
                                 for x in v['STD']])
    if k == 'MODEL':
        if 'EXTRA' in v and 'HEATMAP_SIZE' in v['EXTRA']:
            if isinstance(v['EXTRA']['HEATMAP_SIZE'], int):
                v['EXTRA']['HEATMAP_SIZE'] = np.array(
                    [v['EXTRA']['HEATMAP_SIZE'], v['EXTRA']['HEATMAP_SIZE']])
            else:
                v['EXTRA']['HEATMAP_SIZE'] = np.array(
                    v['EXTRA']['HEATMAP_SIZE'])
        if 'IMAGE_SIZE' in v:
            if isinstance(v['IMAGE_SIZE'], int):
                v['IMAGE_SIZE'] = np.array([v['IMAGE_SIZE'], v['IMAGE_SIZE']])
            else:
                v['IMAGE_SIZE'] = np.array(v['IMAGE_SIZE'])
    for vk, vv in v.items():
        if vk in config[k]:
            config[k][vk] = vv
        else:
            raise ValueError("{}.{} not exist in config.py".format(k, vk))


def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(k, v)
                else:
                    if k == 'SCALES':
                        config[k][0] = (tuple(v))
                    else:
                        config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))


def gen_config(config_file):
    cfg = dict(config)
    for k, v in cfg.items():
        if isinstance(v, edict):
            cfg[k] = dict(v)

    with open(config_file, 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)


def update_dir(model_dir, log_dir, data_dir):
    if model_dir:
        config.OUTPUT_DIR = model_dir

    if log_dir:
        config.LOG_DIR = log_dir

    if data_dir:
        config.DATA_DIR = data_dir

    config.DATASET.ROOT = os.path.join(
            config.DATA_DIR, config.DATASET.ROOT)

    config.TEST.COCO_BBOX_FILE = os.path.join(
            config.DATA_DIR, config.TEST.COCO_BBOX_FILE)

    config.MODEL.PRETRAINED = os.path.join(
            config.DATA_DIR, config.MODEL.PRETRAINED)


def get_model_name(cfg):
    name = cfg.MODEL.NAME
    full_name = cfg.MODEL.NAME
    extra = cfg.MODEL.EXTRA
    if name in ['pose_resnet']:
        name = '{model}_{num_layers}'.format(
            model=name,
            num_layers=extra.NUM_LAYERS)
        deconv_suffix = ''.join(
            'd{}'.format(num_filters)
            for num_filters in extra.NUM_DECONV_FILTERS)
        full_name = '{height}x{width}_{name}_{deconv_suffix}'.format(
            height=cfg.MODEL.IMAGE_SIZE[1],
            width=cfg.MODEL.IMAGE_SIZE[0],
            name=name,
            deconv_suffix=deconv_suffix)
    else:
        raise ValueError('Unkown model: {}'.format(cfg.MODEL))

    return name, full_name


ESIM_REFRACTORY_PERIOD_NS = 0
ESIM_POSITIVE_THRESHOLD = 0.4
ESIM_NEGATIVE_THRESHOLD = 0.4


if __name__ == '__main__':
    import sys
    gen_config(sys.argv[1])
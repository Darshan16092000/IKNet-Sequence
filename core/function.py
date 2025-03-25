import time
import logging
import os
import datetime

import numpy as np
import torch
import cv2

from pathlib import Path
from tqdm import trange

from core.evaluate import accuracy
from dis_est_utils.vis import save_debug_3d_mesh, draw_3d_mesh
from dis_est_utils.skeleton import HandSkeleton
# from dis_est_utils.skeleton import HandSkeleton as Skeleton
import open3d as o3d


from default_config import cfg as config
from data_loaders.mano_layer import MANOHandModel
from utils.rotations import rotation_6d_to_axis_angle, axis_angle_to_rotation_6d
import matplotlib.pyplot as plt
import seaborn as sns

HEATMAP_DIR = None

logger = logging.getLogger(__name__)

final_output_dir = "/netscratch/asodariya/Experiments/Aria/ik_net_develop/logs/output/HOT3DLoader/EgoHPE"


mano_hand_model = MANOHandModel(config.mano_path, joint_mapper=config.mano_mapping, use_pose_pca=False, device='cuda')
MANO_FACES = {
    'left': mano_hand_model.mano_layer_left.faces,
    'right': mano_hand_model.mano_layer_right.faces
}




import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch

final_output_dir = "logs/attention_maps"
os.makedirs(final_output_dir, exist_ok=True)



# for joint features attention map
'''def save_attention_heatmap(attn_weights, epoch, batch, sample=0, num_joint_tokens=5):
    """
    Plots a heatmap of the attention matrix for just the joint features (last 5x5 values).
    
    Args:
        attn_weights (torch.Tensor or np.ndarray): The attention matrix with shape 
                                                  (B, 110, 110) or (110, 110) if no batch dim.
        epoch (int): Current epoch number.
        batch (int): Current batch number.
        sample (int): Which sample in the batch to visualize (default: 0).
        num_joint_tokens (int): Number of joint feature tokens (default: 5).
    """
    # Convert to numpy if attn_weights is a torch.Tensor.
    if isinstance(attn_weights, torch.Tensor):
        attn_weights = attn_weights.detach().cpu().numpy()
    
    # If there's a batch dimension, select the specific sample.
    if attn_weights.ndim == 3:
        attn = attn_weights[sample]  # shape: (110, 110)
    elif attn_weights.ndim == 2:
        attn = attn_weights
    else:
        raise ValueError(f"Unexpected attention weight shape: {attn_weights.shape}")
    
    # Slice off only the last 5x5 values (joint features attention)
    attn = attn[-num_joint_tokens:, -num_joint_tokens:]
    
    # Create the heatmap.
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(attn, cmap="viridis", cbar=True, annot=True, fmt='.3f')
    
    # Set tick labels as indices
    ax.set_xticks(np.arange(0.5, num_joint_tokens, 1))
    ax.set_yticks(np.arange(0.5, num_joint_tokens, 1))
    ax.set_xticklabels(range(num_joint_tokens), rotation=0, fontsize=10)
    ax.set_yticklabels(range(num_joint_tokens), fontsize=10)
    
    # Add titles and axis labels.
    plt.title(f"Joint Features Attention Heatmap (Epoch {epoch}, Batch {batch}, Sample {sample})")
    plt.xlabel("Joint Feature Tokens (105-109)")
    plt.ylabel("Joint Feature Tokens (105-109)")
    
    # Save the figure.
    save_path = os.path.join(final_output_dir, f"joint_attn_epoch_{epoch}_batch_{batch}_sample{sample}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    
    print(f"Joint features attention heatmap saved at {save_path}")'''

def visualize_skeleton_from_joints(joint_positions, save_path):
    """
    Given joint positions for a single frame, visualize and save the skeleton.
    """
    skeleton = HandSkeleton()
    skeleton_mesh = skeleton.joints_2_mesh(joint_positions)

    # Save the skeleton visualization
    o3d.io.write_triangle_mesh(save_path, skeleton_mesh)
    # print(f"✅ Skeleton saved at: {save_path}")

# apply this one 
def save_attention_heatmap(attn_weights, epoch, batch, sample=0, num_global_tokens=630, annotation_keys=None, gt_pr_j3d=None):
    """
    Plots a full heatmap of the attention matrix of shape (105, 105), where all tokens
    are considered global features.
    
    Args:
        attn_weights (torch.Tensor or np.ndarray): The attention matrix with shape 
                                                  (B, 110, 110) or (110, 110) if no batch dim.
        epoch (int): Current epoch number.
        batch (int): Current batch number.
        sample (int): Which sample in the batch to visualize (default: 0).
        num_global_tokens (int): Number of tokens for global features (default: 105).
    """

    global HEATMAP_DIR
    # Initialize HEATMAP_DIR once per run
    if HEATMAP_DIR is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        HEATMAP_DIR = os.path.join(final_output_dir, f"attn_heatmaps_{timestamp}")
        os.makedirs(HEATMAP_DIR, exist_ok=True)


    # Convert to numpy if attn_weights is a torch.Tensor.
    # print(f"Shape of attn_weights before indexing: {attn_weights.shape} and dtype: {attn_weights.dtype}")
    attn_weights = attn_weights[:,:num_global_tokens, :num_global_tokens]
    # attn_weights = attn_weights.view(32, 30, 21, 30, 21)
    # attn_weights = attn_weights.permute(0, 2, 1, 4, 3)
    attn_weights = attn_weights.contiguous().view(32, num_global_tokens, num_global_tokens)
    # print(f"Shape of attn_weights after indexing: {attn_weights.shape} and dtype: {attn_weights.dtype}")


    if isinstance(attn_weights, torch.Tensor):
        attn_weights = attn_weights.detach().cpu().numpy()
    
    # If there's a batch dimension, select the specific sample.
    if attn_weights.ndim == 3:
        attn = attn_weights[sample]  # shape: (110, 110)
    elif attn_weights.ndim == 2:
        attn = attn_weights
    else:
        raise ValueError(f"Unexpected attention weight shape: {attn_weights.shape}")
    
     # Extract the annotation keys for the current sample

    if annotation_keys:
        first_annotation = annotation_keys[sample][0]
        last_annotation = annotation_keys[sample][-1]
        title_text = f"Full Attention Heatmap (Epoch {epoch}, Batch {batch})\nFirst: {first_annotation} | Last: {last_annotation}"
    else:
        title_text = f"Full Attention Heatmap (Epoch {epoch}, Batch {batch})"

    # Create the heatmap.
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(attn, cmap="viridis", cbar=True)
    
    # Set tick labels as indices
    ax.set_xticks(np.arange(0.5, 630, 21))
    ax.set_yticks(np.arange(0.5, 630, 21))
    ax.set_xticklabels(range(0, 630, 21), rotation=90, fontsize=6)
    ax.set_yticklabels(range(0, 630, 21), fontsize=6)
    
    # Add titles and axis labels.
    plt.title(title_text)
    # plt.title(f"Full Attention Heatmap (Epoch {epoch}, Batch {batch}, Sample {sample})")
    plt.xlabel("Joint Tokens (0-630)")
    plt.ylabel("Joint Tokens (0-630)")
    
    # Save the figure.
    save_path = os.path.join(HEATMAP_DIR, f"epoch_{epoch}_batch_{batch}_sample{sample}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    
    print(f"Full attention heatmap saved at {save_path}")

    # **Save Skeletons for Each Frame**
    skeleton_dir = os.path.join(HEATMAP_DIR, f"epoch_{epoch}_batch_{batch}_sample_{sample}_skeletons")
    os.makedirs(skeleton_dir, exist_ok=True)

    if gt_pr_j3d is not None:
        for frame_idx in range(gt_pr_j3d.shape[1]):  # Iterate over 30 frames
            frame_joints = gt_pr_j3d[sample][frame_idx].cpu().numpy()
            skeleton_path = os.path.join(skeleton_dir, f"frame_{frame_idx}.ply")
            visualize_skeleton_from_joints(frame_joints, skeleton_path)

    # print(f"✅ Skeletons for sample {sample} saved in {skeleton_dir}")


def get_mesh_from_mano(handedness, global_orient, hand_pose):
    device = mano_hand_model.device
    B = global_orient.size(0)
    betas = torch.zeros(10).float().to(device)
    transl = torch.zeros(B, 3).float().to(device)

    global_xfrom = torch.cat([global_orient, transl], dim=1)
    is_right_hand = [False if h == 'left' else True for h in handedness]
    is_right_hand = torch.tensor(is_right_hand, dtype=torch.bool, device=device)



    # Ensure it has the correct batch size
    if is_right_hand.shape[0] != global_orient.shape[0]:  
        is_right_hand = is_right_hand.expand(global_orient.shape[0])

    vertices, joints = mano_hand_model.forward_kinematics(shape_params=betas,
                                                          joint_angles=hand_pose,
                                                          global_xfrom=global_xfrom,
                                                          is_right_hand=is_right_hand)
    return vertices


def transpose_meta(meta_batch):
    """Transpose metadata from [seq_len, batch] to [batch, seq_len]"""
    transposed_meta = {}
    for key in meta_batch.keys():
        transposed_meta[key] = [list(item) for item in zip(*meta_batch[key])]
    return transposed_meta

def train(config, train_loader, train_dataset, model, criterions, optimizer, epoch, output_dir, tb_log_dir, writer_dict, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()
    orient_losses = AverageMeter()
    pose_losses = AverageMeter()
    acc = AverageMeter()

    print("in train function")
    
    # switch to train mode
    model.train()
    
    end = time.time()
    for i, batch in enumerate(train_loader):
        if i > config.TRAIN_ITERATIONS_PER_EPOCH:
            break

        data_time.update(time.time() - end)

        data, meta = batch

        meta = transpose_meta(meta)
        # print(f"Meta Type: {type(meta)}")  # Should be dict
        # print(f"Hand Type Shape: {len(meta['hand_type'])}x{len(meta['hand_type'][0])}")  # Should be 32x30
        
        # Get hand type of last frame for all batch items (Shape: [32])
        hand_type = [hand_sequence[-1] for hand_sequence in meta['hand_type']]
        annotation_keys = [anno_key for anno_key in meta['annotation_key']]

        # Extract multi-frame data
        gt_pr_j3d = data['pr_j3d'].to(device)  # Shape: (BZ, T, ...)
        BZ, T = gt_pr_j3d.size(0), gt_pr_j3d.size(1)

        gt_global_orient_aa = data['global_orient'].to(device)  # Shape: (BZ, T, 3)
        gt_global_orient = axis_angle_to_rotation_6d(gt_global_orient_aa.view(-1, 3)).view(BZ, T, -1)

        gt_hand_pose_aa = data['hand_pose_aa'].to(device)  # Shape: (BZ, T, num_joints, 3)
        gt_hand_pose = axis_angle_to_rotation_6d(gt_hand_pose_aa.view(-1, 3)).view(BZ, T, -1, 6)

        # Forward pass for all frames
        outputs, attn_weights = model(gt_global_orient_aa, gt_pr_j3d)        

        pred_global_orient = outputs['global_orient'].view(BZ, -1)
        pred_global_orient_aa = rotation_6d_to_axis_angle(pred_global_orient)

        pred_hand_pose = outputs['hand_pose'].view(BZ, -1, 6)
        pred_hand_pose_aa = rotation_6d_to_axis_angle(pred_hand_pose).view(BZ, -1)
        
        gt_hand_pose = gt_hand_pose[:, -1, :, :].view(BZ, -1, 6)
        gt_global_orient = gt_global_orient[:, -1, :].view(BZ, -1)

        # Compute losses for each frame
        loss_pose = criterions['pose_loss'](pred_hand_pose, gt_hand_pose)
        loss_orient = criterions['orient_loss'](pred_global_orient, gt_global_orient)
        loss = loss_pose + loss_orient

        # Backpropagation and optimizer step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if attn_weights is not None and epoch % 25 == 0:
            # print(f"Shape of attn_weights before indexing: {attn_weights.shape}")

            # Save the heatmap every few iterations
            if i % (config.PRINT_FREQ*2) == 0:
                # print("Max attention value:", attn_weights.max().item())
                # print("Min attention value:", attn_weights.min().item())
                # print("Mean attention value:", attn_weights.mean().item())
                
                
                save_attention_heatmap(attn_weights, epoch, i, annotation_keys= annotation_keys, gt_pr_j3d=gt_pr_j3d)
        


        pose_losses.update(loss_pose.item(), BZ)
        orient_losses.update(loss_orient.item(), BZ)
      
        losses.update(loss.item(), BZ)

        gt_global_orient_aa = gt_global_orient_aa[:, -1, :]
        gt_hand_pose_aa = gt_hand_pose_aa[:, -1, :]

        gt_vertices = get_mesh_from_mano(hand_type, gt_global_orient_aa, gt_hand_pose_aa) * 1000 # scale to mm
        pred_vertices = get_mesh_from_mano(hand_type, pred_global_orient_aa, pred_hand_pose_aa) * 1000 # scale to mm

        # Accuracy computation
        avg_acc, cnt = accuracy(
            gt_vertices.view(-1, *gt_vertices.shape[2:]),  # Flatten batch and time dimensions
            pred_vertices.view(-1, *pred_vertices.shape[2:]),  # Same flattening
            torch.ones_like(gt_vertices[..., :1])  # Validity mask
        )
        acc.update(avg_acc, cnt)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # Logging and Debugging
        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Orient_Loss {orient_loss.val:.5f} ({orient_loss.avg:.5f})\t' \
                  'Pose_Loss {pose_loss.val:.5f} ({pose_loss.avg:.5f})\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'MPJPE {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                speed=BZ * T / batch_time.val,
                data_time=data_time,
                loss=losses,
                pose_loss=pose_losses,
                orient_loss=orient_losses,
                acc=acc
            )
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            if i % (config.PRINT_FREQ * 10) == 0:
                save_debug_3d_mesh(config, meta, gt_vertices, pred_vertices, hand_type, MANO_FACES, 'train', writer, global_steps)
                



def validate(config, val_loader, val_dataset, model, criterions, output_dir, tb_log_dir, writer_dict, device):
    batch_time = AverageMeter()

    acc_verts = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, batch in enumerate(val_loader):
            data, meta = batch
            meta = transpose_meta(meta)
            hand_type = [hand_sequence[-1] for hand_sequence in meta['hand_type']]
            # if i > 50: break

            # Extract multi-frame data
            gt_pr_j3d = data['pr_j3d'].to(device)  # Shape: (BZ, T, ...)
            BZ, T = gt_pr_j3d.size(0), gt_pr_j3d.size(1)

            gt_global_orient_aa = data['global_orient'].to(device)  # Shape: (BZ, T, 3)
            gt_global_orient = axis_angle_to_rotation_6d(gt_global_orient_aa.view(-1, 3)).view(BZ, T, -1)

            gt_hand_pose_aa = data['hand_pose_aa'].to(device)  # Shape: (BZ, T, num_joints, 3)
            gt_hand_pose = axis_angle_to_rotation_6d(gt_hand_pose_aa.view(-1, 3)).view(BZ, T, -1, 6)

            outputs, attn_weights = model(gt_global_orient_aa, gt_pr_j3d)
 
            pred_global_orient = outputs['global_orient'].view(BZ, -1)
            pred_global_orient_aa = rotation_6d_to_axis_angle(pred_global_orient)

            pred_hand_pose = outputs['hand_pose'].view(BZ, -1, 6)
            pred_hand_pose_aa = rotation_6d_to_axis_angle(pred_hand_pose).view(BZ, -1)

            gt_hand_pose = gt_hand_pose[:, -1, :, :].view(BZ, -1, 6)
            gt_global_orient = gt_global_orient[:, -1, :].view(BZ, -1)

            gt_global_orient_aa = gt_global_orient_aa[:, -1, :]
            gt_hand_pose_aa = gt_hand_pose_aa[:, -1, :]

            gt_vertices = get_mesh_from_mano(hand_type, gt_global_orient_aa, gt_hand_pose_aa) * 1000 # scale to mm
            pred_vertices = get_mesh_from_mano(hand_type, pred_global_orient_aa, pred_hand_pose_aa) * 1000 # scale to mm

            avg_acc, cnt = accuracy(
                gt_vertices.view(-1, *gt_vertices.shape[2:]),  # Flatten batch and time dimensions
                pred_vertices.view(-1, *pred_vertices.shape[2:]),  # Same flattening
                torch.ones_like(gt_vertices[..., :1])  # Validity mask
            )

            acc_verts.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                    'MPJPE {acc.val:.4f} ({acc.avg:.4f})\t'.format(
                        i, len(val_loader), batch_time=batch_time,
                        acc=acc_verts)
                logger.info(msg)

        perf_indicator = acc_verts.avg

        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar(
            'valid_acc_verts',
            acc_verts.avg,
            global_steps
        )
        writer_dict['valid_global_steps'] = global_steps + 1

        writer = writer_dict['writer']

        save_debug_3d_mesh(config, meta, gt_vertices, pred_vertices, hand_type, MANO_FACES, 'val', writer, global_steps)

    return perf_indicator


def test(config, val_loader, val_dataset, model, tb_log_dir, writer_dict, device, seq_time_in_sec=10):
    fps = 30

    seq_len = seq_time_in_sec * fps
    data_len = len(val_dataset)

    global_steps = writer_dict['valid_global_steps']
    np.random.seed(int(global_steps))
    start = np.random.randint(0, data_len)
    stop = min(start + seq_len, data_len)

    tb_log_dir = Path(tb_log_dir)
    video_path = str(tb_log_dir / f'{global_steps}.mp4')

    vid_height = 1000
    vid_width = 3000
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (vid_width, vid_height))
    model.eval()

    elev = np.random.uniform(10, 170)  # Random elevation angle between 10 and 170 degrees
    azim = np.random.uniform(0, 360)   # Random azimuth angle between 0 and 360 degrees

    for i in trange(start, stop, config.BATCH_SIZE):
        # if i > 50: break

        gt_j3d = []
        gt_hand_pose_aa = []
        gt_global_orient_aa = []
        hand_type = []
        gt_pr_j3d = []
        for j in range(config.BATCH_SIZE):
            if i + j >= data_len: break

            data, meta = val_dataset[i + j]
            # meta = transpose_meta(meta)


            gt_pr_j3d.append(data['pr_j3d'][None, ...])
            gt_global_orient_aa.append(data['global_orient'][None, ...])
            gt_hand_pose_aa.append(data['hand_pose_aa'][None, ...])
            # gt_j3d.append(meta['j3d'][None, ...])
            hand_type.append([meta['hand_type']])

        hand_types = [seq[-1] for seq in hand_type]
        hand_types = [seq[-1] for seq in hand_types]
        hand_type = hand_types

        # gt_j3d = np.concatenate(gt_j3d, axis=0) * 1000 # scale to mm
        gt_pr_j3d = torch.cat(gt_pr_j3d, dim=0).to(device)
        gt_global_orient_aa = torch.cat(gt_global_orient_aa, dim=0).to(device)
        gt_global_orient = axis_angle_to_rotation_6d(gt_global_orient_aa)
        gt_hand_pose_aa = torch.cat(gt_hand_pose_aa, dim=0).to(device)

        BZ = gt_pr_j3d.size(0)

        with torch.no_grad():
            outputs, attn_weights = model(gt_global_orient_aa, gt_pr_j3d)

        gt_global_orient_aa = gt_global_orient_aa[:, -1, :]
        gt_hand_pose_aa = gt_hand_pose_aa[:, -1, :]
        pred_global_orient = outputs['global_orient']
        pred_global_orient_aa = rotation_6d_to_axis_angle(pred_global_orient)

        pred_hand_pose = outputs['hand_pose']
        pred_hand_pose_aa = rotation_6d_to_axis_angle(pred_hand_pose).view(BZ, -1)

        gt_vertices = get_mesh_from_mano(hand_type, gt_global_orient_aa, gt_hand_pose_aa).cpu().numpy() * 1000 # scale to mm
        pred_vertices = get_mesh_from_mano(hand_type, pred_global_orient_aa, pred_hand_pose_aa).cpu().numpy() * 1000 # scale to mm


        'for video'
        for i in range(BZ):
            pred_vertices_ = pred_vertices[i]
            gt_vertices_ = gt_vertices[i]
            hand_type_ = hand_type[i]
            faces = MANO_FACES[hand_type_]
            # print("create mesh")

            gt_image = draw_3d_mesh(gt_vertices_, None, faces, camera={'elev': elev, 'azim': azim})
            pred_image = draw_3d_mesh(None, pred_vertices_, faces, camera={'elev': elev, 'azim': azim})
            img_overlay = draw_3d_mesh(gt_vertices_, pred_vertices_, faces, camera={'elev': elev, 'azim': azim})
            # print("draw mesh")

            hstack = np.concatenate([gt_image, pred_image, img_overlay], axis=1)
            hstack = cv2.cvtColor(hstack, cv2.COLOR_RGB2BGR)
            video.write(hstack)
            # print("write to video")

    video.release()


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

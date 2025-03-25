import os

import math
import torch
import traceback
import numpy as np
import torchvision
import cv2
import pyrender
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from settings import config


def plot_hand_mesh(ax, vertices, faces, color, label):
    """Plot a hand mesh on the given Axes3D object."""
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    ax.plot_trisurf(x, y, faces, z, color=color, alpha=0.6, edgecolor='none', label=label)


def draw_3d_mesh(gt_vertices=None, pred_vertices=None, faces=None, camera=None):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    if gt_vertices is not None:
        plot_hand_mesh(ax, gt_vertices, faces, color='green', label='GT Mesh')
    
    if pred_vertices is not None:
        plot_hand_mesh(ax, pred_vertices, faces, color='red', label='Pred Mesh')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    if camera is None:
        elev = random.uniform(10, 170)  # Random elevation angle between 10 and 170 degrees
        azim = random.uniform(0, 360)   # Random azimuth angle between 0 and 360 degrees
        ax.view_init(elev=elev, azim=azim)
    else:
        ax.view_init(elev=camera['elev'], azim=camera['azim'])

    fig.canvas.draw()  # Draw the canvas to render the plot
    image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Close the figure to free up memory
    plt.close(fig)

    return image_array


def save_debug_3d_mesh(config, meta, gt_vertices, pred_vertices, hand_type, faces, prefix, writer, global_step, camera=None):    
    idx = np.random.randint(0, gt_vertices.shape[0])
    
    if isinstance(gt_vertices, torch.Tensor):
        gt_vertices = gt_vertices.detach().cpu().numpy()
        
    if isinstance(pred_vertices, torch.Tensor):
        pred_vertices = pred_vertices.detach().cpu().numpy()

    hand_type = hand_type[idx]
    faces = faces[hand_type]
    gt_vertices = gt_vertices[idx] / 100 # mm -> cm
    pred_vertices = pred_vertices[idx] / 100 # mm -> cm

    image_array = draw_3d_mesh(gt_vertices, pred_vertices, faces, camera)

    writer.add_image(prefix + '_mesh', image_array, global_step, dataformats='HWC')

    return image_array


def main():
    import open3d as o3d
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering

    app = gui.Application.instance
    app.initialize()

    points = make_point_cloud(100, (0, 0, 0), 1.0)

    w = app.create_window("Open3D - 3D Text", 1024, 768)
    widget3d = gui.SceneWidget()
    widget3d.scene = rendering.Open3DScene(w.renderer)
    mat = rendering.Material()
    mat.shader = "defaultUnlit"
    mat.point_size = 5 * w.scaling
    widget3d.scene.add_geometry("Points", points, mat)
    for idx in range(0, len(points.points)):
        widget3d.add_3d_label(points.points[idx], "{}".format(idx))
    bbox = widget3d.scene.bounding_box
    widget3d.setup_camera(60.0, bbox, bbox.get_center())
    w.add_child(widget3d)

    app.run()

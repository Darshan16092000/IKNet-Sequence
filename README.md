# IKNet-Develop: Enhanced Sequence-Based 3D Hand Pose Estimation

## Overview
IKNet-Develop is an enhanced version of the base IKNet model, developed as part of **Project Aria** at **Chris10M/Aria**. This improved model is specifically designed to handle sequence data (motion history) for predicting joint rotations in 3D hand pose estimation. The model incorporates novel techniques for interpreting the results, including **attention maps** and **skeleton visualizations**, to provide better insights into its predictions.

## Key Features

- **Sequence Handling**:  
  Unlike the original IKNet model, IKNet-Develop is optimized to process sequences of motion data, capturing temporal dependencies to predict joint rotations more accurately.

- **Attention Maps**:  
  The model includes visualization techniques, such as attention maps, to help understand how previous frames influence the prediction of the current frame. This allows for more transparent model behavior.

- **Skeleton View Visualization**:  
  A skeleton view representation is incorporated to visualize the predicted 3D hand poses in a comprehensible way, helping in the analysis of hand motion.

## Goals
The primary goal of IKNet-Develop is to **improve the accuracy and interpretability** of 3D hand pose estimation. By leveraging sequence data and introducing interpretability methods, this model aims to enhance the understanding of how different factors, such as hand motion history, contribute to final pose predictions.

## How it Works

1. **Sequence Data Input**:  
   The model takes a sequence of hand poses (multiple frames of motion) as input, processed as a combined tensor.

2. **Transformer Encoder**:  
   A **Transformer encoder** is used to capture temporal dependencies across frames, allowing the model to leverage previous motion history when predicting the hand pose at the current time step.

3. **Prediction**:  
   The model predicts **joint rotations** for the last frame of the sequence, based on the temporal context provided by earlier frames.

4. **Interpretation**:  
   - **Attention maps** are generated to analyze which frames (or parts of frames) the model attends to most when making predictions.
   - Additionally, the model can visualize the predicted hand pose in 3D using a **skeleton view**.

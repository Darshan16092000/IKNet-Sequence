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
   
4. **Performance Metrics: Training vs Validation Accuracy**
To quantitatively evaluate the model's performance, we measured the **Mean Per Joint Position Error (MPJPE)** in millimeters — a widely used metric for 3D hand pose estimation. This metric calculates the average Euclidean distance between predicted and ground-truth joint positions.

**Note:**

**All values are in millimeters (mm).**

**Lower MPJPE indicates better accuracy.**

**The blue line in the plots corresponds to the IKNet-Sequence model, which incorporates temporal information via a Transformer encoder and black line corresponds to the baseline IKNet model.**

| Training Accuracy | Validation Accuracy |
|-------------------|---------------------|
| <img src="https://github.com/user-attachments/assets/9869676a-25e9-40b6-801b-1a99e67ed954" width="300"/> | <img src="https://github.com/user-attachments/assets/ea70eaab-5768-4eb7-9043-e2dda9a0064d" width="300"/> |



5. **Interpretation  & Visualization**:
To evaluate the effectiveness of the proposed real-time 3D hand pose estimation approach using IKNet and temporal information, I analyzed both qualitative and quantitative outputs across different experimental configurations.
   - **Attention maps** are generated to analyze which frames (or parts of frames) the model attends to most when making predictions. A key component of the temporal model was the self-attention mechanism within the Transformer encoder. By visualizing the attention maps, we could interpret how the model weighted each frame in the input sequence when predicting the final frame. This helped us assess whether the model learned to focus more on temporally closer frames or leveraged distant frame information for motion consistency.
   - **Observation**: The attention visualization revealed that the model primarily focused on the immediate previous frame but also gave subtle weight to earlier frames, suggesting that temporal context helps stabilize noisy predictions.

| Attention Map | Original Image |
|---------------|----------------|
| <img src="https://github.com/user-attachments/assets/ecd05470-da60-4dea-b1b2-ee5ed5f46bae" width="300"/> | <img src="https://github.com/user-attachments/assets/9a5a91c8-da43-41ab-a2e8-28d17d514531" width="300"/> |


   - **Pose Prediction Output** is plotted the predicted 3D joint coordinates for different hand poses and compared them with the ground-truth annotations to assess spatial accuracy. These were visualized as 3D skeletons to better understand joint alignment and articulation.
   - **Observation**: While single-frame predictions performed reasonably well, adding temporal information led to smoother and more anatomically correct predictions, especially in cases involving fast or complex hand gestures.
   - So, the model can visualize the predicted hand pose in 3D using a **skeleton view**.
   <img src="https://github.com/user-attachments/assets/fb570e91-cb72-4a31-8257-2c26c41e12e2" alt="skeleton-viz" width="300"/>

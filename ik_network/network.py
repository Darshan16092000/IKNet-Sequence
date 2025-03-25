import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch_geometric.data as pyg_data
import math	

class IKNet1(nn.Module):
    def __init__(self, cfg, in_channels=3, hidden_dim=64, num_heads=4, global_rot_dim=6):
        super(IKNet, self).__init__()

        self.cfg = cfg

        # Update in_channels to account for global rotation input
        self.in_channels = in_channels + global_rot_dim  # New input feature dimension

        # Define the hand skeleton edge index for hand (21 joints)
        self.edge_index = torch.tensor([
            [0, 1], [1, 2], [2, 3], [3, 4],   # Thumb
            [0, 5], [5, 6], [6, 7], [7, 8],   # Index
            [0, 9], [9, 10], [10, 11], [11, 12],  # Middle
            [0, 13], [13, 14], [14, 15], [15, 16],  # Ring
            [0, 17], [17, 18], [18, 19], [19, 20],  # Pinky
        ], dtype=torch.long).t().contiguous()  # Shape: (2, num_edges)

        # GAT layers with updated in_channels
        self.gat1 = pyg_nn.GATConv(self.in_channels, hidden_dim, heads=num_heads, concat=True)
        self.gat2 = pyg_nn.GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, concat=True)
        self.gat3 = pyg_nn.GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, concat=False)  # Last layer outputs hidden_dim

        # Output layers for predicting joint properties
        self.theta_head = nn.Linear(hidden_dim, global_rot_dim)  # Predicting 6D rotation for each joint

        # Global Rotation Prediction Head
        # We can aggregate node features to predict global rotation
        self.global_rot_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, global_rot_dim)
        )

    def forward(self, palm_normalized_root_relative_joints, global_rotation):
        B, J, D = palm_normalized_root_relative_joints.shape  # B: Batch size, J: 21 joints, D: (x, y, z)

        self.edge_index = self.edge_index.to(palm_normalized_root_relative_joints.device)

        # Process global rotation
        global_rot_features = global_rotation.view(B, -1)  # Shape: (B, global_rot_dim)

        batched_graphs = []
        for i in range(B):
            # Repeat global rotation features for each node
            repeated_global_rot = global_rot_features[i].unsqueeze(0).repeat(J, 1)  # Shape: (J, global_rot_dim)

            # Concatenate node features with global rotation
            node_features = palm_normalized_root_relative_joints[i]  # Shape: (J, in_channels)
            node_features = torch.cat([node_features, repeated_global_rot], dim=1)  # Shape: (J, in_channels + global_rot_dim)

            data = pyg_data.Data(x=node_features, edge_index=self.edge_index)
            batched_graphs.append(data)

        data = pyg_data.Batch.from_data_list(batched_graphs)

        # Apply GAT layers with batched data
        x = self.gat1(data.x, data.edge_index)
        x = torch.relu(x)
        x = self.gat2(x, data.edge_index)
        x = torch.relu(x)
        x = self.gat3(x, data.edge_index)  # Shape: (B * J, hidden_dim)

        # Predict joint rotations
        rot_6ds = self.theta_head(x)  # Shape: (B * J, 6)

        # Reshape to (B, J, 6)
        rot_6ds = rot_6ds.view(B, J, -1)

        # Aggregate node features for global rotation prediction
        # Here we use mean pooling across nodes
        x_pooled = pyg_nn.global_mean_pool(x, data.batch)  # Shape: (B, hidden_dim)

        # Predict global rotation
        predicted_global_rot = self.global_rot_head(x_pooled)  # Shape: (B, global_rot_dim)

        output = {
            'rot_6ds': rot_6ds,
            'predicted_global_rot': predicted_global_rot
        }

        return output


class IKNet2(nn.Module):
    def __init__(self, cfg, in_channels=3, hidden_dim=64, num_heads=4, global_rot_dim=6):
        super(IKNet, self).__init__()

        self.cfg = cfg
        self.num_joints = 21
        self.hidden_dim = hidden_dim
        self.global_rot_dim = global_rot_dim

        # Define the hand skeleton edge index for hand (21 joints)
        # Edges represent hierarchical relationships (parent to child)
        self.edge_index = torch.tensor([
            # Wrist (node 0) connections
            [0, 1], [1, 2], [2, 3], [3, 4],    # Thumb
            [0, 5], [5, 6], [6, 7], [7, 8],    # Index
            [0, 9], [9, 10], [10, 11], [11, 12],  # Middle
            [0, 13], [13, 14], [14, 15], [15, 16],  # Ring
            [0, 17], [17, 18], [18, 19], [19, 20],  # Pinky
        ], dtype=torch.long).t().contiguous()  # Shape: (2, num_edges)

        # Build a fully connected graph for global attention
        self.full_edge_index = self.get_fully_connected_edges(self.num_joints)

        # Node feature dimension
        self.in_channels = in_channels

        # Initial node embedding layer
        self.node_emb = nn.Linear(self.in_channels, hidden_dim)

        # Global rotation embedding layer
        self.global_rot_emb = nn.Linear(global_rot_dim, hidden_dim)

        # Self-Attention Layers (Transformer Encoder Layers)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Output layers for predicting joint rotations
        self.theta_head = nn.Linear(hidden_dim, 6)  # Predicting 6D rotation for each joint

        # Global Rotation Refinement Head
        self.global_rot_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, global_rot_dim)
        )

    def get_fully_connected_edges(self, num_nodes):
        # Create edges for a fully connected graph
        row = torch.arange(num_nodes).repeat_interleave(num_nodes)
        col = torch.arange(num_nodes).repeat(num_nodes)
        edge_index = torch.stack([row, col], dim=0)
        return edge_index

    def forward(self, palm_normalized_root_relative_joints, global_rotation):
        B, J, D = palm_normalized_root_relative_joints.shape  # B: Batch size, J: 21 joints, D: (x, y, z)

        device = palm_normalized_root_relative_joints.device
        self.edge_index = self.edge_index.to(device)
        self.full_edge_index = self.full_edge_index.to(device)

        # Flatten batch and joints for processing
        node_features = palm_normalized_root_relative_joints.view(B * J, -1)  # Shape: (B * J, in_channels)

        # Initial node embeddings
        x = self.node_emb(node_features)  # Shape: (B * J, hidden_dim)

        # Reshape for transformer input
        x = x.view(B, J, self.hidden_dim).permute(1, 0, 2)  # Shape: (J, B, hidden_dim)

        # Process global rotation
        # Embed global rotation
        global_rot_emb = self.global_rot_emb(global_rotation)  # Shape: (B, hidden_dim)
        global_rot_emb = global_rot_emb.unsqueeze(0)  # Shape: (1, B, hidden_dim)

        # Concatenate global rotation as an additional node
        x = torch.cat([x, global_rot_emb], dim=0)  # Shape: (J + 1, B, hidden_dim)

        # Apply transformer encoder (self-attention layers)
        x = self.transformer(x)  # Shape: (J + 1, B, hidden_dim)

        # Split the output back to joint features and global rotation feature
        joint_features = x[:-1, :, :]  # Shape: (J, B, hidden_dim)
        global_feature = x[-1, :, :]    # Shape: (B, hidden_dim)

        # Predict joint rotations
        joint_features = joint_features.permute(1, 0, 2).contiguous()  # Shape: (B, J, hidden_dim)
        rot_6ds = self.theta_head(joint_features)  # Shape: (B, J, 6)

        # Predict refined global rotation
        predicted_global_rot = self.global_rot_head(global_feature)  # Shape: (B, global_rot_dim)

        output = {
            'rot_6ds': rot_6ds,
            'predicted_global_rot': predicted_global_rot
        }

        return output


class IKNet(nn.Module):
    # Create the paths for each joint to its root (joint 0)
    print('IKNetv model')
    def create_joint_paths(self):
        joint_paths = {
            # Joint paths from the root joint (0) to each specific joint
            0: [0],
            1: [0, 1],
            2: [0, 1, 2],
            3: [0, 1, 2, 3],
            4: [0, 1, 2, 3, 4],
            5: [0, 5],
            6: [0, 5, 6],
            7: [0, 5, 6, 7],
            8: [0, 5, 6, 7, 8],
            9: [0, 9],
            10: [0, 9, 10],
            11: [0, 9, 10, 11],
            12: [0, 9, 10, 11, 12],
            13: [0, 13],
            14: [0, 13, 14],
            15: [0, 13, 14, 15],
            16: [0, 13, 14, 15, 16],
            17: [0, 17],
            18: [0, 17, 18],
            19: [0, 17, 18, 19],
            20: [0, 17, 18, 19, 20],
        }
        # Maximum path length determines the tensor dimensions
        max_path_length = max(len(path) for path in joint_paths.values())
        
        # Initialize tensors to store joint paths and their masks
        paths = torch.zeros((self.num_joints, max_path_length), dtype=torch.long)
        path_masks = torch.zeros((self.num_joints, max_path_length), dtype=torch.bool)
        
        for j in range(self.num_joints):
            # Fill tensors with joint paths and their respective masks
            path = joint_paths[j]
            path_length = len(path)
            paths[j, :path_length] = torch.tensor(path, dtype=torch.long)
            path_masks[j, :path_length] = 1
        
        # Store paths and masks as buffers to avoid being updated during training
        self.register_buffer('paths', paths)
        self.register_buffer('path_masks', path_masks)

    def __init__(self, cfg, in_channels=3, hidden_dim=64, num_heads=4, rot_dim=6, num_layers=2):
        super(IKNet, self).__init__()
        
        self.cfg = cfg  # Configuration parameters
        self.in_channels = in_channels  # Input channels (e.g., x, y, z for each joint)
        self.hidden_dim = hidden_dim  # Transformer hidden dimension
        self.rot_dim = rot_dim  # Output rotation dimensions
        self.num_joints = 21  # Total number of hand joints
        self.sequence_length = 30  # Number of time steps in the input sequence

        max_len = self.sequence_length * self.num_joints  # 30 frames * 21 joints
        position = torch.arange(max_len).unsqueeze(1)  # [630, 1]
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim))
        pe = torch.zeros(1, max_len, hidden_dim)  # [1, 630, hidden_dim]
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # Register as buffer
        
        # Linear layers for embedding input data
        self.node_emb = nn.Linear(self.in_channels, hidden_dim)
        self.global_rot_emb = nn.Linear(3, hidden_dim)
        
        # Embeddings for joint indices and temporal information
        self.joint_index_emb = nn.Embedding(num_embeddings=self.num_joints, embedding_dim=hidden_dim)
        self.time_emb = nn.Embedding(self.sequence_length, hidden_dim)
        


        # Transformer encoder setup
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=None)
        self.input_norm = nn.LayerNorm(hidden_dim)  # Add normalization layer


        # Store attention weights (initialize as empty)
        self.attn_weights = None

        # Force the self-attention in the last layer to return attention weights.
        # Save the original forward function.
        old_forward = self.transformer.layers[-1].self_attn.forward

        def new_forward(query, key, value, **kwargs):
            # Force need_weights to True.
            kwargs['need_weights'] = True
            return old_forward(query, key, value, **kwargs)

        self.transformer.layers[-1].self_attn.forward = new_forward


        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                self.attn_weights = output[1].detach()  # output is (attn_output, attn_weights)
                # print(f"Attention Weights: {self.attn_weights.shape}")

            else:
                # This branch should not be hit now.
                self.attn_weights = None
                # print("Warning: attention weights not returned!")
        self.transformer.layers[-1].self_attn.register_forward_hook(hook_fn)


        
        # Output heads for joint rotations and global orientation
        self.theta_head = nn.Linear(hidden_dim, rot_dim)
        self.global_rot_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, rot_dim)
        )
        
        # Create joint paths and masks
        self.create_joint_paths()
        
        # MANO-specific joint indices for rotation prediction
        self.mano_joint_indices = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]

    def forward(self, global_orient, palm_normalized_root_relative_joints):
        # Input shapes:
        # global_orient: (B, 3) where B is the batch size
        # palm_normalized_root_relative_joints: (B, T, J, D)
        B, T, J, D = palm_normalized_root_relative_joints.shape
        device = palm_normalized_root_relative_joints.device  # Ensure compatibility with device
        
         # Global orientation [32, 30, 3]
         # Palm normalized root relative joints [32, 30, 21, 3]

        # Flatten joint coordinates for embedding
        node_features = palm_normalized_root_relative_joints.view(B * T * J, -1) #[20160, 3]
        x = self.node_emb(node_features) #[20160, 64]
        x = x.view(B, T, J, self.hidden_dim) #[32, 30, 21, 64]
        

        # Add temporal embeddings
        temporal_indices = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        time_embeddings = self.time_emb(temporal_indices).unsqueeze(2)  # Shape: (B, T, 1, hidden_dim)
        x = x + time_embeddings  # Add time embeddings to node features

        # Flatten time and joint dimensions for transformer input
        x = x.view(B, T * J, self.hidden_dim)

        # ========== NEW: Add positional encoding ==========
        x = x + self.pe[:, :T*J, :].to(device)  # [B, 630, hidden_dim]
        
        # Expand joint paths and masks for the batch
        paths_expanded = self.paths.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1).to(device)  #[32, 5, 21, 5]
        path_masks_expanded = self.path_masks.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1).to(device)  #[32, 5, 21, 5]
    

        # Embed joint paths and apply masks
        path_embeddings = self.joint_index_emb(paths_expanded)
        path_embeddings = path_embeddings * path_masks_expanded.unsqueeze(-1)
        path_encodings = path_embeddings.sum(dim=3)

        # Add joint path encodings to the input
        x = x + path_encodings.view(B, T * J, self.hidden_dim)
        
        # Transformer expects input shape: (seq_len, batch_size, hidden_dim)
        x = x.permute(1, 0, 2)  # [105, 32, 64]  Shape: (T * J, B, hidden_dim)

        
        # Embed the global orientation and concatenate it with joint features
        global_rot_emb = self.global_rot_emb(global_orient)  # Shape: (32, 5, 64)
        global_rot_emb = global_rot_emb.permute(1, 0, 2)  # Shape: (5, 32, 64)
        x = torch.cat([x, global_rot_emb], dim=0)
    
        
        x = self.input_norm(x)  # Normalize input embeddings
        # Pass through the transformer
        x = self.transformer(x)
        # print(f"X: {x.shape}")  # [110, 32, 64]
        
        

        # Split transformer output into joint-specific and global features
        joint_features = x[:-30, :, :]
        global_feature = x[-30:, :, :]

    
        # Reshape joint features back to original dimensions
        joint_features = joint_features.permute(1, 0, 2).contiguous()
        joint_features = joint_features.view(B, T, J, self.hidden_dim)

        global_feature = global_feature.permute(1, 0, 2).contiguous()
        global_feature = global_feature.view(B, T, self.hidden_dim)

        # Select only the last time step's features
        last_joint_features = joint_features[:, -1, :, :]  # Shape: (B, J, hidden_dim)
        last_global_feature = global_feature[:, -1, :]     # Shape: (B, hidden_dim)
        
         # Predict rotations for the last frame
        selected_last_joints = last_joint_features[:, self.mano_joint_indices, :]  # [32, 15, 64]
        joint_rots = self.theta_head(selected_last_joints)  # Shape: (B, 15, 6)

        # Predict global orientation
        predicted_global_rot = self.global_rot_head(last_global_feature)

        # Return predictions as a dictionary
        output = {
            'hand_pose': joint_rots,  # Joint rotations for MANO joints
            'global_orient': predicted_global_rot  # Global hand orientation
        }
        # print(f"Output: {output}")
        return output, self.attn_weights



class IKNet3(nn.Module):
    print('IKNet3 model')
    def create_joint_paths(self):
        # Define the paths from the root to each joint
        joint_paths = {
            0: [0],
            1: [0, 1],
            2: [0, 1, 2],
            3: [0, 1, 2, 3],
            4: [0, 1, 2, 3, 4],
            5: [0, 5],
            6: [0, 5, 6],
            7: [0, 5, 6, 7],
            8: [0, 5, 6, 7, 8],
            9: [0, 9],
            10: [0, 9, 10],
            11: [0, 9, 10, 11],
            12: [0, 9, 10, 11, 12],
            13: [0, 13],
            14: [0, 13, 14],
            15: [0, 13, 14, 15],
            16: [0, 13, 14, 15, 16],
            17: [0, 17],
            18: [0, 17, 18],
            19: [0, 17, 18, 19],
            20: [0, 17, 18, 19, 20],
        }

        # Compute maximum path length
        max_path_length = max(len(path) for path in joint_paths.values())

        # Create tensors for paths and masks
        paths = torch.zeros((self.num_joints, max_path_length), dtype=torch.long)
        path_masks = torch.zeros((self.num_joints, max_path_length), dtype=torch.bool)

        # Populate paths and masks
        for j in range(self.num_joints):
            path = joint_paths[j]
            path_length = len(path)
            paths[j, :path_length] = torch.tensor(path, dtype=torch.long)
            path_masks[j, :path_length] = 1

        # Register buffers
        self.register_buffer('paths', paths)
        self.register_buffer('path_masks', path_masks)

    def __init__(self, cfg, in_channels=3, hidden_dim=64, num_heads=4, rot_dim=6):
        super(IKNet3, self).__init__()

        self.cfg = cfg
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.rot_dim = rot_dim
        self.num_joints = 21

        # Initial node embedding layer
        self.node_emb = nn.Linear(self.in_channels, hidden_dim)

        # Global rotation embedding layer
        self.global_rot_emb = nn.Linear(3, hidden_dim)

        # Joint index embeddings
        self.joint_index_emb = nn.Embedding(num_embeddings=self.num_joints, embedding_dim=hidden_dim)

        # Self-Attention Layers (Transformer Encoder Layers)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Output layers for predicting joint rotations
        self.theta_head = nn.Linear(hidden_dim, rot_dim)  # Predicting 6D rotation for each joint

        # Global Rotation Refinement Head
        self.global_rot_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, rot_dim)
        )

        self.create_joint_paths()

        self.mano_joint_indices = [
                                    1, 2, 3, 
                                    5, 6, 7, 
                                    9, 10, 11, 
                                    13, 14, 15, 
                                    17, 18, 19
                                ]
        
    def forward(self, global_orient, palm_normalized_root_relative_joints):
        B, J, D = palm_normalized_root_relative_joints.shape  # B: Batch size, J: 21 joints, D: (x, y, z)
        device = palm_normalized_root_relative_joints.device

        # Flatten batch and joints for processing
        node_features = palm_normalized_root_relative_joints.view(B * J, -1)  # Shape: (B * J, in_channels)

        # Initial node embeddings
        x = self.node_emb(node_features)  # Shape: (B * J, hidden_dim)

        # Reshape for batch processing
        x = x.view(B, J, self.hidden_dim)  # Shape: (B, J, hidden_dim)

        # Expand paths and masks
        paths_expanded = self.paths.unsqueeze(0).expand(B, -1, -1).to(device)  # Shape: (B, J, max_path_length)
        path_masks_expanded = self.path_masks.unsqueeze(0).expand(B, -1, -1).to(device)  # Shape: (B, J, max_path_length)

        # Embed the paths
        path_embeddings = self.joint_index_emb(paths_expanded)  # Shape: (B, J, max_path_length, hidden_dim)

        # Apply masks and aggregate embeddings
        path_embeddings = path_embeddings * path_masks_expanded.unsqueeze(-1)  # Apply masks
        path_encodings = path_embeddings.sum(dim=2)  # Sum over path_length dimension

        # Incorporate path encodings into node features
        x = x + path_encodings  # Shape: (B, J, hidden_dim)

        # Transpose for transformer input
        x = x.permute(1, 0, 2)  # Shape: (J, B, hidden_dim)

        # Process global rotation
        # Embed global rotation
        global_rot_emb = self.global_rot_emb(global_orient)  # Shape: (B, hidden_dim)
        global_rot_emb = global_rot_emb.unsqueeze(0)  # Shape: (1, B, hidden_dim)

        # Concatenate global rotation as an additional node
        x = torch.cat([x, global_rot_emb], dim=0)  # Shape: (J + 1, B, hidden_dim)

        # Apply transformer encoder (self-attention layers)
        x = self.transformer(x)  # Shape: (J + 1, B, hidden_dim)

        # Split the output back to joint features and global rotation feature
        joint_features = x[:-1, :, :]  # Shape: (J, B, hidden_dim)
        global_feature = x[-1, :, :]    # Shape: (B, hidden_dim)

        # Predict joint rotations
        joint_features = joint_features.permute(1, 0, 2).contiguous()  # Shape: (B, J, hidden_dim)

        selected_joint_features = joint_features[:, self.mano_joint_indices, :]  # Shape: (B, 15, hidden_dim)

        joint_rots = self.theta_head(selected_joint_features)  # Shape: (B, J, 6)

        # Predict refined global rotation
        predicted_global_rot = self.global_rot_head(global_feature)  # Shape: (B, global_rot_dim)

        output = {
            'hand_pose': joint_rots,
            'global_orient': predicted_global_rot
        }

        return output

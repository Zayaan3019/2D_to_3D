# netisg/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetOffsetPredictor(nn.Module):
    """
    A PointNet-based architecture to predict a 3D offset vector for each
    input point, moving it closer to the skeleton.
    """
    def __init__(self, num_points=2048, in_channels=3, out_channels=3):
        super(PointNetOffsetPredictor, self).__init__()
        self.num_points = num_points
        
        # Point-wise feature encoders
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        
        # Global feature extractor
        self.conv4 = nn.Conv1d(256, 512, 1)
        
        # Decoder for per-point offset prediction
        self.dec_conv1 = nn.Conv1d(512 + 256, 256, 1) # Concatenated global and local features
        self.dec_conv2 = nn.Conv1d(256, 128, 1)
        self.dec_conv3 = nn.Conv1d(128, out_channels, 1)

    def forward(self, x):
        # x shape: (B, C, N) -> (B, 3, num_points)
        
        # --- Encoder ---
        # Extract point-wise features
        x1 = F.relu(self.conv1(x))    # (B, 64, N)
        x2 = F.relu(self.conv2(x1))   # (B, 128, N)
        x3 = F.relu(self.conv3(x2))   # (B, 256, N)
        
        # --- Global Feature Vector ---
        global_features_raw = F.relu(self.conv4(x3)) # (B, 512, N)
        global_features = torch.max(global_features_raw, 2, keepdim=True)[0] # (B, 512, 1)
        global_features_expanded = global_features.repeat(1, 1, self.num_points) # (B, 512, N)
        
        # --- Decoder ---
        # Concatenate global features with local features
        combined_features = torch.cat([x3, global_features_expanded], dim=1) # (B, 256 + 512, N)
        
        # Predict per-point offsets
        d1 = F.relu(self.dec_conv1(combined_features)) # (B, 256, N)
        d2 = F.relu(self.dec_conv2(d1))               # (B, 128, N)
        offsets = self.dec_conv3(d2)                  # (B, 3, N)
        
        # The output shape should be (B, 3, N), which is a 3D vector per point.
        # Transpose to get (B, N, 3) for easier use.
        return offsets.transpose(1, 2)

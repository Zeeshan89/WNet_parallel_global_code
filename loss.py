import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import warnings
from warnings import filterwarnings

filterwarnings("ignore", category=DeprecationWarning)
filterwarnings("ignore", category=FutureWarning)
filterwarnings("ignore", category=UserWarning)
filterwarnings("ignore", category=RuntimeWarning)
from scipy.ndimage import zoom
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from torch.cuda.amp import autocast
import torch.optim as optim
import GPUtil
from torch.nn import functional as F
from scipy import ndimage
from skimage.measure import regionprops
import math


# Option 1: Dice Loss - GPU Optimized
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Ensure inputs are probabilities (sigmoid already applied)
        # Flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        # Intersection is equivalent to True Positive count
        intersection = (inputs * targets).sum()                            
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)  
        
        return 1 - dice


# Option 2: Focal Loss - GPU Optimized with autocast compatibility
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.40, gamma=2.5, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Use binary_cross_entropy_with_logits for autocast compatibility
        # Assume inputs are logits, not probabilities
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


def gaussian_kernel(radius: int = 3, sigma: float = 4, device='cpu'):
    """GPU-optimized Gaussian kernel generation"""
    x_2 = np.linspace(-radius, radius, 2*radius+1) ** 2
    dist = np.sqrt(x_2.reshape(-1, 1) + x_2.reshape(1, -1)) / sigma
    kernel = norm.pdf(dist) / norm.pdf(0)
    kernel = torch.from_numpy(kernel.astype(np.float32))
    kernel = kernel.view((1, 1, kernel.shape[0], kernel.shape[1]))
    return kernel.to(device)


# Option 3: NCut Loss - GPU Optimized
class NCutLoss2D(nn.Module):
    def __init__(self, radius: int = 4, sigma_1: float = 4, sigma_2: float = 1):
        super(NCutLoss2D, self).__init__()
        self.radius = radius
        self.sigma_1 = sigma_1  # Spatial standard deviation
        self.sigma_2 = sigma_2  # Pixel value standard deviation
        self.kernel = None

    def forward(self, labels: Tensor, inputs: Tensor) -> Tensor:
        """
        Args:
            labels: Predicted segmentation probabilities (B, C, H, W)
            inputs: Input images (B, C_in, H, W)
        """
        num_classes = labels.shape[1]
        
        # Create kernel if not exists or device changed
        if self.kernel is None or self.kernel.device != labels.device:
            self.kernel = gaussian_kernel(
                radius=self.radius, 
                sigma=self.sigma_1, 
                device=labels.device
            )
        
        loss = 0
        
        # Use mean of input channels for similarity computation if multi-channel
        if inputs.shape[1] > 1:
            inputs_mean = inputs.mean(dim=1, keepdim=True)
        else:
            inputs_mean = inputs
            
        for k in range(num_classes):
            # Compute the average pixel value for this class, and the difference from each pixel
            class_probs = labels[:, k].unsqueeze(1)
            class_mean = torch.mean(inputs_mean * class_probs, dim=(2, 3), keepdim=True) / \
                torch.add(torch.mean(class_probs, dim=(2, 3), keepdim=True), 1e-5)
            diff = (inputs_mean - class_mean).pow(2).sum(dim=1).unsqueeze(1)
            # Weight the loss by the difference from the class average.
            weights = torch.exp(diff.pow(2).mul(-1 / self.sigma_2 ** 2))
            # Compute N-cut loss, using the computed weights matrix, and a Gaussian spatial filter
            numerator = torch.sum(class_probs * F.conv2d(class_probs * weights, self.kernel, padding=self.radius))
            denominator = torch.sum(class_probs * F.conv2d(weights, self.kernel, padding=self.radius))
            loss += nn.L1Loss()(numerator / torch.add(denominator, 1e-6), torch.zeros_like(numerator))
            
        return num_classes - loss


# Option 4: GPU-Optimized AR Segmentation Loss
class ARSegmentationLoss(nn.Module):
    def __init__(self, 
             # AR-specific weights (keep these)
             ar_pattern_weight=4.0, ar_intensity_weight=3.5, ar_consistency_weight=2.0,
             ar_positive_weight=8.0, ar_focal_weight=5.0,ar_multi_criteria_weight=7.0, ar_synergy_weight=4.5,
             
             # Hurricane-specific weights - REMOVE multi_criteria and synergy weights
             hurricane_intensity_weight=4.0,
             hurricane_focal_weight=6.0, hurricane_positive_weight=12.0,
             hurricane_consistency_weight=2.0, hurricane_pattern_weight=5.0,
             
             # ETC-specific weights - REMOVE multi_criteria, synergy, and component weights
             etc_intensity_weight=3.5,
             etc_focal_weight=5.0, etc_positive_weight=10.0, etc_consistency_weight=2.0,
             
             # MCS-specific weights - REMOVE multi_criteria and synergy weights
             mcs_intensity_weight=3.0,
             mcs_focal_weight=4.0, mcs_positive_weight=8.0, mcs_consistency_weight=2.0,
             mcs_pattern_weight=3.5,
             
             # Class weights
             class_weight=[1.0, 10.0, 15.0, 12.0, 10.0]):
        
        super(ARSegmentationLoss, self).__init__()
        
        # Initialize loss functions with autocast compatibility
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()  # Changed to BCEWithLogitsLoss for autocast
        self.focal_loss = FocalLoss(alpha=0.35, gamma=2.5)  # Reduced gamma for stability
        
        # AR-specific weights
        # self.ivt_weight = ivt_weight
        # self.area_weight = area_weight
        # self.shape_weight = shape_weight
        # self.directional_weight = directional_weight
        # self.equatorial_penalty = equatorial_penalty
        # self.min_length_weight = min_length_weight
        # self.min_aspect_weight = min_aspect_weight
        # self.min_span_weight = min_span_weight
        self.ar_multi_criteria_weight = ar_multi_criteria_weight
        self.ar_pattern_weight = ar_pattern_weight
        self.ar_intensity_weight = ar_intensity_weight
        self.ar_consistency_weight = ar_consistency_weight
        self.ar_positive_weight = ar_positive_weight
        self.ar_focal_weight = ar_focal_weight
        self.ar_synergy_weight = ar_synergy_weight
        
        # Hurricane-specific weights (optimized for GPU memory)
        # self.hurricane_eye_weight = hurricane_eye_weight
        # self.hurricane_wind_weight = hurricane_wind_weight
        # self.hurricane_vorticity_weight = hurricane_vorticity_weight
        # self.hurricane_multi_criteria_weight = hurricane_multi_criteria_weight
        self.hurricane_intensity_weight = hurricane_intensity_weight
        self.hurricane_focal_weight = hurricane_focal_weight
        self.hurricane_positive_weight = hurricane_positive_weight
        # self.hurricane_shape_weight = hurricane_shape_weight
        self.hurricane_consistency_weight = hurricane_consistency_weight
        self.hurricane_pattern_weight = hurricane_pattern_weight
        
        # ETC-specific weights (new)
        # self.etc_cold_core_weight = etc_cold_core_weight
        # self.etc_pressure_weight = etc_pressure_weight
        # self.etc_vorticity_weight = etc_vorticity_weight
        # self.etc_wind_weight = etc_wind_weight
        # self.etc_multi_criteria_weight = etc_multi_criteria_weight
        self.etc_intensity_weight = etc_intensity_weight
        self.etc_focal_weight = etc_focal_weight
        self.etc_positive_weight = etc_positive_weight
        self.etc_consistency_weight = etc_consistency_weight
        # self.etc_pattern_weight = etc_pattern_weight
        # self.etc_core_weight = etc_core_weight
        
        # MCS-specific weights (new)
        # self.mcs_convection_weight = mcs_convection_weight
        # self.mcs_precip_weight = mcs_precip_weight
        # self.mcs_multi_criteria_weight = mcs_multi_criteria_weight
        self.mcs_intensity_weight = mcs_intensity_weight
        self.mcs_focal_weight = mcs_focal_weight
        self.mcs_positive_weight = mcs_positive_weight
        self.mcs_consistency_weight = mcs_consistency_weight
        self.mcs_pattern_weight = mcs_pattern_weight
        # self.mcs_core_weight = mcs_core_weight
        
        # Class weights for handling class imbalance (now 4 classes)
        self.class_weight = class_weight
        
        # Cache for spatial kernels
        self.spatial_kernel = None
    
    
    # MCS part
    # @autocast(enabled=False)
    # def multi_criteria_mcs_detection_loss(self, pred_logits, mcs_blobs_mask, 
    #                                     mcs_boundary_mask, mcs_core_pixels_mask):
    #     """
    #     Multi-criteria loss for MCS detection using TWO-THRESHOLD approach
    #     - Boundary threshold (241K) defines blob extent
    #     - Core threshold (225K) must exist in blob
    #     """
    #     pred_probs = torch.sigmoid(pred_logits)
        
    #     # Primary loss: Direct detection of MCS blobs (with core requirement)
    #     blob_detection = F.binary_cross_entropy_with_logits(
    #         pred_logits, mcs_blobs_mask, reduction='mean'
    #     )
        
    #     # Secondary losses: Alignment with boundary and core requirements
    #     boundary_component_loss = F.mse_loss(
    #         pred_probs * mcs_boundary_mask, 
    #         mcs_boundary_mask * mcs_blobs_mask
    #     )
        
    #     core_component_loss = F.mse_loss(
    #         pred_probs * mcs_core_pixels_mask, 
    #         mcs_core_pixels_mask * mcs_blobs_mask
    #     )
        
    #     # Combined multi-criteria loss
    #     multi_criteria_loss = (blob_detection + 
    #                         0.4 * boundary_component_loss +  # Boundary alignment
    #                         0.6 * core_component_loss)       # Core requirement (higher weight)
        
    #     return multi_criteria_loss

    # @autocast(enabled=False)
    # def mcs_core_pattern_loss(self, pred_logits, mcs_core_pixels_mask, mcs_blobs_mask):
    #     """
    #     Pattern matching loss for MCS core pixel structures (225K threshold)
    #     This ensures the model learns to identify the coldest, most intense convection
    #     """
    #     pred_probs = torch.sigmoid(pred_logits)
        
    #     # Direct pattern alignment for core pixels
    #     core_mse = F.mse_loss(
    #         pred_probs * mcs_core_pixels_mask, 
    #         mcs_core_pixels_mask * mcs_blobs_mask
    #     )
        
    #     # Focus on core intensity preservation (cores are critical for MCS identification)
    #     core_intensity_loss = F.mse_loss(
    #         pred_probs * (mcs_core_pixels_mask > 0).float(), 
    #         mcs_core_pixels_mask * mcs_blobs_mask
    #     )
        
    #     return core_mse + 0.5 * core_intensity_loss

    # @autocast(enabled=False)
    # def mcs_component_synergy_loss(self, pred_logits, mcs_boundary_mask, mcs_core_pixels_mask):
    #     """
    #     Component synergy loss for MCS - ensures boundary contains core
    #     This enforces the two-threshold relationship: cores must exist within boundaries
    #     """
    #     pred_probs = torch.sigmoid(pred_logits)
        
    #     # Key MCS synergy: boundary regions that contain core pixels
    #     boundary_with_core = mcs_boundary_mask * mcs_core_pixels_mask
        
    #     # Synergy loss focusing on the blob requirement (boundary + core)
    #     synergy_loss = F.mse_loss(
    #         pred_probs * boundary_with_core, 
    #         boundary_with_core
    #     )
        
    #     return synergy_loss

    @autocast(enabled=False)
    def mcs_boundary_pattern_loss(self, pred_logits, mcs_boundary_mask, mcs_blobs_mask):
        """
        Pattern matching loss for MCS boundary structures (241K threshold)
        """
        pred_probs = torch.sigmoid(pred_logits)
        
        # Direct pattern alignment for boundary regions
        boundary_mse = F.mse_loss(
            pred_probs * mcs_boundary_mask, 
            mcs_boundary_mask * mcs_blobs_mask
        )
        
        # Cosine similarity for boundary pattern matching
        pred_flat = (pred_probs * mcs_boundary_mask).view(pred_probs.size(0), -1)
        boundary_flat = (mcs_boundary_mask * mcs_blobs_mask).view(mcs_boundary_mask.size(0), -1)
        
        # Avoid division by zero
        pred_norm = torch.norm(pred_flat, dim=1, keepdim=True) + 1e-8
        boundary_norm = torch.norm(boundary_flat, dim=1, keepdim=True) + 1e-8
        cos_sim = torch.sum(pred_flat * boundary_flat, dim=1) / (pred_norm.squeeze() * boundary_norm.squeeze())
        correlation_loss = 1 - cos_sim.mean()
        
        return boundary_mse + 0.3 * correlation_loss

    @autocast(enabled=False)
    def mcs_intensity_preservation_loss(self, pred_logits, mcs_core_mask):
        """
        Intensity preservation loss for MCS regions
        """
        pred_probs = torch.sigmoid(pred_logits)
        
        # Focus on positive MCS regions
        positive_mcs_regions = (mcs_core_mask > 0).float()
        
        # Intensity matching in MCS regions
        intensity_loss = F.mse_loss(pred_probs * positive_mcs_regions, mcs_core_mask * positive_mcs_regions)
        
        return intensity_loss

    @autocast(enabled=False)
    def mcs_spatial_consistency_loss(self, pred_logits, mcs_core_mask):
        """
        Spatial consistency loss for MCS detection
        """
        pred_probs = torch.sigmoid(pred_logits)
        
        # Spatial smoothing with average pooling
        kernel_size = 3
        
        pred_smooth = F.avg_pool2d(pred_probs.unsqueeze(1), kernel_size, stride=1, padding=kernel_size//2).squeeze(1)
        mcs_smooth = F.avg_pool2d(mcs_core_mask.unsqueeze(1), kernel_size, stride=1, padding=kernel_size//2).squeeze(1)
        
        # Spatial consistency loss
        spatial_loss = F.mse_loss(pred_smooth, mcs_smooth)
        
        return spatial_loss
    
    @autocast(enabled=False)
    def mcs_positive_detection_loss(self, pred_logits, mcs_core_mask):
        """
        Positive MCS detection loss
        """
        # Binary mask for any positive MCS values
        positive_mask = (mcs_core_mask > 0).float()
        
        # BCE loss for positive detection
        positive_bce = F.binary_cross_entropy_with_logits(pred_logits, positive_mask, reduction='mean')
        
        # Focal loss for hard examples
        focal_loss = self.focal_loss(pred_logits, positive_mask)
        
        return positive_bce + 0.4 * focal_loss



    # Hurricane part
    @autocast(enabled=False)
    def hurricane_eye_pattern_matching_loss(self, pred_logits, strict_eye_mask):
        """
        Memory-efficient pattern matching loss for hurricane eye structures
        """
        pred_probs = torch.sigmoid(pred_logits)
        
        # Direct pattern alignment using MSE (more memory efficient than cross-correlation)
        pattern_mse = F.mse_loss(pred_probs, strict_eye_mask)
        
        # Simplified correlation using cosine similarity (more memory efficient)
        pred_flat = pred_probs.view(pred_probs.size(0), -1)
        eye_flat = strict_eye_mask.view(strict_eye_mask.size(0), -1)
        
        # Cosine similarity (more memory efficient than normalized cross-correlation)
        cos_sim = F.cosine_similarity(pred_flat, eye_flat, dim=1)
        correlation_loss = 1 - cos_sim.mean()
        
        return pattern_mse + 0.5 * correlation_loss  # Reduced weight for memory
    
    @autocast(enabled=False)
    def hurricane_eye_intensity_preservation_loss(self, pred_logits, strict_eye_mask):
        """
        Memory-efficient intensity preservation loss
        """
        pred_probs = torch.sigmoid(pred_logits)
        
        # Mask to focus only on positive eye regions
        positive_eye_regions = (strict_eye_mask > 0).float()
        
        # Intensity matching in eye regions (using MSE only for efficiency)
        intensity_loss = F.mse_loss(pred_probs * positive_eye_regions, strict_eye_mask * positive_eye_regions)
        
        return intensity_loss
    
    @autocast(enabled=False)
    def hurricane_eye_spatial_consistency_loss(self, pred_logits, strict_eye_mask):
        """
        Memory-efficient spatial consistency loss
        """
        pred_probs = torch.sigmoid(pred_logits)
        
        # Simplified spatial consistency using average pooling (more memory efficient)
        kernel_size = 3  # Reduced kernel size for memory efficiency
        
        # Apply average pooling for smoothing
        pred_smooth = F.avg_pool2d(pred_probs.unsqueeze(1), kernel_size, stride=1, padding=kernel_size//2).squeeze(1)
        eye_smooth = F.avg_pool2d(strict_eye_mask.unsqueeze(1), kernel_size, stride=1, padding=kernel_size//2).squeeze(1)
        
        # Spatial consistency loss
        spatial_loss = F.mse_loss(pred_smooth, eye_smooth)
        
        return spatial_loss
    
    @autocast(enabled=False)
    def hurricane_component_synergy_loss(self, pred_logits, wind_mask, vorticity_mask, pressure_mask):
        """
        Memory-efficient component synergy loss
        """
        pred_probs = torch.sigmoid(pred_logits)
        
        # Simplified synergy - only compute most important combinations
        wind_vorticity_combo = wind_mask * vorticity_mask
        
        # Single synergy loss for memory efficiency
        synergy_loss = F.mse_loss(pred_probs * wind_vorticity_combo, wind_vorticity_combo)
        
        return synergy_loss
    
    @autocast(enabled=False)
    def hurricane_positive_eye_detection_loss(self, pred_logits, strict_eye_mask):
        """
        Memory-efficient positive eye detection loss
        """
        # Binary mask for any positive values
        positive_mask = (strict_eye_mask > 0).float()
        
        # Use BCEWithLogitsLoss for autocast compatibility
        positive_bce = F.binary_cross_entropy_with_logits(pred_logits, positive_mask, reduction='mean')
        
        # Simplified focal loss
        focal_loss = self.focal_loss(pred_logits, positive_mask)
        
        return positive_bce + 0.5 * focal_loss  # Reduced weight for memory

    # @autocast(enabled=False)  # Disable autocast for specific operations
    # def multi_criteria_hurricane_detection_loss(self, pred_logits, strict_eye_mask, wind_mask, vorticity_mask, pressure_mask):
    #     """
    #     Enhanced loss for detecting hurricane eyes using multi-criteria approach
    #     Uses logits for better numerical stability
    #     """
    #     # Convert logits to probabilities for some operations
    #     pred_probs = torch.sigmoid(pred_logits)
        
    #     # Primary loss: Direct detection of strict hurricane eye (all 3 criteria)
    #     # Use BCEWithLogitsLoss for autocast compatibility
    #     strict_eye_detection = F.binary_cross_entropy_with_logits(
    #         pred_logits, strict_eye_mask, reduction='mean'
    #     )
        
    #     # Secondary losses: Individual component alignment (using MSE for efficiency)
    #     wind_component_loss = F.mse_loss(pred_probs * wind_mask, wind_mask * strict_eye_mask)
    #     vorticity_component_loss = F.mse_loss(pred_probs * vorticity_mask, vorticity_mask * strict_eye_mask)
    #     pressure_component_loss = F.mse_loss(pred_probs * pressure_mask, pressure_mask * strict_eye_mask)
        
    #     # Combined multi-criteria loss
    #     multi_criteria_loss = (strict_eye_detection + 
    #                           0.3 * wind_component_loss + 
    #                           0.3 * vorticity_component_loss + 
    #                           0.3 * pressure_component_loss)
        
    #     return multi_criteria_loss
    

    # ETC part
    # @autocast(enabled=False)
    # def multi_criteria_etc_detection_loss(self, pred_logits, etc_adaptive_mask, etc_pressure_mask, 
    #                                     etc_vorticity_mask, etc_wind_mask):
    #     """
    #     Multi-criteria loss for ETC detection with continuous adaptive mask support
    #     Now handles weighted masks instead of binary masks
    #     """
    #     # Convert logits to probabilities for some operations
    #     pred_probs = torch.sigmoid(pred_logits)
        
    #     # Primary loss: Weighted detection using adaptive mask intensities
    #     # Use MSE instead of BCE for continuous targets
    #     adaptive_detection_loss = F.mse_loss(pred_probs, etc_adaptive_mask)
        
    #     # Secondary losses: Individual ETC component alignment with continuous weighting
    #     pressure_component_loss = F.mse_loss(
    #         pred_probs * etc_pressure_mask, 
    #         etc_pressure_mask * etc_adaptive_mask
    #     )
        
    #     vorticity_component_loss = F.mse_loss(
    #         pred_probs * etc_vorticity_mask, 
    #         etc_vorticity_mask * etc_adaptive_mask
    #     )
        
    #     wind_component_loss = F.mse_loss(
    #         pred_probs * etc_wind_mask, 
    #         etc_wind_mask * etc_adaptive_mask
    #     )
        
    #     # Intensity-aware BCE for high-confidence regions
    #     high_confidence_mask = (etc_adaptive_mask > 0.7).float()
    #     if high_confidence_mask.sum() > 0:
    #         high_conf_bce = F.binary_cross_entropy_with_logits(
    #             pred_logits * high_confidence_mask, 
    #             high_confidence_mask, 
    #             reduction='mean'
    #         )
    #     else:
    #         high_conf_bce = torch.tensor(0.0, device=pred_logits.device)
        
    #     # Combined multi-criteria loss for ETC with adaptive weighting
    #     multi_criteria_loss = (adaptive_detection_loss + 
    #                         0.25 * pressure_component_loss + 
    #                         0.25 * vorticity_component_loss + 
    #                         0.2 * wind_component_loss +
    #                         0.3 * high_conf_bce)  # Extra term for high-confidence regions
        
    #     return multi_criteria_loss
    
    # @autocast(enabled=False)
    # def etc_intensity_preservation_loss(self, pred_logits, etc_adaptive_mask):
    #     """
    #     Enhanced intensity preservation loss for continuous ETC masks
    #     Now preserves the full intensity spectrum instead of just binary regions
    #     """
    #     pred_probs = torch.sigmoid(pred_logits)
        
    #     # Direct intensity matching for all ETC intensities
    #     intensity_mse_loss = F.mse_loss(pred_probs, etc_adaptive_mask)
        
    #     # Gradient-based intensity preservation (helps maintain smooth transitions)
    #     pred_grad_x = torch.abs(pred_probs[:, :, 1:] - pred_probs[:, :, :-1])
    #     pred_grad_y = torch.abs(pred_probs[:, 1:, :] - pred_probs[:, :-1, :])
        
    #     target_grad_x = torch.abs(etc_adaptive_mask[:, :, 1:] - etc_adaptive_mask[:, :, :-1])
    #     target_grad_y = torch.abs(etc_adaptive_mask[:, 1:, :] - etc_adaptive_mask[:, :-1, :])
        
    #     gradient_loss = (F.mse_loss(pred_grad_x, target_grad_x) + 
    #                     F.mse_loss(pred_grad_y, target_grad_y)) * 0.5
        
    #     # Weighted intensity focus (higher weight for stronger ETC signals)
    #     intensity_weights = etc_adaptive_mask + 0.1  # Ensure non-zero weights
    #     weighted_intensity_loss = F.mse_loss(pred_probs * intensity_weights, 
    #                                     etc_adaptive_mask * intensity_weights)
        
    #     return intensity_mse_loss + 0.3 * gradient_loss + 0.4 * weighted_intensity_loss
    
    # @autocast(enabled=False)
    # def etc_component_synergy_loss(self, pred_logits, etc_pressure_mask, etc_vorticity_mask, etc_adaptive_mask):
    #     """
    #     Enhanced component synergy loss with continuous mask support
    #     """
    #     pred_probs = torch.sigmoid(pred_logits)
        
    #     # Pressure-vorticity synergy weighted by adaptive mask
    #     pressure_vorticity_combo = etc_pressure_mask * etc_vorticity_mask
    #     weighted_combo = pressure_vorticity_combo * etc_adaptive_mask
        
    #     # Multi-level synergy loss
    #     basic_synergy = F.mse_loss(pred_probs * pressure_vorticity_combo, weighted_combo)
        
    #     # Intensity-aware synergy (focus on stronger systems)
    #     strong_systems_mask = (etc_adaptive_mask > 0.5).float()
    #     if strong_systems_mask.sum() > 0:
    #         strong_synergy = F.mse_loss(
    #             pred_probs * strong_systems_mask * pressure_vorticity_combo,
    #             strong_systems_mask * weighted_combo
    #         )
    #     else:
    #         strong_synergy = torch.tensor(0.0, device=pred_logits.device)
        
    #     return basic_synergy + 0.5 * strong_synergy

    # @autocast(enabled=False)
    # def etc_positive_detection_loss(self, pred_logits, etc_adaptive_mask):
    #     """
    #     Enhanced positive ETC detection loss for continuous masks
    #     Uses multi-threshold approach for different intensity levels
    #     """
    #     pred_probs = torch.sigmoid(pred_logits)
        
    #     # Multi-threshold detection for different intensities
    #     # thresholds = [0.3, 0.5, 0.7]  # Different intensity levels
    #     thresholds = [0.85]  # Different intensity levels
        
    #     threshold_losses = []
        
    #     for thresh in thresholds:
    #         binary_mask = (etc_adaptive_mask > thresh).float()
    #         if binary_mask.sum() > 0:
    #             # BCE loss for this threshold level
    #             thresh_bce = F.binary_cross_entropy_with_logits(pred_logits, binary_mask, reduction='mean')
    #             # Focal loss for hard examples at this threshold
    #             thresh_focal = self.focal_loss(pred_logits, binary_mask)
    #             threshold_losses.append(thresh_bce + 0.4 * thresh_focal)
        
    #     # Average across thresholds
    #     multi_threshold_loss = torch.mean(torch.stack(threshold_losses)) if threshold_losses else torch.tensor(0.0, device=pred_logits.device)
        
    #     # Continuous regression loss for preserving exact intensities
    #     continuous_loss = F.mse_loss(pred_probs, etc_adaptive_mask)
        
    #     return 0.6 * multi_threshold_loss + 0.4 * continuous_loss
    
    @autocast(enabled=False)
    def etc_spatial_consistency_loss(self, pred_logits, etc_adaptive_mask):
        """Spatial consistency for ETC - simplified"""
        pred_probs = torch.sigmoid(pred_logits)
        
        if torch.sum(etc_adaptive_mask) == 0:
            return torch.tensor(0.0, device=pred_logits.device)
        
        kernel_size = 3
        pred_smooth = F.avg_pool2d(pred_probs.unsqueeze(1), kernel_size, stride=1, padding=kernel_size//2).squeeze(1)
        target_smooth = F.avg_pool2d(etc_adaptive_mask.unsqueeze(1), kernel_size, stride=1, padding=kernel_size//2).squeeze(1)
        
        return F.mse_loss(pred_smooth, target_smooth)
    
    

    ##Ars part
    @autocast(enabled=False)
    def multi_criteria_ar_detection_loss(self, pred_logits, ar_mask, ivt_channel):
        """
        Multi-criteria loss for AR detection using IVT-based mask
        """
        pred_probs = torch.sigmoid(pred_logits)
        
        # Primary loss: Direct detection of AR regions from mask
        ar_detection = F.binary_cross_entropy_with_logits(
            pred_logits, ar_mask, reduction='mean'
        )
        
        # Secondary loss: IVT intensity alignment with AR regions
        ivt_norm = (ivt_channel - ivt_channel.min()) / (ivt_channel.max() - ivt_channel.min() + 1e-8)
        ivt_component_loss = F.mse_loss(pred_probs * ivt_norm, ivt_norm * ar_mask)
        
        # Combined multi-criteria loss
        multi_criteria_loss = ar_detection + 0.4 * ivt_component_loss
        
        return multi_criteria_loss

    @autocast(enabled=False)
    def ar_pattern_matching_loss(self, pred_logits, ar_mask):
        """
        Pattern matching loss for AR structures
        """
        pred_probs = torch.sigmoid(pred_logits)
        
        # Direct pattern alignment
        pattern_mse = F.mse_loss(pred_probs, ar_mask)
        
        # Cosine similarity for pattern matching
        pred_flat = pred_probs.view(pred_probs.size(0), -1)
        ar_flat = ar_mask.view(ar_mask.size(0), -1)
        
        # Avoid division by zero
        pred_norm = torch.norm(pred_flat, dim=1, keepdim=True) + 1e-8
        ar_norm = torch.norm(ar_flat, dim=1, keepdim=True) + 1e-8
        cos_sim = torch.sum(pred_flat * ar_flat, dim=1) / (pred_norm.squeeze() * ar_norm.squeeze())
        correlation_loss = 1 - cos_sim.mean()
        
        return pattern_mse + 0.3 * correlation_loss

    @autocast(enabled=False)
    def ar_intensity_preservation_loss(self, pred_logits, ar_mask):
        """
        Intensity preservation loss for AR regions
        """
        pred_probs = torch.sigmoid(pred_logits)
        
        # Focus on positive AR regions
        positive_ar_regions = (ar_mask > 0).float()
        
        # Intensity matching in AR regions
        intensity_loss = F.mse_loss(pred_probs * positive_ar_regions, ar_mask * positive_ar_regions)
        
        return intensity_loss

    @autocast(enabled=False)
    def ar_spatial_consistency_loss(self, pred_logits, ar_mask):
        """
        Spatial consistency loss for AR detection
        """
        pred_probs = torch.sigmoid(pred_logits)
        
        # Spatial smoothing with average pooling
        kernel_size = 3
        
        pred_smooth = F.avg_pool2d(pred_probs.unsqueeze(1), kernel_size, stride=1, padding=kernel_size//2).squeeze(1)
        ar_smooth = F.avg_pool2d(ar_mask.unsqueeze(1), kernel_size, stride=1, padding=kernel_size//2).squeeze(1)
        
        # Spatial consistency loss
        spatial_loss = F.mse_loss(pred_smooth, ar_smooth)
        
        return spatial_loss

    @autocast(enabled=False)
    def ar_positive_detection_loss(self, pred_logits, ar_mask):
        """
        Positive AR detection loss
        """
        # Binary mask for any positive AR values
        positive_mask = (ar_mask > 0).float()
        
        # BCE loss for positive detection
        positive_bce = F.binary_cross_entropy_with_logits(pred_logits, positive_mask, reduction='mean')
        
        # Focal loss for hard examples
        focal_loss = self.focal_loss(pred_logits, positive_mask)
        
        return positive_bce + 0.4 * focal_loss

    @autocast(enabled=False)
    def ar_ivt_component_synergy_loss(self, pred_logits, ar_mask, ivt_channel):
        """
        Component synergy loss for AR - focuses on IVT intensity alignment
        """
        pred_probs = torch.sigmoid(pred_logits)
        
        # Normalize IVT
        ivt_norm = (ivt_channel - ivt_channel.min()) / (ivt_channel.max() - ivt_channel.min() + 1e-8)
        
        # IVT-AR synergy
        ivt_ar_combo = ivt_norm * ar_mask
        
        # Synergy loss
        synergy_loss = F.mse_loss(pred_probs * ivt_ar_combo, ivt_ar_combo)
        
        return synergy_loss
    
    @autocast(enabled=False)
    def compute_ar_losses(self, ar_probs, target_ar, ivt_channel, ar_mask):
        """
        Compute AR-specific losses following the same pattern as other classes
        Now uses ar_mask (ivt_binary) as the primary detection target
        """
        # Convert to logits for BCEWithLogitsLoss
        ar_logits = torch.logit(ar_probs.clamp(1e-7, 1-1e-7))
        
        # Basic AR losses
        bce_ar = F.binary_cross_entropy_with_logits(
            ar_logits, target_ar, reduction='mean'
        ) * self.class_weight[1]  # AR is class index 1
        
        dice_ar = self.dice_loss(ar_probs, target_ar) * self.class_weight[1]
        
        # Multi-criteria AR detection using mask
        multi_criteria_loss = self.multi_criteria_ar_detection_loss(
            ar_logits, ar_mask, ivt_channel)
        
        # Pattern matching loss
        pattern_matching_loss = self.ar_pattern_matching_loss(
            ar_logits, ar_mask)
        
        # Intensity preservation loss
        intensity_preservation_loss = self.ar_intensity_preservation_loss(
            ar_logits, ar_mask)
        
        # Spatial consistency loss
        spatial_consistency_loss = self.ar_spatial_consistency_loss(
            ar_logits, ar_mask)
        
        # Component synergy loss (IVT alignment)
        component_synergy_loss = self.ar_ivt_component_synergy_loss(
            ar_logits, ar_mask, ivt_channel)
        
        # Positive AR detection loss
        positive_ar_loss = self.ar_positive_detection_loss(
            ar_logits, ar_mask)
        
        # Focal loss for hard examples
        positive_mask = (ar_mask > 0).float()
        ar_focal_loss = self.focal_loss(ar_logits, positive_mask)
        
        # Combine AR losses with same pattern as other classes
        ar_loss = (bce_ar + dice_ar + 
                 self.ar_multi_criteria_weight * multi_criteria_loss + 
                self.ar_pattern_weight * pattern_matching_loss + 
                self.ar_intensity_weight * intensity_preservation_loss + 
                self.ar_consistency_weight * spatial_consistency_loss + 
                self.ar_synergy_weight * component_synergy_loss + 
                self.ar_positive_weight * positive_ar_loss + 
                self.ar_focal_weight * ar_focal_loss)
        
        return ar_loss

    @autocast(enabled=False)
    def compute_hurricane_losses(self, hurricane_probs, target_hurricane, wind_mask, vorticity_mask, 
                            pressure_mask, strict_eye_mask, wind_speed_binary):
        """
        Compute hurricane-specific losses - SIMPLIFIED version using pre-processed masks
        """
        hurricane_logits = torch.logit(hurricane_probs.clamp(1e-7, 1-1e-7))
        
        # Basic hurricane losses using the strict mask (already validated in dataloader)
        bce_hurricane = F.binary_cross_entropy_with_logits(
            hurricane_logits, target_hurricane, reduction='mean'
        ) * self.class_weight[2]
        
        dice_hurricane = self.dice_loss(hurricane_probs, target_hurricane) * self.class_weight[2]
        
        # Pattern matching on strict mask (no re-validation needed)
        pattern_matching_loss = self.hurricane_eye_pattern_matching_loss(
            hurricane_logits, strict_eye_mask)
        
        # Intensity preservation
        intensity_preservation_loss = self.hurricane_eye_intensity_preservation_loss(
            hurricane_logits, strict_eye_mask)
        
        # Spatial consistency
        spatial_consistency_loss = self.hurricane_eye_spatial_consistency_loss(
            hurricane_logits, strict_eye_mask)
        
        # Positive eye detection
        positive_eye_loss = self.hurricane_positive_eye_detection_loss(
            hurricane_logits, strict_eye_mask)
        
        # Focal loss for hard examples
        positive_mask = (strict_eye_mask > 0).float()
        hurricane_focal_loss = self.focal_loss(hurricane_logits, positive_mask)
        
        # Combine losses - REMOVED redundant multi-criteria and component synergy
        hurricane_loss = (bce_hurricane + dice_hurricane + 
                        self.hurricane_pattern_weight * pattern_matching_loss + 
                        self.hurricane_intensity_weight * intensity_preservation_loss + 
                        self.hurricane_consistency_weight * spatial_consistency_loss + 
                        self.hurricane_positive_weight * positive_eye_loss + 
                        self.hurricane_focal_weight * hurricane_focal_loss)
        
        return hurricane_loss

    # @autocast(enabled=False)
    # def multi_criteria_etc_detection_loss_high_confidence(self, pred_logits, high_conf_mask, 
    #                                                     etc_pressure_mask, etc_vorticity_mask, etc_wind_mask):
    #     """Multi-criteria loss focused ONLY on high-confidence regions"""
    #     pred_probs = torch.sigmoid(pred_logits)
        
    #     # Primary loss: predict 1.0 in high-confidence regions only
    #     primary_loss = F.binary_cross_entropy_with_logits(pred_logits, high_conf_mask, reduction='mean')
        
    #     # Component losses ONLY where high-confidence
    #     if torch.sum(high_conf_mask) > 0:
    #         pressure_loss = F.mse_loss(
    #             pred_probs * high_conf_mask * etc_pressure_mask,
    #             high_conf_mask * etc_pressure_mask
    #         )
    #         vorticity_loss = F.mse_loss(
    #             pred_probs * high_conf_mask * etc_vorticity_mask,
    #             high_conf_mask * etc_vorticity_mask
    #         )
    #         wind_loss = F.mse_loss(
    #             pred_probs * high_conf_mask * etc_wind_mask,
    #             high_conf_mask * etc_wind_mask
    #         )
    #     else:
    #         pressure_loss = vorticity_loss = wind_loss = torch.tensor(0.0, device=pred_logits.device)
        
    #     return primary_loss + 0.3 * pressure_loss + 0.3 * vorticity_loss + 0.2 * wind_loss

    # @autocast(enabled=False)
    # def etc_spatial_consistency_loss_high_confidence(self, pred_logits, high_conf_mask):
    #     """Spatial consistency ONLY in high-confidence regions"""
    #     pred_probs = torch.sigmoid(pred_logits)
        
    #     if torch.sum(high_conf_mask) == 0:
    #         return torch.tensor(0.0, device=pred_logits.device)
        
    #     # Apply spatial smoothing only to high-confidence regions
    #     kernel_size = 3
    #     pred_masked = pred_probs * high_conf_mask
    #     target_masked = high_conf_mask
        
    #     pred_smooth = F.avg_pool2d(pred_masked.unsqueeze(1), kernel_size, stride=1, padding=kernel_size//2).squeeze(1)
    #     target_smooth = F.avg_pool2d(target_masked.unsqueeze(1), kernel_size, stride=1, padding=kernel_size//2).squeeze(1)
        
    #     return F.mse_loss(pred_smooth, target_smooth)

    # @autocast(enabled=False)  
    # def etc_component_synergy_loss_high_confidence(self, pred_logits, etc_pressure_mask, 
    #                                             etc_vorticity_mask, high_conf_mask):
    #     """Component synergy ONLY in high-confidence regions"""
    #     pred_probs = torch.sigmoid(pred_logits)
        
    #     if torch.sum(high_conf_mask) == 0:
    #         return torch.tensor(0.0, device=pred_logits.device)
        
    #     # Synergy calculation only in high-confidence regions
    #     pressure_vorticity_combo = etc_pressure_mask * etc_vorticity_mask
    #     masked_combo = pressure_vorticity_combo * high_conf_mask
    #     masked_pred = pred_probs * high_conf_mask
        
    #     return F.mse_loss(masked_pred * masked_combo, masked_combo)

    @autocast(enabled=False)
    def compute_etc_losses(self, etc_probs, target_etc, etc_pressure_mask, 
                        etc_vorticity_mask, etc_wind_mask, etc_adaptive_mask):
        """
        Compute ETC-specific losses - SIMPLIFIED using pre-processed adaptive mask
        """
        etc_logits = torch.logit(etc_probs.clamp(1e-7, 1-1e-7))
        
        # Check if any ETC regions exist
        if torch.sum(etc_adaptive_mask) == 0:
            return torch.tensor(0.001, device=etc_probs.device, requires_grad=True)
        
        # Primary loss - use adaptive mask directly (already processed in dataloader)
        primary_loss = F.binary_cross_entropy_with_logits(
            etc_logits, etc_adaptive_mask, reduction='mean'
        ) * self.class_weight[3]
        
        # Intensity preservation using adaptive mask
        intensity_preservation_loss = F.mse_loss(etc_probs, etc_adaptive_mask)
        
        # Spatial consistency
        spatial_consistency_loss = self.etc_spatial_consistency_loss(
            etc_logits, etc_adaptive_mask)
        
        # Positive detection
        positive_etc_loss = F.binary_cross_entropy_with_logits(
            etc_logits, (etc_adaptive_mask > 0).float(), reduction='mean')
        
        # Focal loss
        etc_focal_loss = self.focal_loss(etc_logits, (etc_adaptive_mask > 0).float())
        
        # Combine losses - REMOVED redundant multi-criteria and component synergy
        etc_loss = (primary_loss + 
                    self.etc_intensity_weight * intensity_preservation_loss + 
                    self.etc_consistency_weight * spatial_consistency_loss + 
                    self.etc_positive_weight * positive_etc_loss + 
                    self.etc_focal_weight * etc_focal_loss)
        
        return etc_loss

    @autocast(enabled=False)
    def compute_mcs_losses(self, mcs_probs, target_mcs, mcs_boundary_mask, 
                        mcs_core_pixels_mask, mcs_blobs_mask):
        """
        Compute MCS-specific losses - SIMPLIFIED using pre-processed blob masks
        """
        mcs_logits = torch.logit(mcs_probs.clamp(1e-7, 1-1e-7))
        
        # Basic MCS losses using blob mask (already validated in dataloader)
        bce_mcs = F.binary_cross_entropy_with_logits(
            mcs_logits, target_mcs, reduction='mean'
        ) * self.class_weight[4]
        
        dice_mcs = self.dice_loss(mcs_probs, target_mcs) * self.class_weight[4]
        
        # Pattern matching for blobs
        pattern_matching_loss = self.mcs_boundary_pattern_loss(
            mcs_logits, mcs_boundary_mask, mcs_blobs_mask)
        
        # Intensity preservation
        intensity_preservation_loss = self.mcs_intensity_preservation_loss(
            mcs_logits, mcs_blobs_mask)
        
        # Spatial consistency
        spatial_consistency_loss = self.mcs_spatial_consistency_loss(
            mcs_logits, mcs_blobs_mask)
        
        # Positive MCS detection
        positive_mcs_loss = self.mcs_positive_detection_loss(
            mcs_logits, mcs_blobs_mask)
        
        # Focal loss
        positive_mask = (mcs_blobs_mask > 0).float()
        mcs_focal_loss = self.focal_loss(mcs_logits, positive_mask)
        
        # Combine losses - REMOVED redundant multi-criteria and synergy
        mcs_loss = (bce_mcs + dice_mcs + 
                    self.mcs_pattern_weight * pattern_matching_loss + 
                    self.mcs_intensity_weight * intensity_preservation_loss + 
                    self.mcs_consistency_weight * spatial_consistency_loss + 
                    self.mcs_positive_weight * positive_mcs_loss + 
                    self.mcs_focal_weight * mcs_focal_loss)
        
        return mcs_loss

    def forward(self, seg_probs, inputs, target=None, batch_idx=0):
        """
        Forward pass with memory optimization and autocast compatibility
        Now handles 5 classes: background, AR, Hurricane, ETC, MCS
        Returns dictionary with individual and total losses for tracking
        """

        # Extract probability channels (now expecting 4 channels)
        ar_probs = seg_probs[:, 1]  # AR class T
        hurricane_probs = seg_probs[:, 2]  # Hurricane class T
        etc_probs = seg_probs[:, 3]  # ETC class (new) T
        mcs_probs = seg_probs[:, 4]  # MCS class (new) T
        
        # Extract input channels
        ivt_channel = inputs[:, 0]  # IVT T
        ivt_binary = inputs[:, 1]  # Binary IVT mask (AR detection) T
        
        # Extract hurricane detection channels
        pressure_mask = inputs[:, 2]  # Low pressure mask
        vorticity_mask = inputs[:, 3]  # High vorticity mask  
        wind_speed_mask = inputs[:, 4]  # High wind speed mask
        strict_hurricane_eye_mask = inputs[:, 5]  # STRICT hurricane eye mask
        wind_speed_binary = inputs[:, 6]  # Wind speed binary mask
        
        # Extract ETC detection channels (new)
        etc_pressure_mask = inputs[:, 7]  # ETC low pressure mask
        # etc_cold_core_mask = inputs[:, 13]  # ETC cold core mask
        etc_vorticity_mask = inputs[:, 8]  # ETC vorticity mask
        etc_wind_mask = inputs[:, 9]  # ETC wind speed mask
        etc_adaptive_mask = inputs[:, 10]  # ETC strict detection mask
        
        
        # Extract MCS detection channels (new)
        # # Note: Based on your channel layout from the dataloader
        # mcs_convection_mask = inputs[:, 11]  # MCS deep convection mask
        # mcs_precip_mask = inputs[:, 12]  # MCS heavy precipitation mask
        # # mcs_wind_mask = inputs[:, 21]  # MCS wind speed mask
        # mcs_core_mask = inputs[:, 13]  # MCS core detection mask


        # Extract MCS detection channels (updated for two-threshold approach)
        mcs_convection_mask = inputs[:, 11]  # MCS deep convection mask
        mcs_precip_mask = inputs[:, 12]  # MCS heavy precipitation mask
        mcs_core_mask = inputs[:, 13]  # MCS core detection mask
    
        # Generate targets
        if target is None:
            target_ar = ivt_binary
            target_hurricane = strict_hurricane_eye_mask
            target_etc = etc_adaptive_mask  # New ETC target
            target_mcs = mcs_core_mask  # New MCS target
        else:
            target_ar = target[:, 1]
            target_hurricane = target[:, 2]
            target_etc = target[:, 3]  # New ETC target
            target_mcs = target[:, 4]  # New MCS target
        
        # Compute AR losses
        ar_loss = self.compute_ar_losses(ar_probs, target_ar, ivt_channel, ivt_binary)
        
        # Compute hurricane losses
        hurricane_loss = self.compute_hurricane_losses(
            hurricane_probs, target_hurricane, wind_speed_mask, vorticity_mask, 
            pressure_mask, strict_hurricane_eye_mask, wind_speed_binary
        )
        
        # Compute ETC losses (new)
        etc_loss = self.compute_etc_losses(
            etc_probs, target_etc, etc_pressure_mask,
            etc_vorticity_mask, etc_wind_mask, etc_adaptive_mask
        )
        
        # Compute MCS losses (updated for two-threshold approach)
        mcs_loss = self.compute_mcs_losses(
            mcs_probs, target_mcs, mcs_convection_mask, mcs_precip_mask, mcs_core_mask)
        
        # # Compute MCS losses (new)
        # mcs_loss = self.compute_mcs_losses(
        #     mcs_probs, target_mcs, mcs_convection_mask, mcs_precip_mask, mcs_core_mask)
        
        # Prevent overlaps between storm types (expanded for MCS)
        ar_hurricane_overlap = (ar_probs * hurricane_probs).mean()
        ar_etc_overlap = (ar_probs * etc_probs).mean()
        ar_mcs_overlap = (ar_probs * mcs_probs).mean()  # New
        hurricane_etc_overlap = (hurricane_probs * etc_probs).mean()
        hurricane_mcs_overlap = (hurricane_probs * mcs_probs).mean()  # New
        etc_mcs_overlap = (etc_probs * mcs_probs).mean()  # New
        
        total_overlap_loss = (ar_hurricane_overlap + ar_etc_overlap + ar_mcs_overlap + 
                            hurricane_etc_overlap + hurricane_mcs_overlap + etc_mcs_overlap)
        
        # Total loss (now includes MCS)
        total_loss = ar_loss + hurricane_loss + etc_loss + mcs_loss + total_overlap_loss
        
        # Return dictionary with individual losses for tracking
        return {
            'total_loss': total_loss,
            'ar_loss': ar_loss.detach(),
            'hurricane_loss': hurricane_loss.detach(),
            'etc_loss': etc_loss.detach(),
            'mcs_loss': mcs_loss.detach(),
            'overlap_loss': total_overlap_loss.detach()
        }
    



# Memory-efficient utility functions
def clear_gpu_cache():
    """Clear GPU cache to free memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_gpu_memory_info():
    """Get GPU memory information"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        return allocated, reserved
    return 0, 0


def optimize_batch_size_for_gpu(model, sample_input, max_batch_size=32):
    """
    Automatically find optimal batch size for GPU memory
    """
    model.eval()
    
    for batch_size in range(max_batch_size, 0, -1):
        try:
            # Create test batch
            test_input = sample_input[:batch_size] if sample_input.size(0) >= batch_size else sample_input
            
            # Test forward pass
            with torch.no_grad():
                _ = model(test_input)
            
            print(f"Optimal batch size found: {batch_size}")
            model.train()
            return batch_size
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                clear_gpu_cache()
                continue
            else:
                raise e
    
    model.train()
    return 1  # Fallback to batch size 1

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1 — loss for the split (predictors / labels) contract
# ═══════════════════════════════════════════════════════════════════════════════

class StormSegLoss(nn.Module):
    """Segmentation loss that reads its targets from a SEPARATE label tensor.

    ARSegmentationLoss pulled its targets out of the input tensor itself
    (loss.py:1043-1047), which is what let the model satisfy it by copying a
    channel through. This one takes `labels` explicitly, so it cannot be
    satisfied that way — the model only ever sees physical fields.

    Deliberately minimal. The old per-class stack of seven terms (bce, dice,
    pattern, intensity, consistency, positive, focal) all scored the SAME mask,
    so it was one loss with an opaque effective weight. Here each class gets
    BCE (imbalance-corrected) + Dice, and the inter-class overlap penalty is
    kept so that Phase 1 changes only the input contract, nothing else. Choosing
    between softmax/mutual-exclusivity and independent sigmoid heads is Phase 2.

    Args:
        class_weight     : per-class scale, ordered (AR, Hurricane, ETC, MCS).
        target_threshold : labels are stored as floats (ETC carries a graded
                           intensity); this binarises them for the loss.
        use_pos_weight   : counter the extreme foreground imbalance (a hurricane
                           eye is ~0.3% of the globe) with a per-batch
                           negative/positive ratio in the BCE.
        max_pos_weight   : clamp on that ratio, so an almost-empty batch cannot
                           produce an enormous gradient.
    """

    def __init__(self,
                 class_weight=(10.0, 15.0, 12.0, 10.0),
                 target_threshold: float = 0.5,
                 dice_weight: float = 1.0,
                 overlap_weight: float = 1.0,
                 use_pos_weight: bool = True,
                 max_pos_weight: float = 50.0):
        super().__init__()
        self.class_weight = list(class_weight)
        self.target_threshold = target_threshold
        self.dice_weight = dice_weight
        self.overlap_weight = overlap_weight
        self.use_pos_weight = use_pos_weight
        self.max_pos_weight = max_pos_weight
        self.class_names = ["ar", "hurricane", "etc", "mcs"]

    @staticmethod
    def _dice(probs, target, smooth=1e-6):
        num = 2.0 * (probs * target).sum() + smooth
        den = probs.sum() + target.sum() + smooth
        return 1.0 - num / den

    def forward(self, seg_probs, labels, inputs=None, batch_idx=0):
        """
        seg_probs : (B, 5, H, W) softmax over {background, AR, Hurricane, ETC, MCS}
        labels    : (B, 4, H, W) heuristic masks, ordered (AR, Hurricane, ETC, MCS)
        inputs    : accepted and ignored — kept so callers can pass predictors
                    without the loss ever being able to read a target from them.
        """
        if labels.shape[1] != 4:
            raise ValueError(f"expected 4 label channels, got {labels.shape[1]}")
        if seg_probs.shape[1] != 5:
            raise ValueError(f"expected 5 segmentation classes, got {seg_probs.shape[1]}")

        # logit() and the ratio arithmetic are unstable in fp16 — force fp32.
        seg_probs = seg_probs.float()
        labels = labels.float()

        per_class = {}
        total = seg_probs.new_zeros(())

        for i, name in enumerate(self.class_names):
            probs = seg_probs[:, i + 1]
            target = (labels[:, i] > self.target_threshold).float()
            logits = torch.logit(probs.clamp(1e-7, 1 - 1e-7))

            pos_weight = None
            if self.use_pos_weight:
                n_pos = target.sum()
                n_neg = target.numel() - n_pos
                if n_pos > 0:
                    pos_weight = torch.clamp(n_neg / n_pos, min=1.0,
                                             max=self.max_pos_weight)

            bce = F.binary_cross_entropy_with_logits(
                logits, target, pos_weight=pos_weight, reduction="mean")
            dice = self._dice(probs, target)

            loss_c = self.class_weight[i] * (bce + self.dice_weight * dice)
            per_class[name] = loss_c
            total = total + loss_c

        # Keep the classes from claiming the same pixels (softmax already makes
        # argmax exclusive, but the probabilities themselves can co-fire).
        overlap = seg_probs.new_zeros(())
        for a in range(1, 5):
            for b in range(a + 1, 5):
                overlap = overlap + (seg_probs[:, a] * seg_probs[:, b]).mean()
        total = total + self.overlap_weight * overlap

        return {
            "total_loss":     total,
            "ar_loss":        per_class["ar"].detach(),
            "hurricane_loss": per_class["hurricane"].detach(),
            "etc_loss":       per_class["etc"].detach(),
            "mcs_loss":       per_class["mcs"].detach(),
            "overlap_loss":   overlap.detach(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2 — multi-label loss + a corrected N-cut regulariser
# ═══════════════════════════════════════════════════════════════════════════════

class NCutLoss2DFixed(nn.Module):
    """Soft N-cut regulariser over a MULTI-CHANNEL feature space.

    Two corrections to NCutLoss2D, which was defined but never called anywhere
    in training:

    1. It collapsed the inputs with `inputs.mean(dim=1)` before measuring
       similarity. Averaging six standardised but physically unrelated fields
       (IVT, MSLP, vorticity, wind speed, BT, IWV) into one scalar destroys the
       feature space the cut is supposed to respect — a cold moist pixel and a
       warm dry one can average to the same number. Here the squared distance is
       summed across channels, i.e. the proper multivariate form. The original
       code already summed over dim=1, so it was structurally ready for this;
       only the pre-averaging was in the way.

    2. Its weights were `exp(-(d^2)^2 / sigma^2)` because `diff` was already a
       squared distance and was then squared again. Soft N-cut wants
       `exp(-d^2 / sigma^2)`. The extra power made the affinity collapse to zero
       for all but near-identical pixels, so the term carried almost no signal.

    This is a per-class association term (class-mean form, not full pairwise), so
    it composes with independent sigmoid heads — it does not require the class
    probabilities to form a partition.
    """

    def __init__(self, radius: int = 4, sigma_1: float = 4.0, sigma_2: float = 2.0):
        super().__init__()
        self.radius = radius
        self.sigma_1 = sigma_1     # spatial std of the neighbourhood kernel
        self.sigma_2 = sigma_2     # feature-space std
        self.kernel = None

    def forward(self, probs: Tensor, inputs: Tensor) -> Tensor:
        """probs: (B, C, H, W) per-class probabilities. inputs: (B, C_in, H, W)."""
        probs = probs.float()
        inputs = inputs.float()
        n_classes = probs.shape[1]

        if (self.kernel is None or self.kernel.device != probs.device
                or self.kernel.dtype != probs.dtype):
            self.kernel = gaussian_kernel(radius=self.radius, sigma=self.sigma_1,
                                          device=probs.device).to(probs.dtype)

        total = probs.new_zeros(())
        for k in range(n_classes):
            p = probs[:, k].unsqueeze(1)                       # (B,1,H,W)
            denom_mean = p.mean(dim=(2, 3), keepdim=True) + 1e-5
            # per-channel weighted mean of the inputs inside class k
            class_mean = (inputs * p).mean(dim=(2, 3), keepdim=True) / denom_mean
            # squared distance in the FULL feature space
            d2 = (inputs - class_mean).pow(2).sum(dim=1, keepdim=True)
            w = torch.exp(-d2 / (self.sigma_2 ** 2))
            num = torch.sum(p * F.conv2d(p * w, self.kernel, padding=self.radius))
            den = torch.sum(p * F.conv2d(w, self.kernel, padding=self.radius)) + 1e-6
            total = total + num / den                          # association in [0,1]

        # maximise association => minimise (n_classes - association)
        return n_classes - total


class StormSegLossV2(nn.Module):
    """Phase 2 loss: four independent binary heads, no mutual exclusivity.

    Per class:  BCEWithLogits(pos_weight) + Tversky.

    Tversky replaces Dice because Dice weights false positives and false
    negatives equally, which is wrong for targets this rare — a hurricane eye is
    ~0.3% of the globe, so predicting nothing scores well. beta > alpha penalises
    missed positives harder and pushes recall up.

    Gone from StormSegLoss / ARSegmentationLoss:
      * the softmax and the pairwise overlap penalty — ARs and ETCs co-occur, and
        penalising that was penalising the correct answer
      * the logit(clamp(probs)) round-trip — this takes logits directly

    N-cut and reconstruction are added by the training loop, not here, so their
    weights can be scheduled independently.

    Args:
        class_weight   : per-class scale, ordered (AR, Hurricane, ETC, MCS)
        tversky_alpha  : false-positive penalty
        tversky_beta   : false-negative penalty (> alpha for rare classes)
        max_pos_weight : cap on the per-batch neg/pos ratio in the BCE, so a
                         nearly empty batch cannot produce a huge gradient
    """

    def __init__(self,
                 class_weight=(1.0, 3.0, 2.0, 1.5),
                 target_threshold: float = 0.5,
                 tversky_alpha=0.3,
                 tversky_beta=0.7,
                 tversky_weight: float = 1.0,
                 use_pos_weight: bool = True,
                 max_pos_weight=50.0,
                 valid_mask=None):
        super().__init__()
        # valid_mask: None (default, unchanged behaviour) or a (H,W) / (C,H,W)
        # boolean-ish tensor. Pixels outside it are excluded from BOTH the BCE mean
        # and the Tversky sums. RegCM needs this because ~29% of every frame is
        # outside the model domain, and the Hurricane head additionally has no label
        # west of -105 deg (TempestExtremes never searched there) — undefined, not
        # negative. Scoring those pixels as negatives would teach the model that
        # visible storms are not storms.
        self._valid_mask = None
        if valid_mask is not None:
            import torch as _t
            vm = _t.as_tensor(valid_mask).float()
            if vm.dim() == 2:
                vm = vm.unsqueeze(0)
            self.register_buffer("_vm", vm, persistent=False)
            self._valid_mask = True
        self.class_names = ["ar", "hurricane", "etc", "mcs"]
        n = len(self.class_names)

        def _per_class(v, what):
            """Accept a scalar (shared by every class) or one value per class."""
            if isinstance(v, (int, float)):
                return [float(v)] * n
            v = list(v)
            if len(v) != n:
                raise ValueError(f"{what} needs 1 or {n} values, got {len(v)}")
            return [float(x) for x in v]

        self.class_weight = _per_class(class_weight, "class_weight")
        self.target_threshold = target_threshold
        # Per-class alpha/beta/pos_weight, because the base rates span an order
        # of magnitude: AR covers ~6.8% of the globe, a hurricane eye ~0.3%.
        # The Phase 2 pilot showed a single recall push that suits hurricanes
        # drives AR to recall 0.99 at precision 0.55 — and for a class whose
        # optimal decision is a crisp threshold, that over-prediction costs IoU
        # directly (AR landed below its own single-threshold baseline).
        self.alpha = _per_class(tversky_alpha, "tversky_alpha")
        self.beta = _per_class(tversky_beta, "tversky_beta")
        self.max_pos_weight = _per_class(max_pos_weight, "max_pos_weight")
        self.tversky_weight = tversky_weight
        self.use_pos_weight = use_pos_weight

    def _tversky(self, probs, target, alpha, beta, smooth=1e-6, w=None):
        if w is not None:
            probs = probs * w
            target = target * w
        tp = (probs * target).sum()
        fp = (probs * (1 - target)).sum()
        fn = ((1 - probs) * target).sum()
        ti = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
        return 1.0 - ti

    def forward(self, logits, labels):
        """logits: (B,4,H,W) pre-sigmoid. labels: (B,4,H,W) heuristic masks."""
        if logits.shape[1] != 4 or labels.shape[1] != 4:
            raise ValueError(f"expected 4 channels, got logits {logits.shape[1]} "
                             f"labels {labels.shape[1]}")
        logits = logits.float()
        labels = labels.float()

        out = {}
        total = logits.new_zeros(())
        for i, name in enumerate(self.class_names):
            lg = logits[:, i]
            target = (labels[:, i] > self.target_threshold).float()

            pos_weight = None
            if self.use_pos_weight:
                if self._valid_mask is None:
                    n_pos = target.sum()
                    n_tot = target.numel()
                else:
                    _w = (self._vm[i] if self._vm.shape[0] > 1 else self._vm[0])
                    _w = _w.to(target.device, target.dtype)
                    n_pos = (target * _w).sum()
                    n_tot = _w.sum() * target.shape[0]
                if n_pos > 0:
                    n_neg = n_tot - n_pos
                    pos_weight = torch.clamp(n_neg / n_pos, min=1.0,
                                             max=self.max_pos_weight[i])

            if self._valid_mask is None:
                bce = F.binary_cross_entropy_with_logits(
                    lg, target, pos_weight=pos_weight, reduction="mean")
                tv = self._tversky(torch.sigmoid(lg), target,
                                   self.alpha[i], self.beta[i])
            else:
                w = self._vm[i] if self._vm.shape[0] > 1 else self._vm[0]
                w = w.to(lg.device, lg.dtype)
                bce_px = F.binary_cross_entropy_with_logits(
                    lg, target, pos_weight=pos_weight, reduction="none")
                bce = (bce_px * w).sum() / w.sum().clamp(min=1.0) / max(lg.shape[0], 1)
                tv = self._tversky(torch.sigmoid(lg), target,
                                   self.alpha[i], self.beta[i], w=w)

            loss_c = self.class_weight[i] * (bce + self.tversky_weight * tv)
            out[f"{name}_loss"] = loss_c.detach()
            total = total + loss_c

        out["total_loss"] = total
        return out

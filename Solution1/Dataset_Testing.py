# Testing.py
# Testing script for satellite change detection using trained model
# Uses the same preprocessing pipeline as mock_train_improved.py
# Required packages:
# pip install torch rasterio geopandas albumentations numpy scikit-image scipy shapely

import os
import zipfile
import tempfile
from pathlib import Path
import logging
import re
import shutil
import multiprocessing

# Fix for PROJ library on Windows - must be done before importing rasterio/geopandas
try:
    import pyproj
    os.environ['PROJ_LIB'] = pyproj.datadir.get_data_dir()
except Exception:
    pass

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
import albumentations as A

from scipy.ndimage import binary_closing, binary_opening, binary_erosion, binary_dilation
from scipy.ndimage import label
from skimage.filters import gaussian, median, sobel
from skimage.morphology import disk, square, remove_small_objects as sk_remove_small_objects
from skimage.measure import regionprops
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

# --------------------------------------------------------------------------
# CONFIGURATION (Same as training script)
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
TEST_DATA_FOLDER = str("D:\Solution1\Solution1\data\raw")
TRAINED_MODEL_PATH = str("D:\Solution1\Solution1\saved_model.pth")
TEST_OUTPUT_FOLDER = str("D:\Solution1\Solution1\data\output")
TEST_PREPROCESSED_FOLDER = str("D:\Solution1\Solution1\data\preprocess")

BATCH_SIZE = 1  # Use batch size 1 for testing
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SIZE = (512, 512)  # H,W - used for patch size
NUM_WORKERS = min(8, max(0, multiprocessing.cpu_count() - 1))

# Patch-wise inference parameters
PATCH_SIZE = 512  # Size of each patch for inference
PATCH_OVERLAP = 64  # Overlap between patches to avoid edge artifacts
MIN_PATCH_SIZE = 256  # Minimum patch size for very small images

FORCED_CRS = "EPSG:32643"
INVERT_TRAIN_MASK = False  # False: white = change, black = no-change

# Enhanced parameters for better man-made change detection
ENHANCEMENT_PARAMS = {
    'histogram_equalization': True,
    'contrast_stretching': True,
    'noise_reduction': True,
    'edge_enhancement': True,
    'texture_analysis': True
}

POSTPROCESSING_PARAMS = {
    'min_object_size': 50,
    'morphology_kernel_size': 2,
    'gaussian_sigma': 0.8,
    'median_filter_size': 2,
    'opening_iterations': 1,
    'closing_iterations': 2,
    'edge_preservation': True
}

# Inference parameters
INFERENCE_PARAMS = {
    'use_patch_wise': True,  # Enable patch-wise inference
    'adaptive_threshold': True,  # Use adaptive thresholding
    'threshold_percentile': 85,  # Percentile for threshold calculation
    'min_threshold': 0.1,  # Minimum threshold value
    'save_probability_map': False,  # Save probability maps for debugging
}

# Preprocessing parameters
PREPROCESSING_PARAMS = {
    'skip_if_exists': True,  # Skip preprocessing if data already exists
    'force_reprocess': False,  # Force reprocessing even if data exists
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --------------------------------------------------------------------------
# UTILITY FUNCTIONS (Same as training script)
# --------------------------------------------------------------------------
def unzip_to(zip_path, extract_to):
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_to)
        return True
    except Exception as e:
        logging.error(f"Failed to extract {zip_path}: {e}")
        return False

def get_band_paths(folder):
    """Detect band paths in a folder. Supports Sentinel-2 naming and generic naming."""
    folder = Path(folder)
    jp2_files = list(folder.rglob("*.jp2"))
    tif_files = list(folder.rglob("*.tif"))
    bands = {"red": None, "green": None, "blue": None, "nir": None}

    # Helper to check if a file is likely a specific band
    def is_band(name, pattern_list):
        return any(p in name.lower() for p in pattern_list)

    if jp2_files:
        jp2_10m = [f for f in jp2_files if any(p in str(f).lower() for p in ["r10m", "10m"])]
        candidates = jp2_10m if jp2_10m else jp2_files
        for f in candidates:
            n = f.name.lower()
            if is_band(n, ["b02", "band2"]): bands["blue"] = f
            elif is_band(n, ["b03", "band3"]): bands["green"] = f
            elif is_band(n, ["b04", "band4"]): bands["red"] = f
            elif is_band(n, ["b08", "band8"]): bands["nir"] = f
        
        if all(bands.values()): return bands

    if tif_files:
        # First try band-specific patterns
        for f in tif_files:
            n = f.name.lower()
            if is_band(n, ["b02", "band2"]): bands["blue"] = f
            elif is_band(n, ["b03", "band3"]): bands["green"] = f
            elif is_band(n, ["b04", "band4"]): bands["red"] = f
            elif is_band(n, ["b08", "band8"]): bands["nir"] = f
        
        if bands["red"] and bands["green"] and bands["blue"]:
            return bands

        # Fallback: If no band patterns match, look for any TIFF (multi-band or single-band)
        for f in tif_files:
            try:
                with rasterio.open(f) as src:
                    logging.info(f"Using provided image: {f.name} ({src.count} bands)")
                    return {"stacked": f}
            except Exception:
                continue

    logging.error(f"No recognizable satellite bands or multi-band images found in {folder}")
    return None

def enhance_image_quality(image):
    """Apply advanced image enhancement for better change detection."""
    enhanced = image.copy().astype(np.float32)
    
    # 1. Histogram equalization for each band
    if ENHANCEMENT_PARAMS['histogram_equalization']:
        for i in range(enhanced.shape[0]):
            band = enhanced[i]
            # Normalize to 0-1
            band_norm = (band - band.min()) / (band.max() - band.min() + 1e-8)
            # Apply histogram equalization
            hist, bins = np.histogram(band_norm, bins=256, range=(0, 1))
            cdf = hist.cumsum()
            cdf_normalized = cdf / cdf.max()
            enhanced[i] = np.interp(band_norm, bins[:-1], cdf_normalized)
    
    # 2. Contrast stretching
    if ENHANCEMENT_PARAMS['contrast_stretching']:
        for i in range(enhanced.shape[0]):
            band = enhanced[i]
            p2, p98 = np.percentile(band, (2, 98))
            enhanced[i] = np.clip((band - p2) / (p98 - p2 + 1e-8), 0, 1)
    
    # 3. Noise reduction
    if ENHANCEMENT_PARAMS['noise_reduction']:
        for i in range(enhanced.shape[0]):
            enhanced[i] = median(enhanced[i], disk(POSTPROCESSING_PARAMS['median_filter_size']))
    
    # 4. Edge enhancement
    if ENHANCEMENT_PARAMS['edge_enhancement']:
        for i in range(enhanced.shape[0]):
            # Apply Gaussian blur and subtract from original for edge enhancement
            blurred = gaussian(enhanced[i], sigma=POSTPROCESSING_PARAMS['gaussian_sigma'])
            enhanced[i] = enhanced[i] + 0.3 * (enhanced[i] - blurred)
            enhanced[i] = np.clip(enhanced[i], 0, 1)
    
    return enhanced

def stack_bands(band_map, output_path):
    ref_band = band_map.get("red") or band_map.get("green")
    if ref_band is None:
        logging.error("No reference band for stacking.")
        return None
    with rasterio.open(ref_band) as ref:
        meta = ref.meta.copy()
        # Always output 3 bands for the Siamese model
        meta.update(driver="GTiff", count=3, dtype='float32', compress="lzw", crs=FORCED_CRS)
        
        with rasterio.open(output_path, "w", **meta) as dst:
            # If the source is already a 3+ band "stacked" image, use its first 3 bands
            if "stacked" in band_map:
                with rasterio.open(band_map["stacked"]) as src:
                    for i in range(1, 4):
                        # Use band 1 if the source has fewer than 3 bands
                        src_idx = i if src.count >= i else 1
                        data = np.empty((ref.height, ref.width), dtype=np.float32)
                        reproject(
                            source=rasterio.band(src, src_idx),
                            destination=data,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=ref.transform,
                            dst_crs=FORCED_CRS,
                            resampling=Resampling.bilinear
                        )
                        dst.write(data, i)
            else:
                # Handle individual band map
                for idx, bname in enumerate(["red", "green", "blue"], start=1):
                    path = band_map.get(bname)
                    if path is None:
                        # Re-use another band if this one is missing
                        path = band_map.get("red") or band_map.get("green") or band_map.get("blue") or band_map.get("nir")
                    
                    if path is None:
                        dst.write(np.zeros((ref.height, ref.width), dtype=np.float32), idx)
                        continue
                        
                    with rasterio.open(path) as src:
                        data = np.empty((ref.height, ref.width), dtype=np.float32)
                        reproject(
                            source=rasterio.band(src, 1),
                            destination=data,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=ref.transform,
                            dst_crs=FORCED_CRS,
                            resampling=Resampling.bilinear
                        )
                        dst.write(data, idx)
    return output_path

def align_rasters(reference_path, to_align_path, out_aligned_path):
    with rasterio.open(reference_path) as ref, rasterio.open(to_align_path) as src:
        # Check for geographic overlap
        from rasterio.warp import transform_bounds
        src_bounds_ref_crs = transform_bounds(src.crs, ref.crs, *src.bounds)
        
        # Simple overlap check
        overlap = not (src_bounds_ref_crs[2] < ref.bounds[0] or 
                      src_bounds_ref_crs[0] > ref.bounds[2] or 
                      src_bounds_ref_crs[3] < ref.bounds[1] or 
                      src_bounds_ref_crs[1] > ref.bounds[3])
        
        if not overlap:
            logging.warning(f"CRITICAL: Images do not overlap! Analysis will likely be blank. Ref: {ref.bounds}, Src(in Ref CRS): {src_bounds_ref_crs}")

        dest_array = np.zeros((ref.count, ref.height, ref.width), dtype=ref.read(1).dtype)
        for i in range(1, ref.count + 1):
            # If src has fewer bands than ref, just use band 1
            src_band_idx = i if src.count >= i else 1
            reproject(
                source=rasterio.band(src, src_band_idx),
                destination=dest_array[i-1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref.transform,
                dst_crs=ref.crs,
                resampling=Resampling.nearest
            )
        
        meta = ref.meta.copy()
        meta.update(count=ref.count, dtype=dest_array.dtype, crs=ref.crs)
        with rasterio.open(out_aligned_path, "w", **meta) as dst:
            dst.write(dest_array)
    return out_aligned_path

def resize_tensor(tensor, size=TARGET_SIZE):
    return F.interpolate(tensor.unsqueeze(0), size=size, mode='bilinear', align_corners=False).squeeze(0)

def get_patch_coordinates(image_height, image_width, patch_size=PATCH_SIZE, overlap=PATCH_OVERLAP):
    """Generate patch coordinates for patch-wise inference."""
    patches = []
    step = patch_size - overlap
    
    # Calculate number of patches needed
    num_patches_h = max(1, (image_height - overlap) // step)
    num_patches_w = max(1, (image_width - overlap) // step)
    
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            # Calculate patch boundaries
            y_start = i * step
            x_start = j * step
            y_end = min(y_start + patch_size, image_height)
            x_end = min(x_start + patch_size, image_width)
            
            # Adjust for last patches to ensure full coverage
            if i == num_patches_h - 1:
                y_start = max(0, image_height - patch_size)
                y_end = image_height
            if j == num_patches_w - 1:
                x_start = max(0, image_width - patch_size)
                x_end = image_width
            
            patches.append((y_start, y_end, x_start, x_end))
    
    return patches

def pad_image_to_patch_size(image, patch_size=PATCH_SIZE):
    """Pad image to ensure it can be divided into patches properly."""
    h, w = image.shape[-2:]
    
    # Calculate padding needed
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    
    if pad_h == 0 and pad_w == 0:
        return image, (0, 0, 0, 0)
    
    # Pad symmetrically
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    # Apply padding
    if len(image.shape) == 3:  # (C, H, W)
        padded = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')
    else:  # (H, W)
        padded = F.pad(image.unsqueeze(0), (pad_left, pad_right, pad_top, pad_bottom), mode='reflect').squeeze(0)
    
    return padded, (pad_top, pad_bottom, pad_left, pad_right)

def remove_padding(prediction, original_shape, padding):
    """Remove padding from prediction to match original image size."""
    pad_top, pad_bottom, pad_left, pad_right = padding
    
    if pad_top == 0 and pad_bottom == 0 and pad_left == 0 and pad_right == 0:
        return prediction
    
    h, w = original_shape[-2:]
    return prediction[pad_top:pad_top+h, pad_left:pad_left+w]

def patch_wise_inference(model, image1, image2, patch_size=PATCH_SIZE, overlap=PATCH_OVERLAP):
    """Perform patch-wise inference on large images."""
    model.eval()
    
    # Get original dimensions
    original_shape = image1.shape
    h, w = image1.shape[-2:]
    
    logging.info(f"Original image size: {h}x{w}")
    
    # Handle very small images
    if h < MIN_PATCH_SIZE or w < MIN_PATCH_SIZE:
        logging.info("Image too small for patch-wise inference, using full image")
        with torch.no_grad():
            # Resize to patch size for inference
            img1_resized = F.interpolate(image1.unsqueeze(0), size=(patch_size, patch_size), 
                                           mode='bilinear', align_corners=False).squeeze(0)
            img2_resized = F.interpolate(image2.unsqueeze(0), size=(patch_size, patch_size), 
                                           mode='bilinear', align_corners=False).squeeze(0)
            
            logits = model(img1_resized.unsqueeze(0), img2_resized.unsqueeze(0))
            prob = torch.sigmoid(logits)
            
            # Resize back to original size
            prob_resized = F.interpolate(prob, size=(h, w), mode='bilinear', align_corners=False)
            return prob_resized.squeeze(0).squeeze(0)
    
    # Pad images to ensure proper patch coverage
    img1_padded, padding = pad_image_to_patch_size(image1, patch_size)
    img2_padded, _ = pad_image_to_patch_size(image2, patch_size)
    
    padded_h, padded_w = img1_padded.shape[-2:]
    logging.info(f"Padded image size: {padded_h}x{padded_w}")
    
    # Get patch coordinates
    patches = get_patch_coordinates(padded_h, padded_w, patch_size, overlap)
    logging.info(f"Processing {len(patches)} patches")
    
    # Initialize output probability map
    prob_map = torch.zeros((padded_h, padded_w), device=DEVICE)
    count_map = torch.zeros((padded_h, padded_w), device=DEVICE)
    
    # Process each patch
    with torch.no_grad():
        for i, (y_start, y_end, x_start, x_end) in enumerate(patches):
            logging.info(f"Processing patch {i+1}/{len(patches)}: ({y_start}:{y_end}, {x_start}:{x_end})")
            
            # Extract patches
            patch1 = img1_padded[:, y_start:y_end, x_start:x_end]
            patch2 = img2_padded[:, y_start:y_end, x_start:x_end]
            
            # Ensure patch is the right size
            if patch1.shape[-2:] != (patch_size, patch_size):
                patch1 = F.interpolate(patch1.unsqueeze(0), size=(patch_size, patch_size), 
                                       mode='bilinear', align_corners=False).squeeze(0)
                patch2 = F.interpolate(patch2.unsqueeze(0), size=(patch_size, patch_size), 
                                       mode='bilinear', align_corners=False).squeeze(0)
            
            # Run inference on patch
            logits = model(patch1.unsqueeze(0), patch2.unsqueeze(0))
            prob_patch = torch.sigmoid(logits).squeeze(0).squeeze(0)
            
            # Resize patch prediction back to original patch size
            patch_h, patch_w = y_end - y_start, x_end - x_start
            if prob_patch.shape != (patch_h, patch_w):
                prob_patch = F.interpolate(prob_patch.unsqueeze(0).unsqueeze(0), 
                                             size=(patch_h, patch_w), 
                                             mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            
            # Add to probability map with overlap handling
            prob_map[y_start:y_end, x_start:x_end] += prob_patch
            count_map[y_start:y_end, x_start:x_end] += 1
    
    # Average overlapping regions
    prob_map = prob_map / (count_map + 1e-8)
    
    # Remove padding
    prob_map = remove_padding(prob_map, original_shape, padding)
    
    logging.info(f"Patch-wise inference completed. Final size: {prob_map.shape}")
    return prob_map

# --------------------------------------------------------------------------
# MODEL ARCHITECTURE (Same as Final_Training/final_training.py)
# --------------------------------------------------------------------------
class UNet(nn.Module):
    """Superior UNet architecture for change detection with proper skip connections."""
    def __init__(self, in_channels=6, out_channels=1):  # 6 channels for concatenated images (3+3)
        super(UNet, self).__init__()
        
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def upconv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.ReLU(inplace=True)
            )

        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)
        
        self.pool = nn.MaxPool2d(2)
        
        self.bottleneck = conv_block(512, 1024)
        
        self.upconv4 = upconv_block(1024, 512)
        self.decoder4 = conv_block(1024, 512)
        self.upconv3 = upconv_block(512, 256)
        self.decoder3 = conv_block(512, 256)
        self.upconv2 = upconv_block(256, 128)
        self.decoder2 = conv_block(256, 128)
        self.upconv1 = upconv_block(128, 64)
        self.decoder1 = conv_block(128, 64)
        
        self.conv_last = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        
        # Encoder
        e1 = self.encoder1(x)
        p1 = self.pool(e1)
        e2 = self.encoder2(p1)
        p2 = self.pool(e2)
        e3 = self.encoder3(p2)
        p3 = self.pool(e3)
        e4 = self.encoder4(p3)
        p4 = self.pool(e4)
        
        # Bottleneck
        b = self.bottleneck(p4)
        
        # Decoder
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.decoder4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.decoder3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.decoder2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.decoder1(d1)
        
        return self.conv_last(d1)

# --------------------------------------------------------------------------
# TESTING FUNCTIONS
# --------------------------------------------------------------------------
def process_test_pair(pair_zip_path, pair_index, output_root):
    """Process a single test pair - same preprocessing as training."""
    logging.info(f"--- Processing test pair {pair_zip_path.name} ---")
    t1_dir, t2_dir = Path(output_root)/"T1", Path(output_root)/"T2"
    t1_dir.mkdir(parents=True, exist_ok=True)
    t2_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        if not unzip_to(pair_zip_path, tmp):
            logging.error(f"Could not unzip {pair_zip_path}")
            return False
        
        nested_zips = sorted(tmp.glob("*.zip"))
        if len(nested_zips) < 2:
            logging.error(f"Missing nested zips in {pair_zip_path.name}")
            return False
        
        first_data_dir, second_data_dir = tmp/"first", tmp/"second"
        unzip_to(nested_zips[0], first_data_dir)
        unzip_to(nested_zips[1], second_data_dir)

        first_bands, second_bands = get_band_paths(first_data_dir), get_band_paths(second_data_dir)
        if not first_bands or not second_bands:
            return False

        # Support both individual bands and pre-stacked images
        if "stacked" in first_bands:
            tmp_old_image = first_bands["stacked"]
        else:
            tmp_old_image = stack_bands(first_bands, tmp/"old_image.tif")

        if "stacked" in second_bands:
            tmp_new_image = second_bands["stacked"]
        else:
            tmp_new_image = stack_bands(second_bands, tmp/"new_image.tif")
        if not tmp_old_image or not tmp_new_image:
            logging.error("Stacking failed")
            return False

        tmp_new_image_aligned = align_rasters(tmp_old_image, tmp_new_image, tmp/"new_image_aligned.tif")
        
        # Apply image enhancement
        with rasterio.open(tmp_old_image) as src:
            old_enhanced = enhance_image_quality(src.read())
            old_enhanced_path = tmp / "old_image_enhanced.tif"
            with rasterio.open(old_enhanced_path, "w", **src.meta) as dst:
                dst.write(old_enhanced)
        
        with rasterio.open(tmp_new_image_aligned) as src:
            new_enhanced = enhance_image_quality(src.read())
            new_enhanced_path = tmp / "new_image_enhanced.tif"
            with rasterio.open(new_enhanced_path, "w", **src.meta) as dst:
                dst.write(new_enhanced)

        # Copy processed images
        shutil.copy(tmp_old_image, t1_dir / f"old_image_{pair_index}.tif")
        shutil.copy(tmp_new_image_aligned, t2_dir / f"new_image_{pair_index}.tif")
        
        logging.info(f"Successfully preprocessed test pair {pair_index}")
        return True

def preprocess_test_data(test_data_folder, preprocessed_folder):
    """Preprocess test data using the same pipeline as training."""
    logging.info("===== STARTING TEST DATA PREPROCESSING =====")
    test_path, prep_path = Path(test_data_folder), Path(preprocessed_folder)
    (prep_path/"T1").mkdir(parents=True, exist_ok=True)
    (prep_path/"T2").mkdir(parents=True, exist_ok=True)
    
    zips = sorted(test_path.glob("Pair*.zip"))
    if not zips:
        logging.error(f"No 'Pair*.zip' files found in {test_data_folder}")
        return False
    
    # Check if preprocessing is already done
    if PREPROCESSING_PARAMS['skip_if_exists'] and not PREPROCESSING_PARAMS['force_reprocess']:
        t1_dir = prep_path / "T1"
        t2_dir = prep_path / "T2"
        
        if t1_dir.exists() and t2_dir.exists():
            t1_files = list(t1_dir.glob("old_image_*.tif"))
            t2_files = list(t2_dir.glob("new_image_*.tif"))
            
            if len(t1_files) == len(zips) and len(t2_files) == len(zips) and len(t1_files) > 0:
                logging.info(f"✅ Preprocessed data already exists with {len(t1_files)} pairs")
                logging.info("⏭  Skipping preprocessing step")
                return True
    
    processed_count = 0
    for i, zip_file in enumerate(zips):
        logging.info(f"🔄 Processing test pair: {zip_file.name}")
        if process_test_pair(zip_file, i + 1, prep_path):
            processed_count += 1
    
    logging.info(f"✅ Processed {processed_count} test pairs.")
    logging.info(f"===== TEST DATA PREPROCESSING COMPLETE: Data saved in {preprocessed_folder} =====")
    return processed_count > 0

def post_process_mask(mask):
    """Enhanced post-processing for better change detection results."""
    if mask.size == 0:
        return mask
    
    # Convert to boolean for morphological operations
    binary_mask = mask.astype(bool)
    
    # Remove small objects
    if POSTPROCESSING_PARAMS['min_object_size'] > 0:
        binary_mask = sk_remove_small_objects(binary_mask, 
                                              min_size=POSTPROCESSING_PARAMS['min_object_size'])
    
    # Morphological operations
    kernel_size = POSTPROCESSING_PARAMS['morphology_kernel_size']
    structure = disk(kernel_size) if kernel_size > 0 else np.ones((3,3), dtype=bool)
    
    # Opening to remove noise
    if POSTPROCESSING_PARAMS['opening_iterations'] > 0:
        for _ in range(POSTPROCESSING_PARAMS['opening_iterations']):
            binary_mask = binary_opening(binary_mask, structure=structure)
    
    # Closing to fill gaps
    if POSTPROCESSING_PARAMS['closing_iterations'] > 0:
        for _ in range(POSTPROCESSING_PARAMS['closing_iterations']):
            binary_mask = binary_closing(binary_mask, structure=structure)
    
    # Edge preservation (optional)
    if POSTPROCESSING_PARAMS['edge_preservation']:
        # Apply slight erosion to clean edges
        binary_mask = binary_erosion(binary_mask, structure=disk(1))
        binary_mask = binary_dilation(binary_mask, structure=disk(1))
    
    return binary_mask.astype(np.uint8)

def raster_to_shapefile(raster_path, output_shapefile):
    """Convert raster to shapefile - same as training script."""
    try:
        with rasterio.open(raster_path) as src:
            image = src.read(1)
            mask = image > 0
            crs = src.crs
            transform = src.transform
            results = list(shapes(image, mask=mask, transform=transform))
            features = []
            for geom, value in results:
                if int(value) == 0: continue
                shapely_geom = shape(geom)
                if shapely_geom.is_valid and shapely_geom.is_valid and shapely_geom.area > 0:
                    features.append({
                        "geometry": shapely_geom,
                        "properties": {"value": int(value), "area_km2": float(shapely_geom.area)/1_000_000.0}
                    })
            if not features:
                logging.warning(f"No valid geometries in {raster_path}")
                return
            gdf = gpd.GeoDataFrame.from_features(features, crs=crs)
            gdf.to_file(output_shapefile)
            logging.info(f"Shapefile created: {output_shapefile}")
    except Exception as e:
        logging.error(f"Shapefile conversion failed: {e}")

# --------------------------------------------------------------------------
# START: NEW FUNCTION ADDED
# --------------------------------------------------------------------------

def generate_qmd_report(qmd_output_path, pair_index, t1_image_path, t2_image_path, change_mask_path, change_percentage):
    """Generates a Quarto Markdown (.qmd) file for a single test pair."""
    
    # Get just the filenames for relative linking within the QMD file
    # This assumes all files (qmd, t1, t2, mask) are in the same output directory
    t1_filename = os.path.basename(t1_image_path)
    t2_filename = os.path.basename(t2_image_path)
    mask_filename = os.path.basename(change_mask_path)

    # Define the content of the .qmd file
    qmd_content = f"""---
title: "Change Detection Report - Pair {pair_index}"
format: html
editor: visual
---

## Change Detection Report for Pair {pair_index}

This report shows the change detection results for image pair {pair_index}.

### Statistics
* **Detected Change Percentage:** {change_percentage:.2f}%

---

## Visual Comparison

Here are the 'before' and 'after' images, alongside the generated change mask.
*Note: Images may not render correctly in the Quarto visual editor due to GeoTIFF format, but they will render in the final HTML output.*

::: {{layout-ncols=3}}
### Before (T1)
![Before Image]({t1_filename})

### After (T2)
![After Image]({t2_filename})

### Change Mask
![Change Mask]({mask_filename})
:::
"""
    
    try:
        with open(qmd_output_path, "w", encoding="utf-8") as f:
            f.write(qmd_content)
        logging.info(f"Generated QMD report: {qmd_output_path}")
    except Exception as e:
        logging.error(f"Failed to generate QMD report: {e}")

# --------------------------------------------------------------------------
# END: NEW FUNCTION ADDED
# --------------------------------------------------------------------------

def load_trained_model(model_path):
    """Load the trained model."""
    if not os.path.exists(model_path):
        logging.error(f"Trained model not found at {model_path}")
        return None
    
    model = UNet(in_channels=6, out_channels=1).to(DEVICE)
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        logging.info(f"✅ Successfully loaded trained model from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return None

def run_inference_on_test_data(preprocessed_folder, model_path, output_folder):
    """Run inference on preprocessed test data."""
    logging.info("===== STARTING TEST INFERENCE =====")
    
    # Load trained model
    model = load_trained_model(model_path)
    if model is None:
        return False
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Get test data paths
    t1_dir = Path(preprocessed_folder) / "T1"
    t2_dir = Path(preprocessed_folder) / "T2"
    
    if not t1_dir.exists() or not t2_dir.exists():
        logging.error("Preprocessed test data not found!")
        return False
    
    t1_files = sorted(t1_dir.glob("old_image_*.tif"))
    t2_files = sorted(t2_dir.glob("new_image_*.tif"))
    
    if len(t1_files) != len(t2_files):
        logging.error("Mismatch in number of T1 and T2 files!")
        return False
    
    logging.info(f"Found {len(t1_files)} test pairs for inference")
    
    # Run inference on each test pair
    for i, (t1_path, t2_path) in enumerate(zip(t1_files, t2_files)):
        logging.info(f"🔄 Running inference on test pair {i+1}/{len(t1_files)}")
        
        try:
            # Read images
            with rasterio.open(t1_path) as src:
                old_img = src.read().astype(np.float32)
                src_bounds = src.bounds
                src_crs = src.crs
                src_transform = src.transform
                src_meta = src.meta.copy()
            
            with rasterio.open(t2_path) as src:
                new_img = src.read().astype(np.float32)
            
            # Prepare tensors for patch-wise inference
            t1 = torch.from_numpy(old_img).float().to(DEVICE)  # Shape: (C, H, W)
            t2 = torch.from_numpy(new_img).float().to(DEVICE)  # Shape: (C, H, W)
            
            logging.info(f"Input image shapes - T1: {t1.shape}, T2: {t2.shape}")
            
            # Run inference (patch-wise or full image)
            if INFERENCE_PARAMS['use_patch_wise']:
                logging.info("Using patch-wise inference")
                prob_map = patch_wise_inference(model, t1, t2, PATCH_SIZE, PATCH_OVERLAP)
            else:
                logging.info("Using full image inference")
                # Fallback to original method for small images or debugging
                t1_resized = resize_tensor(t1).unsqueeze(0)
                t2_resized = resize_tensor(t2).unsqueeze(0)
                
                with torch.no_grad():
                    logits = model(t1_resized, t2_resized)
                    prob = torch.sigmoid(logits)
                    # Resize back to original size
                    prob_map = F.interpolate(prob, size=(t1.shape[-2], t1.shape[-1]), 
                                           mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            
            # Convert to numpy for further processing
            prob_np = prob_map.cpu().numpy()
            
            # Debug: Log probability statistics
            prob_min, prob_max = prob_np.min(), prob_np.max()
            prob_mean = prob_np.mean()
            logging.info(f"Probability range: {prob_min:.6f} to {prob_max:.6f}, mean: {prob_mean:.6f}")
            
            # Adaptive thresholding based on probability distribution
            if INFERENCE_PARAMS['adaptive_threshold']:
                threshold = max(INFERENCE_PARAMS['min_threshold'], 
                                np.percentile(prob_np, INFERENCE_PARAMS['threshold_percentile']))
            else:
                threshold = INFERENCE_PARAMS['min_threshold']
            
            logging.info(f"Using threshold: {threshold:.4f}")
            
            # Create binary mask
            pred_mask_raw = (prob_np > threshold).astype(np.uint8)
            pred_mask_cleaned = post_process_mask(pred_mask_raw)
            
            # Save probability map for debugging if requested
            if INFERENCE_PARAMS['save_probability_map']:
                prob_out = os.path.join(output_folder, f"Test_Probability_Map_{i+1}.tif")
                prob_meta = src_meta.copy()
                prob_meta.update(count=1, dtype='float32', height=src_meta['height'],
                                 width=src_meta['width'], transform=src_transform)
                with rasterio.open(prob_out, "w", **prob_meta) as dst:
                    dst.write((prob_np * 255).astype(np.float32), 1)
                logging.info(f"Saved probability map: {prob_out}")
            
            # Ensure proper mask values: 255 for change (white), 0 for no change (black)
            pred_mask_raster = pred_mask_cleaned * 255
            
            # Debug: Log mask statistics
            unique_values = np.unique(pred_mask_raster)
            change_pixels = np.sum(pred_mask_raster > 0)
            total_pixels = pred_mask_raster.size
            logging.info(f"Predicted mask unique values: {unique_values}")
            logging.info(f"Change pixels: {change_pixels} out of {total_pixels} ({change_pixels/total_pixels*100:.2f}%)")
            
            # Save results
            raster_out = os.path.join(output_folder, f"Test_Predicted_Change_{i+1}.tif")
            shape_out = os.path.join(output_folder, f"Test_Predicted_Change_{i+1}.shp")
            
            meta = src_meta.copy()
            meta.update(count=1, dtype='uint8', height=src_meta['height'],
                        width=src_meta['width'], transform=src_transform)
            
            with rasterio.open(raster_out, "w", **meta) as dst:
                dst.write(pred_mask_raster, 1)
            
            raster_to_shapefile(raster_out, shape_out)

            # --------------------------------------------------------------------------
            # START: NEW CODE BLOCK ADDED
            # --------------------------------------------------------------------------
            
            # Define output paths for T1, T2, and QMD files in the output folder
            t1_out_path = os.path.join(output_folder, f"Test_T1_Image_{i+1}.tif")
            t2_out_path = os.path.join(output_folder, f"Test_T2_Image_{i+1}.tif")
            qmd_out_path = os.path.join(output_folder, f"Test_Report_{i+1}.qmd")
            
            # Copy T1 and T2 images to the output folder to be alongside the QMD report
            try:
                shutil.copy(t1_path, t1_out_path)
                shutil.copy(t2_path, t2_out_path)
            except Exception as e:
                logging.warning(f"Could not copy source T1/T2 images to output folder: {e}")

            # Calculate change percentage
            change_percentage = (change_pixels / total_pixels) * 100 if total_pixels > 0 else 0
            
            # Generate the QMD report
            generate_qmd_report(
                qmd_output_path=qmd_out_path,
                pair_index=i+1,
                t1_image_path=t1_out_path,     # Use the *new* path in the output folder
                t2_image_path=t2_out_path,     # Use the *new* path in the output folder
                change_mask_path=raster_out, # This path is already in the output folder
                change_percentage=change_percentage
            )
            
            # --------------------------------------------------------------------------
            # END: NEW CODE BLOCK ADDED
            # --------------------------------------------------------------------------
            
            logging.info(f"✅ Saved results for test pair {i+1}")
            
        except Exception as e:
            logging.error(f"❌ Error processing test pair {i+1}: {e}")
            continue
    
    logging.info(f"===== TEST INFERENCE COMPLETE: Results saved in {output_folder} =====")
    return True

def run_testing_pipeline():
    """Main testing pipeline."""
    logging.info("🚀 Starting Testing Pipeline")
    
    # Check if trained model exists
    if not os.path.exists(TRAINED_MODEL_PATH):
        logging.error(f"❌ Trained model not found at {TRAINED_MODEL_PATH}")
        logging.error("Please train the model first using mock_train_improved.py")
        return False
    
    # Check if test data exists
    if not os.path.exists(TEST_DATA_FOLDER):
        logging.error(f"❌ Test data folder not found at {TEST_DATA_FOLDER}")
        logging.error("Please create the test data folder and add your test zip files")
        return False
    
    # Create output directories
    os.makedirs(TEST_OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(TEST_PREPROCESSED_FOLDER, exist_ok=True)
    
    # Step 1: Preprocess test data
    logging.info("📋 Step 1: Preprocessing test data...")
    if not preprocess_test_data(TEST_DATA_FOLDER, TEST_PREPROCESSED_FOLDER):
        logging.error("❌ Test data preprocessing failed")
        return False
    
    # Step 2: Run inference
    logging.info("📋 Step 2: Running inference...")
    if not run_inference_on_test_data(TEST_PREPROCESSED_FOLDER, TRAINED_MODEL_PATH, TEST_OUTPUT_FOLDER):
        logging.error("❌ Test inference failed")
        return False
    
    logging.info("🎉 Testing pipeline completed successfully!")
    logging.info(f"📁 Results saved in: {TEST_OUTPUT_FOLDER}")
    return True

# --------------------------------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------------------------------
if __name__ == '__main__':
    # Update these paths according to your setup
    print("🔧 Testing Configuration:")
    print(f"    Test Data Folder: {TEST_DATA_FOLDER}")
    print(f"    Trained Model Path: {TRAINED_MODEL_PATH}")
    print(f"    Test Output Folder: {TEST_OUTPUT_FOLDER}")
    print(f"    Test Preprocessed Folder: {TEST_PREPROCESSED_FOLDER}")
    print(f"    Device: {DEVICE}")
    print(f"    Patch-wise Inference: {INFERENCE_PARAMS['use_patch_wise']}")
    print(f"    Patch Size: {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"    Patch Overlap: {PATCH_OVERLAP} pixels")
    print(f"    Adaptive Thresholding: {INFERENCE_PARAMS['adaptive_threshold']}")
    print(f"    Skip Preprocessing: {PREPROCESSING_PARAMS['skip_if_exists']}")
    print(f"    Force Reprocess: {PREPROCESSING_PARAMS['force_reprocess']}")
    print()
    
    # Run the testing pipeline
    success = run_testing_pipeline()
    
    if success:
        print("\n✅ Testing completed successfully!")
        print(f"📁 Check results in: {TEST_OUTPUT_FOLDER}")
    else:
        print("\n❌ Testing failed. Check the logs above for details.")
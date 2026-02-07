# mock_train_improved.py
# Enhanced version for better man-made change detection
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
from torch import nn, optim
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
# ENHANCED CONFIGURATION
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_FOLDER = str(BASE_DIR / "data" / "raw")
PREPROCESSED_DATA_FOLDER = str(BASE_DIR / "data" / "preprocess")
MODEL_OUTPUT_FOLDER = str(BASE_DIR / "data" / "output")

MODEL_SAVE_PATH = os.path.join(MODEL_OUTPUT_FOLDER, "saved_model.pth")
BATCH_SIZE = 2
EPOCHS = 70 # Increased epochs for better training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SIZE = (512, 512)  # H,W
NUM_WORKERS = min(8, max(0, multiprocessing.cpu_count() - 1))

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
    'min_object_size': 50,  # Reduced to detect smaller changes
    'morphology_kernel_size': 2,  # Smaller kernel to preserve more details
    'gaussian_sigma': 0.8,  # Reduced smoothing to preserve changes
    'median_filter_size': 2,  # Smaller median filter
    'opening_iterations': 1,  # Less aggressive opening
    'closing_iterations': 1,  # Less aggressive closing
    'edge_preservation': True,
    'smoothing_iterations': 1,  # Reduced smoothing passes
    'area_threshold_ratio': 0.0001  # Much smaller area threshold
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --------------------------------------------------------------------------
# ENHANCED UTILITY FUNCTIONS
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
    folder = Path(folder)
    jp2_files = list(folder.rglob("*.jp2"))
    tif_files = list(folder.rglob("*.tif"))
    bands = {"red": None, "green": None, "blue": None, "nir": None}

    if jp2_files:
        jp2_10m = [f for f in jp2_files if "r10m" in str(f).lower() or "10m" in str(f).lower()]
        if jp2_10m:
            jp2_files = jp2_10m
        for f in jp2_files:
            n = f.name.lower()
            if "b02" in n: bands["blue"] = f
            elif "b03" in n: bands["green"] = f
            elif "b04" in n: bands["red"] = f
            elif "b08" in n: bands["nir"] = f
        return bands

    if tif_files:
        for f in tif_files:
            n = f.name.lower()
            if "band2" in n or "b02" in n: bands["blue"] = f
            elif "band3" in n or "b03" in n: bands["green"] = f
            elif "band4" in n or "b04" in n: bands["red"] = f
            elif "band8" in n or "b08" in n: bands["nir"] = f
        return bands

    logging.error(f"No recognizable bands found in {folder}")
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
        meta.update(driver="GTiff", count=3, dtype='float32', compress="lzw", crs=FORCED_CRS)
        with rasterio.open(output_path, "w", **meta) as dst:
            for idx, bname in enumerate(["red", "green", "blue"], start=1):
                path = band_map.get(bname)
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
        dest_array = np.zeros((ref.count, ref.height, ref.width), dtype=src.read().dtype)
        reproject(
            source=src.read(),
            destination=dest_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref.transform,
            dst_crs=FORCED_CRS,
            resampling=Resampling.nearest
        )
        meta = ref.meta.copy()
        meta.update(count=ref.count, dtype=dest_array.dtype, crs=FORCED_CRS)
        with rasterio.open(out_aligned_path, "w", **meta) as dst:
            dst.write(dest_array)
    return out_aligned_path

def advanced_builtup_detection(input_stack_path, output_mask_path):
    """Enhanced built-up detection using multiple spectral indices and texture analysis."""
    with rasterio.open(input_stack_path) as src:
        data = src.read().astype(np.float32)
        red, green, blue = data[0], data[1], data[2]
        
        # Use NIR if available, otherwise create synthetic NIR
        nir = data[3] if len(data) > 3 else (red + green + blue) / 3.0
        
        # Enhanced spectral indices for man-made structure detection
        ndvi = (nir - red) / (nir + red + 1e-6)
        ndbi = (nir - blue) / (nir + blue + 1e-6)  # Normalized Difference Built-up Index
        ndwi = (green - nir) / (green + nir + 1e-6)  # Normalized Difference Water Index
        
        # Original indices
        index = (green - red) / (green + red + 1e-6)
        brightness = (red + green + blue) / 3.0
        
        # Texture analysis for man-made structures
        if ENHANCEMENT_PARAMS['texture_analysis']:
            # Calculate edge strength using Sobel operator
            edge_strength = sobel(brightness)
            # Calculate local variance for texture
            from scipy.ndimage import uniform_filter
            local_mean = uniform_filter(brightness, size=5)
            local_variance = uniform_filter(brightness**2, size=5) - local_mean**2
        else:
            edge_strength = np.zeros_like(brightness)
            local_variance = np.zeros_like(brightness)
        
        # Multi-criteria approach for better man-made structure detection
        # 1. Spectral criteria
        spectral_criteria = (
            (index < np.percentile(index, 20)) &  # Low vegetation index
            (brightness > np.percentile(brightness, 70)) &  # High brightness
            (ndvi < 0.2) &  # Low vegetation
            (ndbi > 0.1)  # Built-up index
        )
        
        # 2. Texture criteria
        texture_criteria = (
            (edge_strength > np.percentile(edge_strength, 60)) |  # High edge strength
            (local_variance > np.percentile(local_variance, 70))  # High local variance
        )
        
        # 3. Combined criteria
        built_up_criteria = spectral_criteria | texture_criteria
        
        # Remove small objects and apply morphological operations
        mask = built_up_criteria.astype(np.uint8)
        
        # Advanced post-processing
        if POSTPROCESSING_PARAMS['edge_preservation']:
            # Preserve edges while cleaning noise
            mask = binary_opening(mask, structure=disk(1))
            mask = binary_closing(mask, structure=disk(2))
        else:
            # Standard morphological operations
            mask = binary_opening(mask, structure=disk(POSTPROCESSING_PARAMS['morphology_kernel_size']))
            mask = binary_closing(mask, structure=disk(POSTPROCESSING_PARAMS['morphology_kernel_size']))
        
        # Remove very small objects
        min_size = POSTPROCESSING_PARAMS['min_object_size']
        mask = sk_remove_small_objects(mask.astype(bool), min_size=min_size).astype(np.uint8)
        
        # Debug: Log detection statistics
        detected_pixels = np.sum(mask > 0)
        logging.info(f"Enhanced built-up detection: {detected_pixels} pixels detected out of {mask.size} total")
        
        meta = src.meta.copy()
        meta.update(count=1, dtype='uint8', compress="lzw", crs=FORCED_CRS)
        with rasterio.open(output_mask_path, "w", **meta) as dst:
            dst.write(mask, 1)
    return output_mask_path

def create_final_change_mask(old_mask_path, new_mask_path, final_change_path):
    with rasterio.open(old_mask_path) as old_src, rasterio.open(new_mask_path) as new_src:
        old_mask = old_src.read(1)
        new_mask = new_src.read(1)
        
        # Enhanced change detection with multiple criteria
        # 1. New built-up areas
        new_builtup = ((new_mask == 1) & (old_mask == 0)).astype(np.uint8)
        
        # 2. Expansion of existing built-up areas
        expansion = ((new_mask == 1) & (old_mask == 1)).astype(np.uint8)
        
        # 3. Significant changes in built-up density
        from scipy.ndimage import uniform_filter
        old_density = uniform_filter(old_mask.astype(float), size=10)
        new_density = uniform_filter(new_mask.astype(float), size=10)
        density_change = (new_density - old_density) > 0.3
        
        # Combine all change types
        change = (new_builtup | expansion | density_change).astype(np.uint8) * 255
        
        # Apply final post-processing
        change_binary = (change > 0).astype(np.uint8)
        change_binary = binary_closing(change_binary, structure=disk(2))
        change_binary = sk_remove_small_objects(change_binary.astype(bool), min_size=30).astype(np.uint8)
        change = change_binary * 255
        
        meta = old_src.meta.copy()
        meta.update(count=1, dtype='uint8', compress="lzw")
        with rasterio.open(final_change_path, "w", **meta) as dst:
            dst.write(change, 1)
    return final_change_path

def process_raw_pair(pair_zip_path, pair_index, output_root):
    logging.info(f"--- Processing {pair_zip_path.name} with enhanced detection ---")
    t1_dir, t2_dir, masks_dir = Path(output_root)/"T1", Path(output_root)/"T2", Path(output_root)/"Masks"
    t1_dir.mkdir(parents=True, exist_ok=True)
    t2_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        if not unzip_to(pair_zip_path, tmp):
            logging.error(f"Could not unzip {pair_zip_path}")
            return
        nested_zips = sorted(tmp.glob("*.zip"))
        if len(nested_zips) < 2:
            logging.error(f"Missing nested zips in {pair_zip_path.name}")
            return
        first_data_dir, second_data_dir = tmp/"first", tmp/"second"
        unzip_to(nested_zips[0], first_data_dir)
        unzip_to(nested_zips[1], second_data_dir)

        first_bands, second_bands = get_band_paths(first_data_dir), get_band_paths(second_data_dir)
        if not first_bands or not second_bands:
            logging.error("Could not detect bands in nested zips.")
            return

        tmp_old_image = stack_bands(first_bands, tmp/"old_image.tif")
        tmp_new_image = stack_bands(second_bands, tmp/"new_image.tif")
        if not tmp_old_image or not tmp_new_image:
            logging.error("Stacking failed")
            return

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
        
        # Use enhanced images for detection
        tmp_old_mask = advanced_builtup_detection(old_enhanced_path, tmp/"old_builtup_mask.tif")
        tmp_new_mask = advanced_builtup_detection(new_enhanced_path, tmp/"new_builtup_mask.tif")
        tmp_change_mask = create_final_change_mask(tmp_old_mask, tmp_new_mask, tmp/"final_change_mask.tif")
        if not tmp_change_mask:
            logging.error("Change mask creation failed")
            return
            
        # Debug: Check change mask statistics
        with rasterio.open(tmp_change_mask) as src:
            change_data = src.read(1)
            change_pixels = np.sum(change_data > 0)
            change_percentage = (change_pixels / change_data.size) * 100
            logging.info(f"Enhanced change mask for pair {pair_index}: {change_pixels} change pixels out of {change_data.size} total ({change_percentage:.2f}%)")

        # Skip dataset if no changes detected (zero pixel changes)
        if change_pixels == 0:
            logging.warning(f"⚠️ SKIPPING pair {pair_index}: No changes detected (0 change pixels) - dataset will be excluded from training")
            return False

        # Skip dataset if very few changes detected (less than 0.01% of image)
        if change_percentage < 0.01:
            logging.warning(f"⚠️ SKIPPING pair {pair_index}: Very few changes detected ({change_percentage:.3f}%) - dataset will be excluded from training")
            return False

        shutil.copy(tmp_old_image, t1_dir / f"old_image_{pair_index}.tif")
        shutil.copy(tmp_new_image_aligned, t2_dir / f"new_image_{pair_index}.tif")
        shutil.copy(tmp_change_mask, masks_dir / f"mask_{pair_index}.tif")
        logging.info(f"✅ Successfully preprocessed and saved enhanced training set {pair_index} with {change_percentage:.2f}% changes")
        return True

def validate_and_clean_files(folder_path):
    """Validate and clean corrupted TIFF files."""
    folder = Path(folder_path)
    corrupted_files = []
    
    for tif_file in folder.glob("*.tif"):
        try:
            with rasterio.open(tif_file) as src:
                # Try to read a small portion to test file integrity
                test_data = src.read(1, window=rasterio.windows.Window(0, 0, 100, 100))
                logging.info(f"✅ File {tif_file.name} is valid")
        except Exception as e:
            logging.error(f"❌ Corrupted file detected: {tif_file.name} - {e}")
            corrupted_files.append(tif_file)
    
    # Remove corrupted files
    for corrupted_file in corrupted_files:
        try:
            corrupted_file.unlink()
            logging.info(f"🗑 Removed corrupted file: {corrupted_file.name}")
        except Exception as e:
            logging.error(f"Failed to remove corrupted file {corrupted_file.name}: {e}")
    
    return len(corrupted_files)

def run_preprocessing(raw_data_folder, preprocessed_folder):
    # Check if preprocessed exists
    t1_dir = Path(preprocessed_folder)/"T1"
    t2_dir = Path(preprocessed_folder)/"T2"
    masks_dir = Path(preprocessed_folder)/"Masks"
    if t1_dir.exists() and t2_dir.exists() and masks_dir.exists():
        if list(t1_dir.glob("*.tif")) and list(t2_dir.glob("*.tif")) and list(masks_dir.glob("*.tif")):
            logging.info("Preprocessed data found. Validating files...")
            # Validate existing files and clean corrupted ones
            validate_and_clean_files(t1_dir)
            validate_and_clean_files(t2_dir)
            validate_and_clean_files(masks_dir)
            logging.info("File validation complete.")
            return True

    logging.info("===== STARTING ENHANCED PREPROCESSING STEP =====")
    raw_path, prep_path = Path(raw_data_folder), Path(preprocessed_folder)
    (prep_path/"T1").mkdir(parents=True, exist_ok=True)
    (prep_path/"T2").mkdir(parents=True, exist_ok=True)
    (prep_path/"Masks").mkdir(parents=True, exist_ok=True)
    zips = sorted(raw_path.glob("Pair*.zip"))
    if not zips:
        logging.error(f"No 'Pair*.zip' files found in {raw_data_folder}")
        return False
    
    processed_count = 0
    skipped_count = 0
    
    for i, zip_file in enumerate(zips):
        result = process_raw_pair(zip_file, i + 1, prep_path)
        if result is True:
            processed_count += 1
        else:
            skipped_count += 1
    
    logging.info(f"===== ENHANCED PREPROCESSING COMPLETE =====")
    logging.info(f"📊 Processing Summary:")
    logging.info(f"   ✅ Successfully processed: {processed_count} datasets")
    logging.info(f"   ⚠️ Skipped (no changes): {skipped_count} datasets")
    logging.info(f"   📁 Data saved in: {preprocessed_folder}")
    
    if processed_count == 0:
        logging.error("❌ No valid datasets processed - all pairs had zero or insufficient changes")
        return False
    
    return True

# --------------------------------------------------------------------------
# ENHANCED DATASET & MODEL
# --------------------------------------------------------------------------
def resize_tensor(tensor, size=TARGET_SIZE):
    return F.interpolate(tensor.unsqueeze(0), size=size, mode='bilinear', align_corners=False).squeeze(0)

class DiceLoss(nn.Module):
    """Simple Dice loss for better training stability."""
    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs).view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice_score = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice_score

class ChangeDetectionDataset(Dataset):
    def __init__(self, root_dir, target_size=TARGET_SIZE):
        self.root_dir, self.target_size = root_dir, target_size
        self.t1_dir = Path(root_dir) / "T1"
        self.t2_dir = Path(root_dir) / "T2"
        self.masks_dir = Path(root_dir) / "Masks"
        self.t1_files, self.t2_files, self.mask_files = [], [], []
        
        # Simple augmentation for better training stability
        self.transform = A.Compose(
            [A.RandomRotate90(p=0.5), A.VerticalFlip(p=0.5), A.HorizontalFlip(p=0.5)],
            additional_targets={'image1': 'image'}
        )
        
        for t1_path in sorted(self.t1_dir.glob("old_image_*.tif")):
            match = re.search(r'_(\d+)\.tif', t1_path.name)
            if not match: continue
            identifier = match.group(1)
            t2_path = self.t2_dir / f"new_image_{identifier}.tif"
            mask_path = self.masks_dir / f"mask_{identifier}.tif"
            if t2_path.exists() and mask_path.exists():
                # Validate that mask has changes before adding to dataset
                try:
                    with rasterio.open(mask_path) as src:
                        mask_data = src.read(1)
                        change_pixels = np.sum(mask_data > 0)
                        if change_pixels > 0:
                            self.t1_files.append(t1_path)
                            self.t2_files.append(t2_path)
                            self.mask_files.append(mask_path)
                            logging.info(f"✅ Added dataset {identifier} with {change_pixels} change pixels")
                        else:
                            logging.warning(f"⚠️ Skipped dataset {identifier}: No changes detected in mask")
                except Exception as e:
                    logging.error(f"❌ Failed to validate mask for dataset {identifier}: {e}")
                    continue
        logging.info(f"📊 Dataset Summary: Found {len(self.t1_files)} valid training sets with changes in {root_dir}")

    def __len__(self):
        return len(self.t1_files)

    def __getitem__(self, idx):
        try:
            # Validate files exist and are readable
            if not self.t1_files[idx].exists():
                raise FileNotFoundError(f"T1 file not found: {self.t1_files[idx]}")
            if not self.t2_files[idx].exists():
                raise FileNotFoundError(f"T2 file not found: {self.t2_files[idx]}")
            if not self.mask_files[idx].exists():
                raise FileNotFoundError(f"Mask file not found: {self.mask_files[idx]}")
            
            # Try to read files with error handling
            try:
                with rasterio.open(self.t1_files[idx]) as src:
                    old_img_chw = src.read().astype(np.float32)
            except Exception as e:
                logging.error(f"Failed to read T1 file {self.t1_files[idx]}: {e}")
                # Create a dummy image as fallback
                old_img_chw = np.zeros((3, 512, 512), dtype=np.float32)
            
            try:
                with rasterio.open(self.t2_files[idx]) as src:
                    new_img_chw = src.read().astype(np.float32)
            except Exception as e:
                logging.error(f"Failed to read T2 file {self.t2_files[idx]}: {e}")
                # Create a dummy image as fallback
                new_img_chw = np.zeros((3, 512, 512), dtype=np.float32)
            
            try:
                with rasterio.open(self.mask_files[idx]) as src:
                    mask_img_chw = src.read().astype(np.float32)
                    # Debug: Log ground truth mask statistics
                    mask_unique = np.unique(mask_img_chw)
                    mask_change_pixels = np.sum(mask_img_chw > 0)
                    logging.info(f"Ground truth mask {self.mask_files[idx].name}: unique values {mask_unique}, change pixels: {mask_change_pixels}")
            except Exception as e:
                logging.error(f"Failed to read mask file {self.mask_files[idx]}: {e}")
                # Create a dummy mask as fallback
                mask_img_chw = np.zeros((1, 512, 512), dtype=np.float32)
                
        except Exception as e:
            logging.error(f"Error loading dataset item {idx}: {e}")
            # Return dummy data to prevent training crash
            old_img_chw = np.zeros((3, 512, 512), dtype=np.float32)
            new_img_chw = np.zeros((3, 512, 512), dtype=np.float32)
            mask_img_chw = np.zeros((1, 512, 512), dtype=np.float32)

        old_hwc = np.transpose(old_img_chw, (1,2,0))
        new_hwc = np.transpose(new_img_chw, (1,2,0))
        mask_hwc = np.transpose(mask_img_chw, (1,2,0))

        augmented = self.transform(image=old_hwc, image1=new_hwc, mask=mask_hwc)

        old_tensor = torch.from_numpy(np.transpose(augmented['image'], (2,0,1))).float()
        new_tensor = torch.from_numpy(np.transpose(augmented['image1'], (2,0,1))).float()
        mask_tensor = torch.from_numpy(np.transpose(augmented['mask'], (2,0,1))).float()

        old_tensor = resize_tensor(old_tensor, self.target_size)
        new_tensor = resize_tensor(new_tensor, self.target_size)
        mask_tensor = resize_tensor(mask_tensor, self.target_size)
        mask_tensor = (mask_tensor > 127).float()

        return old_tensor, new_tensor, mask_tensor

# --------------------------------------------------------------------------
# ENHANCED SIAMESE UNET
# --------------------------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = DoubleConv(in_c, out_c)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        return self.conv(x), self.pool(self.conv(x))

class Up(nn.Module):
    def __init__(self, out_c):
        super().__init__()
        self.out_c = out_c
        self.up = None
        self.conv = None
    def forward(self, x1, x2):
        if self.up is None:
            self.up = nn.ConvTranspose2d(x1.size(1), self.out_c, 2, stride=2).to(x1.device)
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        if self.conv is None:
            in_ch = x1.size(1) + x2.size(1)
            self.conv = DoubleConv(in_ch, self.out_c).to(x1.device)
        return self.conv(torch.cat([x2, x1], dim=1))

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
        
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        
        bottleneck = self.bottleneck(self.pool(enc4))
        
        dec4 = self.upconv4(bottleneck)
        dec4 = self.decoder4(torch.cat((dec4, enc4), dim=1))
        dec3 = self.upconv3(dec4)
        dec3 = self.decoder3(torch.cat((dec3, enc3), dim=1))
        dec2 = self.upconv2(dec3)
        dec2 = self.decoder2(torch.cat((dec2, enc2), dim=1))
        dec1 = self.upconv1(dec2)
        dec1 = self.decoder1(torch.cat((dec1, enc1), dim=1))
        
        return self.conv_last(dec1)

# Keep SiameseUNet as alias for backward compatibility
SiameseUNet = UNet

# --------------------------------------------------------------------------
# ENHANCED TRAINING & INFERENCE
# --------------------------------------------------------------------------
def advanced_post_process_mask(mask):
    """Balanced post-processing to create smooth, professional output without losing changes."""
    logging.info("Applying balanced post-processing for professional output...")
    
    # Check if mask has any changes
    if np.sum(mask) == 0:
        logging.warning("No changes detected in raw mask - skipping post-processing")
        return mask.astype(np.uint8)
    
    # Convert to float for better processing
    mask_float = mask.astype(np.float32)
    
    # 1. Light noise reduction with smaller median filter
    mask_float = median(mask_float, disk(2))
    
    # 2. Remove only very small objects (noise) - be conservative
    min_size = 50  # Remove objects smaller than 50 pixels (much smaller threshold)
    cleaned_mask = sk_remove_small_objects(mask_float.astype(bool), min_size=min_size)
    
    # 3. Light morphological operations for smooth boundaries
    # Single opening to remove small noise
    cleaned_mask = binary_opening(cleaned_mask, structure=disk(2))
    
    # Single closing to fill small gaps
    cleaned_mask = binary_closing(cleaned_mask, structure=disk(3))
    
    # 4. Light area-based filtering - only remove tiny regions
    labeled_mask, num_features = label(cleaned_mask)
    total_pixels = mask.size
    min_area = max(100, int(total_pixels * 0.0001))  # Much smaller area threshold
    
    for region in regionprops(labeled_mask):
        if region.area < min_area:
            cleaned_mask[labeled_mask == region.label] = False
    
    # 5. Single smoothing pass for clean boundaries
    cleaned_mask = gaussian(cleaned_mask.astype(float), sigma=1.0) > 0.5
    
    # 6. Final light morphological smoothing
    cleaned_mask = binary_closing(cleaned_mask, structure=disk(2))
    cleaned_mask = binary_opening(cleaned_mask, structure=disk(1))
    
    # 7. Final light smoothing
    cleaned_mask = gaussian(cleaned_mask.astype(float), sigma=0.8) > 0.5
    
    logging.info(f"Balanced post-processing complete. Final mask has {np.sum(cleaned_mask)} change pixels")
    return cleaned_mask.astype(np.uint8)

def post_process_mask(mask):
    """Wrapper for backward compatibility."""
    return advanced_post_process_mask(mask)

def raster_to_shapefile(raster_path, output_shapefile):
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
                if shapely_geom.is_valid and shapely_geom.area > 0:
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

def mask_for_raster(mask_binary):
    # Convert binary mask to raster format: 255 for change (white), 0 for no change (black)
    return (mask_binary.astype(np.uint8) * 255)

def run_training_and_inference(data_folder, model_path, output_folder):
    logging.info("===== STARTING ENHANCED MODEL TRAINING & INFERENCE =====")
    os.makedirs(output_folder, exist_ok=True)
    dataset = ChangeDetectionDataset(data_folder)
    if len(dataset) == 0:
        logging.error("Training dataset empty. Aborting.")
        return

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0, pin_memory=False)  # Disable multiprocessing to avoid file corruption issues
    model = SiameseUNet().to(DEVICE)
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    logging.info(f"Training for {EPOCHS} epochs on {DEVICE}...")
    best_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for batch_idx, (t1, t2, mask) in enumerate(loader):
            t1 = t1.to(DEVICE)
            t2 = t2.to(DEVICE)
            mask = mask.to(DEVICE)
            
            # Debug: Log mask statistics during training
            if batch_idx == 0:  # Log only for first batch of each epoch
                mask_change_pixels = torch.sum(mask > 0).item()
                logging.info(f"Epoch {epoch+1}, Batch 0: {mask_change_pixels} change pixels in ground truth")
            
            optimizer.zero_grad()
            output = model(t1, t2)
            loss = criterion(output, mask)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss/len(loader)
        logging.info(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), model_path)
    logging.info(f"Training finished. Model saved to {model_path}")

    # Enhanced inference
    logging.info("Starting enhanced inference...")
    model.eval()
    for i in range(len(dataset)):
        old_path = dataset.t1_files[i]
        new_path = dataset.t2_files[i]
        with rasterio.open(old_path) as src:
            old_img = src.read().astype(np.float32)
            src_bounds = src.bounds
            src_crs = src.crs
            src_transform = src.transform
            src_meta = src.meta.copy()
        with rasterio.open(new_path) as src:
            new_img = src.read().astype(np.float32)

        t1 = resize_tensor(torch.from_numpy(old_img).float()).unsqueeze(0).to(DEVICE)
        t2 = resize_tensor(torch.from_numpy(new_img).float()).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(t1, t2)
            prob = torch.sigmoid(logits)
            
            # Debug: Log probability statistics
            prob_min, prob_max = prob.min().item(), prob.max().item()
            prob_mean = prob.mean().item()
            logging.info(f"Probability range: {prob_min:.4f} to {prob_max:.4f}, mean: {prob_mean:.4f}")
            
            # Use adaptive threshold - start with lower threshold
            prob_np = prob.cpu().numpy()[0,0]
            threshold = max(0.1, np.percentile(prob_np, 85))  # Use 85th percentile or 0.1, whichever is higher
            logging.info(f"Using threshold: {threshold:.4f}")
            
            pred_mask_raw = (prob > threshold).cpu().numpy()[0,0].astype(np.uint8)
            
            # Debug: Log raw mask statistics
            raw_change_pixels = np.sum(pred_mask_raw > 0)
            change_percentage = (raw_change_pixels / pred_mask_raw.size) * 100
            logging.info(f"Raw mask after thresholding: {raw_change_pixels} change pixels ({change_percentage:.2f}%)")
            
            pred_mask_cleaned = post_process_mask(pred_mask_raw)
            
            # Debug: Log cleaned mask statistics
            cleaned_change_pixels = np.sum(pred_mask_cleaned > 0)
            cleaned_percentage = (cleaned_change_pixels / pred_mask_cleaned.size) * 100
            logging.info(f"After balanced post-processing: {cleaned_change_pixels} change pixels ({cleaned_percentage:.2f}%)")
            
            # Safety check: if post-processing removed too many changes, use a lighter approach
            if cleaned_percentage < 1.0 and change_percentage > 5.0:
                logging.warning(f"Post-processing too aggressive (removed {change_percentage:.2f}% -> {cleaned_percentage:.2f}%), using lighter processing...")
                # Use much lighter post-processing
                pred_mask_cleaned = binary_closing(pred_mask_raw.astype(bool), structure=disk(2)).astype(np.uint8)
                pred_mask_cleaned = sk_remove_small_objects(pred_mask_cleaned.astype(bool), min_size=20).astype(np.uint8)
                pred_mask_cleaned = gaussian(pred_mask_cleaned.astype(float), sigma=0.5) > 0.5
                pred_mask_cleaned = pred_mask_cleaned.astype(np.uint8)
                
                final_cleaned_pixels = np.sum(pred_mask_cleaned > 0)
                final_cleaned_percentage = (final_cleaned_pixels / pred_mask_cleaned.size) * 100
                logging.info(f"After lighter post-processing: {final_cleaned_pixels} change pixels ({final_cleaned_percentage:.2f}%)")
            
        # Ensure proper mask values: 255 for change (white), 0 for no change (black)
        pred_mask_raster = pred_mask_cleaned * 255
        
        # Debug: Log mask statistics
        unique_values = np.unique(pred_mask_raster)
        change_pixels = np.sum(pred_mask_raster > 0)
        total_pixels = pred_mask_raster.size
        final_percentage = (change_pixels/total_pixels*100)
        logging.info(f"Final predicted mask unique values: {unique_values}")
        logging.info(f"Final change pixels: {change_pixels} out of {total_pixels} ({final_percentage:.2f}%)")
        
        # Quality assessment
        if final_percentage > 50:
            logging.warning(f"⚠️ High change percentage ({final_percentage:.2f}%) - may appear noisy")
        elif final_percentage < 1:
            logging.warning(f"⚠️ Low change percentage ({final_percentage:.2f}%) - may be too conservative")
        else:
            logging.info(f"✅ Good change detection: {final_percentage:.2f}% - should produce clean output")

        # Align output to original raster
        raster_out = os.path.join(output_folder, f"Predicted_Change_{i+1}.tif")
        shape_out = os.path.join(output_folder, f"Predicted_Change_{i+1}.shp")
        meta = src_meta.copy()
        meta.update(count=1, dtype='uint8', height=src.height,
                    width=src.width, transform=src_transform)
        with rasterio.open(raster_out, "w", **meta) as dst:
            dst.write(pred_mask_raster, 1)
        raster_to_shapefile(raster_out, shape_out)

    logging.info(f"Inference complete. Results saved in: {output_folder}")

# --------------------------------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------------------------------
if __name__ == '__main__':
    os.makedirs(PREPROCESSED_DATA_FOLDER, exist_ok=True)
    os.makedirs(MODEL_OUTPUT_FOLDER, exist_ok=True)
    if run_preprocessing(RAW_DATA_FOLDER, PREPROCESSED_DATA_FOLDER):
        run_training_and_inference(PREPROCESSED_DATA_FOLDER, MODEL_SAVE_PATH, MODEL_OUTPUT_FOLDER)
    else:
        logging.error("Preprocessing failed. Halting the pipeline.")
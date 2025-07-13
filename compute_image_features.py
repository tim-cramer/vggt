# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

"""
This script computes dense, pixel-wise 2D features for a single image,
replicating the logic from the Locate-3D repository.
"""
import argparse
import logging
import os
import requests
from typing import List, Optional, Union

import clip
import numpy as np
import torch
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from tqdm import tqdm

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ##################################################################################
# ## CLASSES AND FUNCTIONS FROM THE LOCATE-3D REPOSITORY
# ##################################################################################

class ClipEncoder:
    """Simple wrapper for encoding different things as text."""
    def __init__(self, version="ViT-L/14", device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and torch.cuda.is_available():
            # Use a specific GPU if available
            # device = f"cuda:{torch.cuda.current_device()}"
            pass
        self.device = device
        version = version.replace("_", "/")
        self.version = version
        logger.info(f"Loading CLIP model {self.version} onto device {self.device}...")
        self.model, self.preprocess = clip.load(self.version, device=self.device)
        logger.info("CLIP model loaded.")

    def encode_image(self, image: np.ndarray):
        """Encode this input image to a CLIP vector"""
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy() * 255
        image = image.astype(np.uint8)
        pil_image = Image.fromarray(image)
        processed_image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(processed_image)
        return image_features.float()

    def encode_text(self, text: Union[str, List[str]], truncate: bool = True):
        """Return clip vector for text"""
        if not isinstance(text, list):
            text = [text]
        text = clip.tokenize(text, truncate=truncate).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text)
        return text_features.float()

def get_sam_model(model_path: str, device: str, version="vit_h"):
    """Loads and returns a SAM model."""
    logger.info(f"Loading SAM model {version} from {model_path}...")
    model = sam_model_registry[version](checkpoint=model_path)
    model.to(device)
    logger.info("SAM model loaded.")
    return model

class MaskEmbeddingFeatureImageGenerator:
    """
    Turns an image into pixel-aligned features using SAM masks and a CLIP encoder.
    """
    NO_MASK_IDX = -1

    def __init__(
        self,
        mask_generator: SamAutomaticMaskGenerator,
        image_text_encoder: ClipEncoder,
        device: Optional[str] = None,
    ) -> None:
        self.mask_generator = mask_generator
        self.image_text_encoder = image_text_encoder
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.feat_dim = None

    def generate_mask(self, img: np.ndarray):
        logger.info("Generating masks with SAM...")
        assert not ((img / 255) == 0).all() and not (img > 255).any()
        try:
            masks = self.mask_generator.generate(img)
        except IndexError:
            masks = []
        # remove masks with zero area
        masks = list(filter(lambda x: x["bbox"][2] * x["bbox"][3] != 0, masks))
        logger.info(f"Generated {len(masks)} masks.")
        return masks

    def generate_global_features(self, img: np.ndarray):
        logger.info("Generating global CLIP feature...")
        with torch.cuda.amp.autocast(enabled=self.device.startswith("cuda")):
            global_feat = self.image_text_encoder.encode_image(img)
            global_feat /= global_feat.norm(dim=-1, keepdim=True)

        global_feat = torch.nn.functional.normalize(global_feat, dim=-1)
        global_feat = global_feat.half().to(self.device)

        if self.feat_dim is None:
            self.feat_dim = global_feat.shape[-1]
        
        logger.info(f"Global feature generated with dimension {self.feat_dim}.")
        return global_feat

    def generate_local_features(
        self, img: np.ndarray, masks: List[dict], global_feat: torch.Tensor
    ) -> torch.Tensor:
        load_image_height, load_image_width = img.shape[0], img.shape[1]
        outfeat = torch.zeros(
            load_image_height, load_image_width, self.feat_dim,
            dtype=torch.half, device=self.device
        )
        if not masks:
            logger.warning("No masks generated, returning zero feature map.")
            return outfeat

        logger.info("Generating local features for each mask...")
        feat_per_roi = []
        roi_nonzero_inds = []
        similarity_scores = []

        for mask in tqdm(masks, desc="Encoding local features"):
            _x, _y, _w, _h = map(int, mask["bbox"])
            nonzero_inds = torch.argwhere(torch.from_numpy(mask["segmentation"]))
            img_roi = img[_y : _y + _h, _x : _x + _w, :]
            
            if img_roi.size == 0:
                continue

            roifeat = self.image_text_encoder.encode_image(img_roi)
            roifeat = torch.nn.functional.normalize(roifeat, dim=-1)
            feat_per_roi.append(roifeat)
            roi_nonzero_inds.append(nonzero_inds)
            _sim = self.cosine_similarity(global_feat, roifeat)
            similarity_scores.append(_sim)

        if not feat_per_roi:
             logger.warning("No valid ROIs found, returning zero feature map.")
             return outfeat

        similarity_scores = torch.cat(similarity_scores)
        softmax_scores = torch.nn.functional.softmax(similarity_scores, dim=0)
        
        logger.info("Blending global and local features...")
        for maskidx in tqdm(range(len(masks)), desc="Blending features"):
            weighted_feat = (
                softmax_scores[maskidx] * global_feat
                + (1 - softmax_scores[maskidx]) * feat_per_roi[maskidx]
            )
            weighted_feat = torch.nn.functional.normalize(weighted_feat, dim=-1)
            outfeat[
                roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]
            ] = (weighted_feat[0].detach().half())
        
        return outfeat

    def generate_features(self, image: torch.Tensor):
        logger.info("Starting feature generation process...")
        if self.image_text_encoder is None:
            return None

        uint_img = (image.cpu().numpy() * 255).astype(np.uint8)
        if uint_img.shape[-1] != 3:
            raise ValueError(f"Expected RGB image, got shape {uint_img.shape}")

        masks = self.generate_mask(uint_img)
        global_feat = self.generate_global_features(uint_img)
        outfeat = self.generate_local_features(uint_img, masks, global_feat)
        
        logger.info("Feature generation complete.")
        return outfeat

# ##################################################################################
# ## MAIN SCRIPT LOGIC
# ##################################################################################

def download_file(url, destination):
    """Downloads a file with a progress bar."""
    logger.info(f"Downloading {url} to {destination}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
        desc=destination,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    logger.info("Download complete.")

def main(args):
    """Main function to run the feature computation."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- 1. Setup Models ---
    
    # Download SAM checkpoint if it doesn't exist
    sam_checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    sam_checkpoint_path = "sam_vit_h_4b8939.pth"
    if not os.path.exists(sam_checkpoint_path):
        download_file(sam_checkpoint_url, sam_checkpoint_path)

    # Instantiate CLIP Encoder
    clip_encoder = ClipEncoder(version="ViT-L/14", device=device)
    
    # Instantiate SAM Mask Generator
    sam_model = get_sam_model(model_path=sam_checkpoint_path, version="vit_h", device=device)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam_model,
        points_per_side=32,
        pred_iou_thresh=0.86,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
    )
    
    # Instantiate the main Feature Generator
    feature_generator = MaskEmbeddingFeatureImageGenerator(
        mask_generator=mask_generator,
        image_text_encoder=clip_encoder,
        device=device
    )
    
    # --- 2. Load and Process Image ---
    logger.info(f"Loading image from {args.image_path}...")
    try:
        image_pil = Image.open(args.image_path).convert("RGB")
        image_np = np.array(image_pil)
        image_tensor = torch.from_numpy(image_np / 255.0).float().to(device)
    except FileNotFoundError:
        logger.error(f"Image file not found at {args.image_path}")
        return
    except Exception as e:
        logger.error(f"Could not load or process image: {e}")
        return

    # --- 3. Compute Features ---
    with torch.no_grad():
        pixelwise_features = feature_generator.generate_features(image_tensor)
        
    # --- 4. Save Output ---
    output_path = args.output_path
    if not output_path:
        base_name = os.path.splitext(os.path.basename(args.image_path))[0]
        output_path = f"{base_name}_features.pt"

    logger.info(f"Saving feature tensor of shape {pixelwise_features.shape} to {output_path}...")
    torch.save(pixelwise_features.cpu(), output_path)
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute dense 2D features for a single image using the Locate-3D methodology."
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to the input image file.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the output feature tensor. Defaults to '[image_name]_features.pt'.",
    )
    
    args = parser.parse_args()
    main(args)
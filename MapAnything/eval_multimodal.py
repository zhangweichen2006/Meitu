
from mapanything.utils.image import preprocess_inputs

# Optional config for better memory efficiency
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Required imports
import torch
from mapanything.models import MapAnything
from mapanything.utils.image import load_images

# Get inference device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Init model - This requries internet access or the huggingface hub cache to be pre-downloaded
# For Apache 2.0 license model, use "facebook/map-anything-apache"
model = MapAnything.from_pretrained("facebook/map-anything").to(device)
image_tensor = load_images("../data/meitu_img_test/fixed")
intrinsics_tensor = torch.load("../data/meitu_img_test/fixed/intrinsics.pt")

views_example = [
    {
        "img": image_tensor,  # (H, W, 3) - [0, 255]
        "intrinsics": intrinsics_tensor,  # (3, 3)
    },
    ...
]
# Preprocess inputs to the expected format
processed_views = preprocess_inputs(views_example)

# Run inference with any combination of inputs
predictions = model.infer(
    processed_views,                  # Any combination of input views
    memory_efficient_inference=False, # Trades off speed for more views (up to 2000 views on 140 GB)
    use_amp=True,                     # Use mixed precision inference (recommended)
    amp_dtype="bf16",                 # bf16 inference (recommended; falls back to fp16 if bf16 not supported)
    apply_mask=True,                  # Apply masking to dense geometry outputs
    mask_edges=True,                  # Remove edge artifacts by using normals and depth
    apply_confidence_mask=False,      # Filter low-confidence regions
    confidence_percentile=10,         # Remove bottom 10 percentile confidence pixels
    # Control which inputs to use/ignore
    # By default, all inputs are used when provided
    # If is_metric_scale flag is not provided, all inputs are assumed to be in metric scale
    ignore_calibration_inputs=False,
    ignore_depth_inputs=False,
    ignore_pose_inputs=False,
    ignore_depth_scale_inputs=False,
    ignore_pose_scale_inputs=False,
)

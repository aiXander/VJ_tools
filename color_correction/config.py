import cv2

# -------------------------------------------------------------------------
# Hyper‑parameters ---------------------------------------------------------
TARGET_SIZE = 512          # long edge of working resolution
FPS         = 10
DT          = 1.0 / FPS

INITIAL_BEAMER_OUT_VALUE = 0.25  # initial RGB value 0‑1
AMBIENT_LIGHT_STRENGTH   = 0.2   # scalar added to each channel

# Simulation Parameters --------------------------------------------------
MAX_ROTATION = 5.001    # degrees
MAX_SHIFT    = 0.051   # fraction of image width/height

# Optimiser ---------------------------------------------------------------
BEAMER_LEARNING_RATE = 0.05
WARP_LEARNING_RATE   = 0.001 # Might need tuning
BETA1, BETA2         = 0.9, 0.999
EPS_ADAM             = 1e-8

# --- Two-Phase Optimization ---
PHASE1_ITERATIONS = 40 # Number of iterations for alignment phase

# Loss Weights ------------------------------------------------------------
SSIM_LOSS_W = 0.0   # Weight for Structural Similarity (set to 0 for now)
RGB_LOSS_W  = 1.0   # Weight for RGB MSE loss (primary objective)

# Multi-objective loss weights for robust optimization (reduced for speed)
LAB_LOSS_W  = 0.15  # Weight for LAB color space loss (perceptual matching)
EDGE_LOSS_W = 0.1   # Weight for edge preservation loss (structural features)
HIST_LOSS_W = 0.0   # Disabled - histogram computation is expensive
GRAD_LOSS_W = 0.05  # Weight for gradient magnitude loss (image sharpness)

# Blur parameters for Phase 1 optimization smoothing
PHASE1_BLUR_SIGMA = 1.5  # Gaussian blur sigma for Phase 1 loss smoothing (0 = no blur)
PHASE1_BLUR_KERNEL_SIZE = 5  # Kernel size for Gaussian blur (must be odd)

# -------------------------------------------------------------------------
OUTPUT_FILENAME = "projection_evolution.mp4"
COMPARISON_FILENAME = "comparison_corrected.jpg" # Added for consistency
WINDOW_NAME     = "Projection Mapping Optimiser"

FONT            = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE      = 0.8 # Adjusted for more labels
FONT_THICKNESS  = 2
TEXT_COLOR      = (105, 195, 105)
# ------------------------------------------------------------------------- 
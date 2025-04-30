import cv2

# -------------------------------------------------------------------------
# Hyper‑parameters ---------------------------------------------------------
TARGET_SIZE = 1024          # long edge of working resolution
FPS         = 10
DT          = 1.0 / FPS

INITIAL_BEAMER_OUT_VALUE = 0.25  # initial RGB value 0‑1
AMBIENT_LIGHT_STRENGTH   = 0.2   # scalar added to each channel

# Simulation Parameters --------------------------------------------------
MAX_ROTATION = 10.0    # degrees
MAX_SHIFT    = 0.1   # fraction of image width/height

# Optimiser ---------------------------------------------------------------
BEAMER_LEARNING_RATE = 0.05
WARP_LEARNING_RATE   = 0.001 # Might need tuning
BETA1, BETA2         = 0.9, 0.999
EPS_ADAM             = 1e-8

# --- Two-Phase Optimization ---
PHASE1_ITERATIONS = 25 # Number of iterations for alignment phase

# Loss Weights ------------------------------------------------------------
SSIM_LOSS_W = 0.0   # Weight for Structural Similarity (set to 0 for now)
RGB_LOSS_W  = 1.0   # Weight for RGB MSE loss (Increased significantly)

# -------------------------------------------------------------------------
OUTPUT_FILENAME = "projection_evolution.mp4"
COMPARISON_FILENAME = "comparison_corrected.jpg" # Added for consistency
WINDOW_NAME     = "Projection Mapping Optimiser"

FONT            = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE      = 0.8 # Adjusted for more labels
FONT_THICKNESS  = 2
TEXT_COLOR      = (105, 195, 105)
# ------------------------------------------------------------------------- 
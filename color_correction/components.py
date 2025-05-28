import cv2
import numpy as np
import random
import config # Import config to use constants like INITIAL_BEAMER_OUT_VALUE

# -------------------------------------------------------------------------
# Utility -----------------------------------------------------------------

def composite_frame(images, labels):
    assert len(images) == len(labels)
    processed = [im if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) for im in images]
    # Ensure all images have the same height for hstack
    target_h = processed[0].shape[0]
    processed = [cv2.resize(im, (int(im.shape[1] * target_h / im.shape[0]), target_h)) if im.shape[0] != target_h else im for im in processed]
    # Get consistent width after ensuring height matches first image
    target_w = processed[0].shape[1]
    processed = [cv2.resize(im, (target_w, im.shape[0])) if im.shape[1] != target_w else im for im in processed]

    h, w, _ = processed[0].shape
    canvas = np.hstack(processed)
    for i, lbl in enumerate(labels):
        (tw, th), _ = cv2.getTextSize(lbl, config.FONT, config.FONT_SCALE, config.FONT_THICKNESS)
        x = i * w + (w - tw) // 2
        y = h - 10
        cv2.putText(canvas, lbl, (x, y), config.FONT, config.FONT_SCALE, config.TEXT_COLOR, config.FONT_THICKNESS, cv2.LINE_AA)
    return canvas

def rgb_to_lab(img_rgb_f32):
    """Converts float32 BGR [0,1] image to float32 LAB."""
    return cv2.cvtColor(img_rgb_f32, cv2.COLOR_BGR2LAB)

# -------------------------------------------------------------------------
# Classes -----------------------------------------------------------------

class InputSurface:
    def __init__(self, path):
        img = cv2.imread(path).astype(np.float32) / 255.0
        self.surface = self.resize_max(img, config.TARGET_SIZE)

    @staticmethod
    def resize_max(im, max_size):
        h, w = im.shape[:2]
        s = max_size / max(h, w)
        return cv2.resize(im, (int(w * s), int(h * s)))

    def get(self):
        return self.surface

# -------------------------------------------------------------------------
class BeamerOutLayer:
    """Stores RGB per‑pixel projector intensities."""
    def __init__(self, shape):
        # shape: (H, W, 3)
        self.map = np.full(shape, config.INITIAL_BEAMER_OUT_VALUE, np.float32)

    def update(self, newmap):
        self.map = np.clip(newmap, 0.0, 1.0)

    def get(self):
        return self.map

    def reset(self, value=config.INITIAL_BEAMER_OUT_VALUE):
        """Resets the beamer map to a constant value."""
        self.map.fill(value)

# -------------------------------------------------------------------------
class AdamOptimiser:
    def __init__(self, lr, shape):
        self.lr = lr
        self.shape = shape # Store shape for reset
        self.reset()

    def reset(self):
        """Resets the optimizer state."""
        self.m  = np.zeros(self.shape, np.float32)
        self.v  = np.zeros(self.shape, np.float32)
        self.t  = 0
        print(f"Adam optimizer state reset (shape: {self.shape})")

    def step(self, grad, param):
        self.t += 1
        # Ensure grad has the same shape as param, m, v
        if grad.shape != param.shape:
             # Handle scalar gradient for single param update (e.g. warp params)
             if param.ndim == 1 and grad.size == param.size:
                 grad = grad.reshape(param.shape)
             else:
                raise ValueError(f"Gradient shape {grad.shape} incompatible with parameter shape {param.shape}")

        self.m = config.BETA1 * self.m + (1 - config.BETA1) * grad
        self.v = config.BETA2 * self.v + (1 - config.BETA2) * (grad ** 2)
        m_hat  = self.m / (1 - config.BETA1 ** self.t) if (1 - config.BETA1 ** self.t) != 0 else self.m
        v_hat  = self.v / (1 - config.BETA2 ** self.t) if (1 - config.BETA2 ** self.t) != 0 else self.v
        update = self.lr * m_hat / (np.sqrt(v_hat) + config.EPS_ADAM)
        return param - update

# -------------------------------------------------------------------------
class ProjectionSimulator:
    def __init__(self, surface, ambient): # Removed beamer dependency here
        self.surface = surface  # H×W×3  float32 0‑1
        self.ambient = ambient  # scalar

    def calculate_perceived(self, beamer_output_f32):
        """Calculates the ideal perceived image based on surface, ambient light, and beamer output."""
        # Ensure beamer output is within valid range
        beamer_output_f32 = np.clip(beamer_output_f32, 0.0, 1.0)

        # 1. Additive Light: Sum ambient and beamer light
        total_incident_light = self.ambient + beamer_output_f32

        # 2. Surface Absorption/Reflection: Multiply incident light by surface color
        reflected_light = self.surface * total_incident_light

        # 3. Clip final perceived color to [0, 1]
        return np.clip(reflected_light, 0.0, 1.0)

# -------------------------------------------------------------------------
class Loss:
    def __init__(self, target_rgb_f32):
        self.target_rgb = target_rgb_f32.astype(np.float32)
        self.target_gray = cv2.cvtColor(self.target_rgb, cv2.COLOR_BGR2GRAY)
        
        # Precompute target features for additional objectives
        self.target_lab = cv2.cvtColor(self.target_rgb, cv2.COLOR_BGR2LAB)
        self.target_edges = cv2.Canny((self.target_gray * 255).astype(np.uint8), 50, 150)
        
        # Compute target histogram for color distribution matching
        self.target_hist_b = cv2.calcHist([self.target_rgb], [0], None, [32], [0, 1])
        self.target_hist_g = cv2.calcHist([self.target_rgb], [1], None, [32], [0, 1])
        self.target_hist_r = cv2.calcHist([self.target_rgb], [2], None, [32], [0, 1])

    def compute_rgb_loss(self, perceived_rgb_f32, target_rgb=None):
        """Calculates the L2 loss (MSE) in RGB space."""
        if target_rgb is None:
            target_rgb = self.target_rgb
        diff_rgb = perceived_rgb_f32 - target_rgb
        rgb_loss = np.mean(diff_rgb**2)
        return rgb_loss
    
    def compute_lab_loss(self, perceived_rgb_f32, target_rgb=None):
        """Calculates L2 loss in LAB color space for better perceptual matching."""
        perceived_lab = cv2.cvtColor(perceived_rgb_f32, cv2.COLOR_BGR2LAB)
        if target_rgb is None:
            target_lab = self.target_lab
        else:
            target_lab = cv2.cvtColor(target_rgb, cv2.COLOR_BGR2LAB)
        diff_lab = perceived_lab - target_lab
        lab_loss = np.mean(diff_lab**2)
        return lab_loss
    
    def compute_edge_loss(self, perceived_rgb_f32):
        """Calculates edge-based loss to preserve structural features."""
        perceived_gray = cv2.cvtColor(perceived_rgb_f32, cv2.COLOR_BGR2GRAY)
        perceived_edges = cv2.Canny((perceived_gray * 255).astype(np.uint8), 50, 150)
        
        # Normalize to [0,1] for comparison
        target_edges_norm = self.target_edges.astype(np.float32) / 255.0
        perceived_edges_norm = perceived_edges.astype(np.float32) / 255.0
        
        edge_loss = np.mean((perceived_edges_norm - target_edges_norm)**2)
        return edge_loss
    
    def compute_histogram_loss(self, perceived_rgb_f32):
        """Calculates histogram-based loss for color distribution matching."""
        # Compute histograms for perceived image
        perceived_hist_b = cv2.calcHist([perceived_rgb_f32], [0], None, [32], [0, 1])
        perceived_hist_g = cv2.calcHist([perceived_rgb_f32], [1], None, [32], [0, 1])
        perceived_hist_r = cv2.calcHist([perceived_rgb_f32], [2], None, [32], [0, 1])
        
        # Normalize histograms
        perceived_hist_b = perceived_hist_b / (perceived_hist_b.sum() + 1e-8)
        perceived_hist_g = perceived_hist_g / (perceived_hist_g.sum() + 1e-8)
        perceived_hist_r = perceived_hist_r / (perceived_hist_r.sum() + 1e-8)
        
        target_hist_b_norm = self.target_hist_b / (self.target_hist_b.sum() + 1e-8)
        target_hist_g_norm = self.target_hist_g / (self.target_hist_g.sum() + 1e-8)
        target_hist_r_norm = self.target_hist_r / (self.target_hist_r.sum() + 1e-8)
        
        # Use Chi-squared distance for histogram comparison
        hist_loss_b = np.sum((perceived_hist_b - target_hist_b_norm)**2 / (target_hist_b_norm + 1e-8))
        hist_loss_g = np.sum((perceived_hist_g - target_hist_g_norm)**2 / (target_hist_g_norm + 1e-8))
        hist_loss_r = np.sum((perceived_hist_r - target_hist_r_norm)**2 / (target_hist_r_norm + 1e-8))
        
        hist_loss = (hist_loss_b + hist_loss_g + hist_loss_r) / 3.0
        return hist_loss
    
    def compute_gradient_magnitude_loss(self, perceived_rgb_f32):
        """Computes gradient magnitude loss to preserve image sharpness."""
        # Convert to grayscale for gradient computation
        perceived_gray = cv2.cvtColor(perceived_rgb_f32, cv2.COLOR_BGR2GRAY)
        
        # Compute gradients
        grad_x_perceived = cv2.Sobel(perceived_gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y_perceived = cv2.Sobel(perceived_gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag_perceived = np.sqrt(grad_x_perceived**2 + grad_y_perceived**2)
        
        grad_x_target = cv2.Sobel(self.target_gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y_target = cv2.Sobel(self.target_gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag_target = np.sqrt(grad_x_target**2 + grad_y_target**2)
        
        grad_loss = np.mean((grad_mag_perceived - grad_mag_target)**2)
        return grad_loss

    def apply_blur(self, image):
        """Applies Gaussian blur to an image for loss smoothing."""
        if config.PHASE1_BLUR_SIGMA > 0:
            return cv2.GaussianBlur(image, (config.PHASE1_BLUR_KERNEL_SIZE, config.PHASE1_BLUR_KERNEL_SIZE), 
                                  config.PHASE1_BLUR_SIGMA)
        return image

    def compute_rgb_gradient_wrt_input(self, perceived_rgb_f32):
        """Calculates the analytical gradient of RGB MSE loss w.r.t perceived_rgb."""
        diff_rgb = perceived_rgb_f32 - self.target_rgb
        N = perceived_rgb_f32.size # Total number of elements H*W*3 
        grad_rgb = 2.0 * diff_rgb / N
        return grad_rgb

    def compute_loss_components(self, perceived_rgb_f32, use_blur=False):
        """Computes multi-objective loss components for robust optimization."""
        perceived_rgb_f32 = perceived_rgb_f32.astype(np.float32)
        
        # Apply blur if requested (for Phase 1 smoothing)
        if use_blur:
            perceived_rgb_f32 = self.apply_blur(perceived_rgb_f32)
            target_rgb_for_loss = self.apply_blur(self.target_rgb)
        else:
            target_rgb_for_loss = self.target_rgb
        
        # Primary RGB loss
        rgb_L = 0.0
        if config.RGB_LOSS_W > 0:
            rgb_L = self.compute_rgb_loss(perceived_rgb_f32, target_rgb_for_loss)
        
        # Additional objective: LAB color space loss for better perceptual matching
        lab_L = 0.0
        if hasattr(config, 'LAB_LOSS_W') and config.LAB_LOSS_W > 0:
            lab_L = self.compute_lab_loss(perceived_rgb_f32, target_rgb_for_loss)
        
        # Additional objective: Edge preservation
        edge_L = 0.0
        if hasattr(config, 'EDGE_LOSS_W') and config.EDGE_LOSS_W > 0:
            edge_L = self.compute_edge_loss(perceived_rgb_f32)
        
        # Additional objective: Histogram matching
        hist_L = 0.0
        if hasattr(config, 'HIST_LOSS_W') and config.HIST_LOSS_W > 0:
            hist_L = self.compute_histogram_loss(perceived_rgb_f32)
        
        # Additional objective: Gradient magnitude preservation
        grad_L = 0.0
        if hasattr(config, 'GRAD_LOSS_W') and config.GRAD_LOSS_W > 0:
            grad_L = self.compute_gradient_magnitude_loss(perceived_rgb_f32)

        # Legacy SSIM placeholder
        ssim_L = 0.0

        # Combine all losses with weights
        rgb_weight = getattr(config, 'RGB_LOSS_W', 1.0)
        lab_weight = getattr(config, 'LAB_LOSS_W', 0.0)
        edge_weight = getattr(config, 'EDGE_LOSS_W', 0.0)
        hist_weight = getattr(config, 'HIST_LOSS_W', 0.0)
        grad_weight = getattr(config, 'GRAD_LOSS_W', 0.0)
        ssim_weight = getattr(config, 'SSIM_LOSS_W', 0.0)
        
        total_L = (rgb_weight * rgb_L + 
                  lab_weight * lab_L + 
                  edge_weight * edge_L + 
                  hist_weight * hist_L + 
                  grad_weight * grad_L + 
                  ssim_weight * ssim_L)
        
        return total_L, rgb_L, ssim_L
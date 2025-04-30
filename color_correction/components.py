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

    def compute_rgb_loss(self, perceived_rgb_f32):
        """Calculates the L2 loss (MSE) in RGB space."""
        diff_rgb = perceived_rgb_f32 - self.target_rgb
        rgb_loss = np.mean(diff_rgb**2)
        return rgb_loss

    def compute_rgb_gradient_wrt_input(self, perceived_rgb_f32):
        """Calculates the analytical gradient of RGB MSE loss w.r.t perceived_rgb."""
        diff_rgb = perceived_rgb_f32 - self.target_rgb
        N = perceived_rgb_f32.size # Total number of elements H*W*3 
        grad_rgb = 2.0 * diff_rgb / N
        return grad_rgb

    def compute_loss_components(self, perceived_rgb_f32):
        """Computes loss value(s) based on the input perceived image."""
        perceived_rgb_f32 = perceived_rgb_f32.astype(np.float32)
        rgb_L = 0.0
        if config.RGB_LOSS_W > 0:
            rgb_L = self.compute_rgb_loss(perceived_rgb_f32)

        ssim_L = 0.0

        total_L = config.RGB_LOSS_W * rgb_L + config.SSIM_LOSS_W * ssim_L
        return total_L, rgb_L, ssim_L
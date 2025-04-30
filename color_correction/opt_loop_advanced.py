import cv2
import numpy as np
import time

# -------------------------------------------------------------------------
# Hyper‑parameters ---------------------------------------------------------
TARGET_SIZE = 1024          # long edge of working resolution
FPS         = 10
DT          = 1.0 / FPS

INITIAL_BEAMER_OUT_VALUE = 0.25  # initial RGB value 0‑1
AMBIENT_LIGHT_STRENGTH   = 0.2   # scalar added to each channel

# Optimiser ---------------------------------------------------------------
LEARNING_RATE = 0.05
BETA1, BETA2  = 0.9, 0.999
EPS_ADAM      = 1e-8
GRAD_EPSILON  = 1e-4 # Epsilon for finite difference gradient calculation

# Loss Weights ------------------------------------------------------------
SSIM_LOSS_W = 0.0   # Weight for Structural Similarity (set to 0 for now)
RGB_LOSS_W  = 1.0   # Weight for RGB MSE loss (Increased significantly)

# -------------------------------------------------------------------------
OUTPUT_FILENAME = "projection_evolution.mp4"
WINDOW_NAME     = "Projection Mapping Optimiser"

FONT            = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE      = 1.0
FONT_THICKNESS  = 2
TEXT_COLOR      = (105, 195, 105)

# -------------------------------------------------------------------------
# Utility -----------------------------------------------------------------

def composite_frame(images, labels):
    assert len(images) == len(labels)
    processed = [im if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) for im in images]
    h, w, _ = processed[0].shape
    canvas = np.hstack(processed)
    for i, lbl in enumerate(labels):
        (tw, th), _ = cv2.getTextSize(lbl, FONT, FONT_SCALE, FONT_THICKNESS)
        x = i * w + (w - tw) // 2
        y = h - 10
        cv2.putText(canvas, lbl, (x, y), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
    return canvas

# -------------------------------------------------------------------------
class InputSurface:
    def __init__(self, path):
        img = cv2.imread(path).astype(np.float32) / 255.0
        self.surface = self.resize_max(img, TARGET_SIZE)

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
        self.map = np.full(shape, INITIAL_BEAMER_OUT_VALUE, np.float32)

    def update(self, newmap):
        self.map = np.clip(newmap, 0.0, 1.0)

    def get(self):
        return self.map

# -------------------------------------------------------------------------
class AdamOptimiser:
    def __init__(self, lr, shape):
        self.lr = lr
        self.m  = np.zeros(shape, np.float32)
        self.v  = np.zeros(shape, np.float32)
        self.t  = 0

    def step(self, grad, param):
        self.t += 1
        self.m = BETA1 * self.m + (1 - BETA1) * grad
        self.v = BETA2 * self.v + (1 - BETA2) * (grad ** 2)
        m_hat  = self.m / (1 - BETA1 ** self.t)
        v_hat  = self.v / (1 - BETA2 ** self.t)
        return param - self.lr * m_hat / (np.sqrt(v_hat) + EPS_ADAM)

# -------------------------------------------------------------------------
class ProjectionSimulator:
    def __init__(self, surface, beamer, ambient):
        self.surface = surface  # H×W×3  float32 0‑1
        self.beamer  = beamer   # BeamerOutLayer
        self.ambient = ambient  # scalar

    def calculate_perceived(self, beamer_output_f32):
        """Calculates the perceived image based on surface, ambient light, and beamer output."""
        # Ensure beamer output is within valid range
        beamer_output_f32 = np.clip(beamer_output_f32, 0.0, 1.0)

        # 1. Additive Light: Sum ambient and beamer light
        total_incident_light = self.ambient + beamer_output_f32

        # 2. Surface Absorption/Reflection: Multiply incident light by surface color
        reflected_light = self.surface * total_incident_light

        # 3. Clip final perceived color to [0, 1]
        return np.clip(reflected_light, 0.0, 1.0)

    def render(self):
        """Renders the current perceived image using the beamer's state."""
        return self.calculate_perceived(self.beamer.get())

# -------------------------------------------------------------------------

def rgb_to_lab(img_rgb_f32):
    """Converts float32 BGR [0,1] image to float32 LAB."""
    # cv2.cvtColor expects float32 BGR [0, 1]
    return cv2.cvtColor(img_rgb_f32, cv2.COLOR_BGR2LAB)

class Loss:
    def __init__(self, target_rgb_f32):
        # Store target in BGR float32 [0,1] format
        self.target_rgb = target_rgb_f32.astype(np.float32)
        # Precompute target in LAB space
        self.target_lab = rgb_to_lab(self.target_rgb)
        # Precompute target grayscale (for potential SSIM later)
        self.target_gray = cv2.cvtColor(self.target_rgb, cv2.COLOR_BGR2GRAY)

    def compute_rgb_loss(self, perceived_rgb_f32):
        """Calculates the L2 loss (MSE) in RGB space."""
        diff_rgb = perceived_rgb_f32 - self.target_rgb
        rgb_loss = np.mean(diff_rgb**2)
        return rgb_loss

    def compute_rgb_gradient(self, perceived_rgb_f32):
        """Calculates the analytical gradient of RGB MSE loss w.r.t perceived_rgb."""
        diff_rgb = perceived_rgb_f32 - self.target_rgb
        N = perceived_rgb_f32.size # Total number of elements H*W*3
        grad_perceived_rgb = 2.0 * diff_rgb / N
        return grad_perceived_rgb

    def compute(self, perceived_rgb_f32, surface):
        # Ensure input is float32
        perceived_rgb_f32 = perceived_rgb_f32.astype(np.float32)

        # --- RGB MSE Loss Term ---
        if RGB_LOSS_W > 0:
            rgb_L = self.compute_rgb_loss(perceived_rgb_f32)
            grad_p_rgb_rgb = self.compute_rgb_gradient(perceived_rgb_f32)
        else:
            rgb_L = 0.0
            grad_p_rgb_rgb = np.zeros_like(perceived_rgb_f32)

        # --- SSIM Loss Term (Placeholder) ---
        ssim_L = 0.0 # No SSIM calculation yet
        # TODO: Implement SSIM calculation and gradient if SSIM_LOSS_W > 0
        # Example using scikit-image (requires pip install scikit-image):
        # from skimage.metrics import structural_similarity
        # perceived_gray = cv2.cvtColor(perceived_rgb_f32, cv2.COLOR_BGR2GRAY)
        # ssim_score, ssim_grad_img = structural_similarity(perceived_gray, self.target_gray, gradient=True, data_range=1.0)
        # ssim_L = (1.0 - ssim_score) / 2.0
        # Need to derive grad_p_rgb_ssim from ssim_grad_img (grayscale gradient)
        grad_p_rgb_ssim = np.zeros_like(perceived_rgb_f32)

        # --- Combine ---
        total_L = RGB_LOSS_W * rgb_L + SSIM_LOSS_W * ssim_L

        # Combine gradients (apply chain rule: dLoss/dBeamer = dLoss/dPerceived * dPerceived/dBeamer)
        # dPerceived/dBeamer = surface (where 0 < perceived < 1)
        grad_p_rgb = (RGB_LOSS_W * grad_p_rgb_rgb +
                      SSIM_LOSS_W * grad_p_rgb_ssim)

        # Apply clipping mask: gradient is zero where perceived image was clipped
        clip_mask = (perceived_rgb_f32 > 0.0) & (perceived_rgb_f32 < 1.0)

        # --- Calculate gradient w.r.t. Beamer map --- 
        # Apply chain rule: dLoss/dBeamer = dLoss/dPerceived * dPerceived/dBeamer
        # where dPerceived/dBeamer = surface (approximately, where not clipped)
        grad_bmr = (grad_p_rgb * surface * clip_mask).astype(np.float32) # Element-wise multiplication

        # --- Debug: Print gradient magnitudes ---
        if RGB_LOSS_W > 0:
             print(f"  Grad Mag (RGB wrt Perceived): {np.mean(np.abs(grad_p_rgb_rgb)):.2e}")
        print(f"  Grad Mag (Final wrt Beamer):  {np.mean(np.abs(grad_bmr)):.2e}")
        # --------------------------------------

        # Return total loss, individual components, and gradient for beamer
        return total_L, rgb_L, ssim_L, grad_bmr

# -------------------------------------------------------------------------

def main(surface_path, target_path):
    surf = InputSurface(surface_path).get()  # H×W×3 float32
    H, W, _ = surf.shape

    beamer = BeamerOutLayer((H, W, 3))
    sim    = ProjectionSimulator(surf, beamer, AMBIENT_LIGHT_STRENGTH)

    # Load target, resize, convert to float32 [0, 1]
    target_bgr_u8 = cv2.imread(target_path)
    target_bgr_f32 = cv2.resize(target_bgr_u8, (W, H)).astype(np.float32) / 255.0

    lossfn = Loss(target_bgr_f32)
    opt    = AdamOptimiser(LEARNING_RATE, (H, W, 3))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    writer = None
    last_perceived = None # Store the last perceived frame

    while True:
        t0 = time.time()
        # Render perceived image (float32 [0, 1])
        perceived = sim.render()

        # Compute loss and gradient
        loss, rgb_L, ssim_L, grad = lossfn.compute(perceived, surf)

        # Update beamer using optimizer
        beamer.update(opt.step(grad, beamer.get()))

        last_perceived = perceived # Update last perceived frame

        # Print loss components
        # Only show SSIM loss if its weight is non-zero
        loss_str = f"Step {opt.t}: Total Loss: {loss:.4f}"
        if RGB_LOSS_W > 0:
            loss_str += f", RGB Loss: {rgb_L:.4f}"
        if SSIM_LOSS_W > 0:
            loss_str += f", SSIM Loss: {ssim_L:.4f}"
        print(loss_str)


        # --- Visualization ---
        # Convert float [0,1] images to uint8 [0,255] for display/writing
        vis_surf = (surf * 255).astype(np.uint8)
        vis_beamer = (beamer.get() * 255).astype(np.uint8)
        vis_perceived = (perceived * 255).astype(np.uint8)
        vis_target = (target_bgr_f32 * 255).astype(np.uint8) # Use the float target we loaded

        vis = composite_frame([
            vis_surf, vis_beamer, vis_perceived, vis_target
        ], ["Surface", "Beamer RGB", "Perceived", "Target"])
        cv2.imshow(WINDOW_NAME, vis)

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(OUTPUT_FILENAME, fourcc, FPS, (vis.shape[1], vis.shape[0]))
        writer.write(vis)

        key = cv2.waitKey(1)
        if key in (27, ord('q')):
            break

        dt = time.time() - t0
        if dt < DT:
            time.sleep(DT - dt)

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    # --- Final Comparison Image ---
    if last_perceived is not None:
        # Generate default perceived view (using target *as* beamer output for comparison)
        default_beamer_out = target_bgr_f32
        # Use the simulator's calculation method
        default_perceived = sim.calculate_perceived(default_beamer_out)

        # Get final beamer output
        final_beamer_map = beamer.get()

        # Convert float [0,1] images to uint8 [0,255] for saving
        vis_final_beamer = (final_beamer_map * 255).astype(np.uint8)
        vis_default_perceived = (default_perceived * 255).astype(np.uint8)
        vis_last_perceived = (last_perceived * 255).astype(np.uint8)
        vis_target = (target_bgr_f32 * 255).astype(np.uint8)

        comparison_img = composite_frame([
            vis_final_beamer,
            vis_default_perceived,
            vis_last_perceived,
            vis_target
        ], [
            "Raw Beamer",
            "Default Perceived",
            "Optimized Perceived",
            "Target"
        ])

        comparison_filename = "comparison.jpg"
        cv2.imwrite(comparison_filename, comparison_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"Saved final comparison image to {comparison_filename}")

if __name__ == "__main__":
    # Ensure you have surface.jpg and target.jpg in the same directory
    # or provide correct paths.
    main('../assets/surface.jpg', '../assets/target.jpg')

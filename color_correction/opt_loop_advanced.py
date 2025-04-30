import time
import cv2
import numpy as np
import random

# Import constants and components from separate files
from config import *
from components import (
    InputSurface, BeamerOutLayer, AdamOptimiser,
    ProjectionSimulator, Loss, composite_frame
)

# -------------------------------------------------------------------------
class SpatialWarp:
    """Stores and manages optimizable affine warp parameters."""
    def __init__(self):
        self.theta = 0.0 # degrees
        self.tx = 0.0    # pixels
        self.ty = 0.0    # pixels

    def get_params(self):
        return np.array([self.theta, self.tx, self.ty], dtype=np.float32)

    def update(self, params):
        self.theta, self.tx, self.ty = params[0], params[1], params[2]

    def get_matrix(self, center_xy):
        """Calculates the 2x3 affine warp matrix."""
        rot_mat = cv2.getRotationMatrix2D(center_xy, self.theta, 1.0)
        # Combine rotation and translation
        rot_mat[0, 2] += self.tx
        rot_mat[1, 2] += self.ty
        return rot_mat

    @staticmethod
    def get_matrix_from_params(theta, tx, ty, center_xy):
        """Static method to get matrix from explicit parameters."""
        rot_mat = cv2.getRotationMatrix2D(center_xy, theta, 1.0)
        rot_mat[0, 2] += tx
        rot_mat[1, 2] += ty
        return rot_mat

# -------------------------------------------------------------------------
# New Class: SimulatedCamera ----------------------------------------------
class SimulatedCamera:
    """Simulates a camera capturing a projection, including fixed misalignment."""
    def __init__(self, surf, ambient_light_strength, max_rotation, max_shift, width, height, center_xy):
        self.W, self.H = width, height
        self.img_center = center_xy
        # --- Internal simulator for ideal perception ---
        self.simulator = ProjectionSimulator(surf, ambient_light_strength)

        # --- Simulate initial misalignment (Calculated ONCE) ---
        misalign_theta = random.uniform(-max_rotation, max_rotation)
        misalign_tx    = random.uniform(-max_shift, max_shift) * self.W
        misalign_ty    = random.uniform(-max_shift, max_shift) * self.H
        self.M_misalign = SpatialWarp.get_matrix_from_params(
            misalign_theta, misalign_tx, misalign_ty, self.img_center
        )
        print(f"Simulated Camera Misalignment: θ={misalign_theta:.2f}°, tx={misalign_tx:.1f}px, ty={misalign_ty:.1f}px")

    def capture(self, beamer_map):
        """Simulates projection and captures with fixed misalignment."""
        # 1. Calculate the ideal perceived image
        ideal_perceived_image = self.simulator.calculate_perceived(beamer_map)
        # 2. Apply the fixed misalignment warp
        return cv2.warpAffine(
            ideal_perceived_image, self.M_misalign, (self.W, self.H),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0)
        )

# -------------------------------------------------------------------------
# New Function: Optimize Warp Step ----------------------------------------
def optimize_warp_step(spatial_warp, warp_opt, lossfn, corrected_perceived, misaligned_perceived, img_center, current_warp_params):
    """Calculates warp gradients and performs one optimization step."""
    H, W, _ = corrected_perceived.shape
    grad_corrected_perceived = lossfn.compute_rgb_gradient_wrt_input(corrected_perceived)

    # Calculate spatial gradients of misaligned_perceived
    misaligned_f64 = misaligned_perceived.astype(np.float64)
    grad_misaligned_x = cv2.Sobel(misaligned_f64, cv2.CV_64F, 1, 0, ksize=3).astype(np.float32)
    grad_misaligned_y = cv2.Sobel(misaligned_f64, cv2.CV_64F, 0, 1, ksize=3).astype(np.float32)

    # Get warp parameters and calculate Jacobian components
    theta_rad = np.radians(current_warp_params[0])
    cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
    cx, cy = img_center
    u_coords, v_coords = np.meshgrid(np.arange(W), np.arange(H))

    dx_dTheta_x = -(u_coords - cx) * sin_t - (v_coords - cy) * cos_t
    dx_dTheta_y =  (u_coords - cx) * cos_t - (v_coords - cy) * sin_t

    # Chain rule calculation
    dL_dCorrected_sum = np.sum(grad_corrected_perceived, axis=2)
    grad_misaligned_x_sum = np.sum(grad_misaligned_x, axis=2)
    grad_misaligned_y_sum = np.sum(grad_misaligned_y, axis=2)

    warp_grad = np.zeros(3, dtype=np.float32)

    # Gradient w.r.t tx
    grad_tx_map = dL_dCorrected_sum * grad_misaligned_x_sum
    warp_grad[1] = np.mean(grad_tx_map)

    # Gradient w.r.t ty
    grad_ty_map = dL_dCorrected_sum * grad_misaligned_y_sum
    warp_grad[2] = np.mean(grad_ty_map)

    # Gradient w.r.t theta
    grad_theta_map = dL_dCorrected_sum * (grad_misaligned_x_sum * dx_dTheta_x + grad_misaligned_y_sum * dx_dTheta_y)
    dL_dTheta_rad = np.mean(grad_theta_map)
    warp_grad[0] = dL_dTheta_rad * (np.pi / 180.0) # Convert to degree gradient

    # Update Warp parameters
    updated_params = warp_opt.step(warp_grad, current_warp_params)
    spatial_warp.update(updated_params)

    return warp_grad # Return the calculated gradient for logging

# -------------------------------------------------------------------------

def main(surface_path, target_path):
    surf = InputSurface(surface_path).get()  # H×W×3 float32
    H, W, _ = surf.shape
    img_center = (W // 2, H // 2)

    # Beamer layer (starts with initial value, optimized in phase 2)
    beamer = BeamerOutLayer((H, W, 3))
    # Simulated Camera with fixed misalignment and internal projection simulator
    camera = SimulatedCamera(surf, AMBIENT_LIGHT_STRENGTH, MAX_ROTATION, MAX_SHIFT, W, H, img_center)

    # Load target, resize, convert to float32 [0, 1]
    target_bgr_f32 = cv2.resize(cv2.imread(target_path), (W, H)).astype(np.float32) / 255.0

    lossfn = Loss(target_bgr_f32)
    color_opt = AdamOptimiser(BEAMER_LEARNING_RATE, (H, W, 3))
    warp_opt   = AdamOptimiser(WARP_LEARNING_RATE, (3,)) # Optimizer for theta, tx, ty
    spatial_warp = SpatialWarp() # Holds the optimizable warp parameters

    # --- Optimization Phase Control ---
    current_phase = 1 # Start with phase 1 (Alignment)
    phase1_step = 0
    final_M_corrective = None # Store the corrective warp from phase 1
    total_steps = 0
    # ---

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    writer = None
    last_corrected_perceived = None # Store the final corrected frame

    while True:
        total_steps += 1
        t_start_iter = time.time()

        # --- Phase-Dependent Setup ---
        if current_phase == 1:
            # Phase 1: Align geometry. Use TARGET as fixed beamer output.
            current_beamer_map = target_bgr_f32 # Fixed
            current_warp_params = spatial_warp.get_params() # Optimizable
            M_corrective = spatial_warp.get_matrix(img_center) # Optimizable
        else: # current_phase == 2
            # Phase 2: Correct color. Use optimizable beamer output, fixed warp.
            current_beamer_map = beamer.get() # Optimizable
            if final_M_corrective is None:
                # Should not happen if phase transition logic is correct
                print("ERROR: final_M_corrective is None in Phase 2!")
                final_M_corrective = spatial_warp.get_matrix(img_center) # Fallback
            M_corrective = final_M_corrective # Fixed
            current_warp_params = spatial_warp.get_params() # Fixed (though read for info)

        # --- Forward Pass ---
        # 1. Simulate camera capture (includes projection and misalignment)
        misaligned_perceived = camera.capture(current_beamer_map) # Takes beamer map

        # 2. Apply corrective warp (Optimizable in P1, Fixed in P2)
        corrected_perceived = cv2.warpAffine(misaligned_perceived, M_corrective, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) # Use replicate border for correction

        # --- Compute Loss ---
        loss, rgb_L, ssim_L = lossfn.compute_loss_components(corrected_perceived)

        # --- Compute Gradients & Update (Phase Dependent) ---
        grad_bmr = np.zeros_like(current_beamer_map)
        warp_grad = np.zeros(3, dtype=np.float32)

        if current_phase == 1:
            # Phase 1: Optimize Warp Only
            warp_grad = optimize_warp_step(
                spatial_warp, warp_opt, lossfn, corrected_perceived,
                misaligned_perceived, img_center, current_warp_params
            )

            # Phase Transition Check
            phase1_step += 1
            if phase1_step >= PHASE1_ITERATIONS:
                print(f"\n--- Phase 1 Complete ({phase1_step} iterations) ---")
                print(f"Final Warp: θ={spatial_warp.theta:.2f}, tx={spatial_warp.tx:.1f}, ty={spatial_warp.ty:.1f}")
                final_M_corrective = spatial_warp.get_matrix(img_center) # Store final warp
                current_phase = 2
                # Reset beamer map to initial value and reset its optimizer
                beamer.reset()
                color_opt.reset()
                print("--- Starting Phase 2: Color Optimization ---\n")


        else: # current_phase == 2
            # Phase 2: Optimize Beamer Only
            grad_corrected_perceived = lossfn.compute_rgb_gradient_wrt_input(corrected_perceived)

            # Approximation: Ignore warp Jacobians for beamer gradient path
            # dL/dBeamer ~ dL/dCorrected * surf
            beamer_clip_mask = (current_beamer_map > 0.0) & (current_beamer_map < 1.0)
            grad_bmr = (grad_corrected_perceived * surf * beamer_clip_mask).astype(np.float32)

            # Update Beamer
            beamer.update(color_opt.step(grad_bmr, current_beamer_map))

        last_corrected_perceived = corrected_perceived # Store for final image

        # Print info
        phase_str = f"P{current_phase}"
        step_str = f"Step {total_steps}" if current_phase == 2 else f"Step {phase1_step}/{PHASE1_ITERATIONS}"
        loss_str = f"{phase_str} {step_str}: L={loss:.4f}"
        if RGB_LOSS_W > 0: loss_str += f" RGB={rgb_L:.4f}"
        if SSIM_LOSS_W > 0: loss_str += f" SSIM={ssim_L:.4f}"
        warp_str = f"Warp: θ={spatial_warp.theta:.2f}, tx={spatial_warp.tx:.1f}, ty={spatial_warp.ty:.1f}"
        # Show relevant gradient based on phase
        if current_phase == 1:
            grad_str = f"G_Warp=[{warp_grad[0]:.2e}, {warp_grad[1]:.2e}, {warp_grad[2]:.2e}]"
        else:
            grad_str = f"G_Bmr={np.mean(np.abs(grad_bmr)):.2e}"

        print(f"{loss_str} | {warp_str} | {grad_str}")


        # --- Visualization ---
        vis_surf = (surf * 255).astype(np.uint8)
        # Display the beamer map being USED in the current phase
        vis_beamer = (current_beamer_map * 255).astype(np.uint8)
        vis_misaligned = (np.clip(misaligned_perceived, 0, 1) * 255).astype(np.uint8)
        vis_corrected = (np.clip(corrected_perceived, 0, 1) * 255).astype(np.uint8)
        vis_target = (target_bgr_f32 * 255).astype(np.uint8)

        # Label beamer differently in phase 1
        beamer_label = "Beamer (Target Fixed)" if current_phase == 1 else "Beamer (Optimizing)"

        vis = composite_frame([
            vis_surf, vis_beamer, vis_misaligned, vis_corrected, vis_target
        ], ["Surface", beamer_label, "Misaligned", "Corrected", "Target"])
        cv2.imshow(WINDOW_NAME, vis)

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(OUTPUT_FILENAME, fourcc, FPS, (vis.shape[1], vis.shape[0]))
        writer.write(vis)

        key = cv2.waitKey(1)
        if key in (27, ord('q')):
            break

        iter_dt = time.time() - t_start_iter
        if iter_dt < DT:
            time.sleep(DT - iter_dt)

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    # --- Final Comparison Image ---
    if last_corrected_perceived is not None:
        # Generate default perceived view (using target *as* beamer output for comparison)
        # Apply the original misalignment to simulate the "uncorrected" view
        default_beamer_out = target_bgr_f32 # Projecting the target directly
        # Use the camera simulation directly
        default_misaligned_perceived = camera.capture(default_beamer_out)

        # Get final beamer output (state after phase 2)
        final_beamer_map = beamer.get()

        # Convert float [0,1] images to uint8 [0,255] for saving
        vis_final_beamer = (final_beamer_map * 255).astype(np.uint8)
        vis_default_misaligned = (np.clip(default_misaligned_perceived, 0, 1) * 255).astype(np.uint8)
        vis_last_corrected = (np.clip(last_corrected_perceived, 0, 1) * 255).astype(np.uint8)
        vis_target = (target_bgr_f32 * 255).astype(np.uint8)

        comparison_img = composite_frame([
            vis_final_beamer,
            vis_default_misaligned,
            vis_last_corrected,
            vis_target
        ], [
            "Final Beamer",
            "Default (Misaligned)", # Show what happens if you just project target with misalignment
            "Optimized (Corrected)",
            "Target"
        ])

        cv2.imwrite(config.COMPARISON_FILENAME, comparison_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"Saved final comparison image to {config.COMPARISON_FILENAME}")

if __name__ == "__main__":
    main('../assets/surface.jpg', '../assets/target.jpg')
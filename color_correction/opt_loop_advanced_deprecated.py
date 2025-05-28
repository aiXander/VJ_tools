import time
import cv2
import numpy as np
import random
import math

from config import *
from components import (
    InputSurface, BeamerOutLayer, AdamOptimiser,
    ProjectionSimulator, Loss, composite_frame
)
from cmaes import CMAESOptimizer
from spatial_warp import SpatialWarp # Import from the new file

# -------------------------------------------------------------------------
# New Class: SimulatedCamera ----------------------------------------------
class SimulatedCamera:
    """Simulates a camera capturing a projection, including fixed misalignment."""
    def __init__(self, surf, ambient_light_strength, max_rotation, max_shift_factor, width, height, center_xy):
        self.W, self.H = width, height
        self.img_center = center_xy
        self.max_shift_factor = max_shift_factor # Store max_shift

        # Calculate padding needed to avoid black borders after warp
        # Use ceil for safety and a factor of 1.5 for rotation margin
        self.pad_x = math.ceil(self.W * self.max_shift_factor * 1.5)
        self.pad_y = math.ceil(self.H * self.max_shift_factor * 1.5)

        # --- Internal simulator for ideal perception ---
        self.simulator = ProjectionSimulator(surf, ambient_light_strength)

        # --- Simulate initial misalignment (Calculated ONCE using original center) ---
        misalign_theta = random.uniform(-max_rotation, max_rotation)
        misalign_tx    = random.uniform(-max_shift_factor, max_shift_factor) * self.W
        misalign_ty    = random.uniform(-max_shift_factor, max_shift_factor) * self.H
        # M_misalign is calculated based on the *original* image center
        self.M_misalign = SpatialWarp.get_matrix_from_params(
            misalign_theta, misalign_tx, misalign_ty, self.img_center
        )
        print(f"Simulated Camera Misalignment: θ={misalign_theta:.2f}°, tx={misalign_tx:.1f}px, ty={misalign_ty:.1f}px")
        print(f"Calculated Padding: pad_x={self.pad_x}px, pad_y={self.pad_y}px")

    def capture(self, beamer_map):
        """Simulates projection, pads, warps on larger canvas, and crops."""
        # 1. Calculate the ideal perceived image
        ideal_perceived_image = self.simulator.calculate_perceived(beamer_map) # Shape (H, W, 3)

        # 2. Pad the ideal image
        padded_H = self.H + 2 * self.pad_y
        padded_W = self.W + 2 * self.pad_x
        padded_ideal = cv2.copyMakeBorder(
            ideal_perceived_image,
            self.pad_y, self.pad_y, self.pad_x, self.pad_x,
            cv2.BORDER_REPLICATE # Replicate edges to avoid introducing new colors
        )

        # 3. Apply the fixed misalignment warp to the PADDED image
        #    Use the original M_misalign (calculated with original center)
        #    Output size is the padded size
        warped_padded = cv2.warpAffine(
            ideal_perceived_image, self.M_misalign, (padded_W, padded_H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0) # Keep original border mode
            # Using BORDER_REPLICATE here might be alternative if black edges persist
        )

        # 4. Crop the center (original HxW) region from the warped padded image
        center_y, center_x = padded_H // 2, padded_W // 2
        start_y = self.pad_y # Equivalent to center_y - self.H // 2 if H is even
        start_x = self.pad_x # Equivalent to center_x - self.W // 2 if W is even
        end_y = start_y + self.H
        end_x = start_x + self.W

        cropped_warped = warped_padded[start_y:end_y, start_x:end_x]

        return cropped_warped

def prep_target(target, W, H):
    if not target:
        target = 0.5

    if isinstance(target, str):
        target_bgr_f32 = cv2.resize(cv2.imread(target), (W, H)).astype(np.float32) / 255.0
    else: # Make a flat color image with the target color:
        target_bgr_f32 = np.ones((H, W, 3), dtype=np.float32) * target
    return target_bgr_f32

def main(surface_path, target):
    surf = InputSurface(surface_path).get()  # H×W×3 float32
    H, W, _ = surf.shape
    img_center = (W // 2, H // 2)

    # Beamer layer (starts with initial value, optimized in phase 2)
    beamer = BeamerOutLayer((H, W, 3))
    # Simulated Camera with fixed misalignment and internal projection simulator
    camera = SimulatedCamera(surf, AMBIENT_LIGHT_STRENGTH, MAX_ROTATION, MAX_SHIFT, W, H, img_center)

    target_bgr_f32 = prep_target(target, W, H)

    lossfn = Loss(target_bgr_f32)
    color_opt = AdamOptimiser(BEAMER_LEARNING_RATE, (H, W, 3))
    # Warp optimizer removed, will use CMA-ES in Phase 1
    spatial_warp = SpatialWarp() # Holds the optimizable warp parameters

    # --- Setup CMA-ES Optimizer ---
    CMAES_POPULATION_SIZE = 50 # Population size for CMA-ES
    cma_optimizer = CMAESOptimizer(
        initial_params=spatial_warp.get_params(), # Start at [0, 0, 0]
        W=W, H=H, img_center=img_center,
        max_rotation=MAX_ROTATION,
        max_shift=MAX_SHIFT,
        population_size=CMAES_POPULATION_SIZE,
        verbose=-9 # Suppress internal CMA-ES prints
    )
    # --- End CMA-ES Setup ---

    # --- Optimization Phase Control ---
    current_phase = 1 # Start with phase 1 (Alignment)
    phase1_step = 0
    final_M_corrective = None # Store the corrective warp from phase 1
    total_steps = 0
    # ---

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    writer = None
    last_corrected_perceived = None # Store the final corrected frame

    # Init loss variables outside loop
    loss, rgb_L, ssim_L = 0.0, 0.0, 0.0
    grad_bmr = np.zeros_like(beamer.get()) # Init grad_bmr

    while True:
        total_steps += 1
        t_start_iter = time.time()

        # --- Phase-Dependent Setup & Forward/Backward Pass ---
        if current_phase == 1:
            # Phase 1: Align geometry using CMA-ES. Use TARGET as fixed beamer output.
            current_beamer_map = target_bgr_f32 # Fixed beamer map for phase 1 evaluations

            # Simulate camera capture ONCE per iteration (fixed beamer map, fixed camera misalignment)
            misaligned_perceived = camera.capture(current_beamer_map) # Fixed for Phase 1 evaluations

            # --- CMA-ES Optimization Step using the optimizer class ---
            if not cma_optimizer.should_stop():
                # Ask for new solutions (scaled)
                solutions_scaled = cma_optimizer.ask()
                # Evaluate them and tell CMA-ES the results
                cma_optimizer.evaluate_and_tell(misaligned_perceived, lossfn)

                # Update spatial_warp with the current *mean* parameters (unscaled) for logging/visualization
                current_warp_params_unscaled = cma_optimizer.get_mean_unscaled_params()
                spatial_warp.update(current_warp_params_unscaled)

                # Get the corrective matrix based on the updated unscaled MEAN parameters
                M_corrective = spatial_warp.get_matrix(img_center)
                # Calculate the corrected image using the MEAN parameters for loss logging and visualization
                corrected_perceived = cv2.warpAffine(misaligned_perceived, M_corrective, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                # Update loss components based on the mean parameters' result for consistent logging
                loss, rgb_L, ssim_L = lossfn.compute_loss_components(corrected_perceived)
                # loss variable here now reflects the loss for the *mean* parameters

            # Phase Transition Check
            phase1_step += 1
            # Stop if max iterations reached OR CMA-ES converges/stalls
            if phase1_step >= PHASE1_ITERATIONS or cma_optimizer.should_stop():
                # Final update using the absolute best parameters found by CMA-ES (unscaled)
                best_params_unscaled_final = cma_optimizer.get_best_unscaled_params()
                spatial_warp.update(best_params_unscaled_final) # Update with unscaled best params
                print(f"\n--- Phase 1 Complete ({phase1_step} iterations) ---")
                stop_reason = cma_optimizer.get_stop_reason() # Get the dictionary of stop criteria
                print(f"CMA-ES Stop Reason: {stop_reason}")
                # Print the unscaled parameters from the spatial_warp object
                print(f"Final Warp (Best): θ={spatial_warp.theta:.2f}, tx={spatial_warp.tx:.1f}, ty={spatial_warp.ty:.1f}")

                final_M_corrective = spatial_warp.get_matrix(img_center) # Store final warp from best unscaled params
                current_phase = 2
                # Reset beamer map to initial value and reset its optimizer for Phase 2
                beamer.reset()
                color_opt.reset()
                print("--- Starting Phase 2: Color Optimization ---\n")
                # Skip rest of loop for this iteration after transition
                continue


        else: # current_phase == 2
            # Phase 2: Correct color. Use optimizable beamer output, fixed warp.
            current_beamer_map = beamer.get() # Optimizable beamer map
            M_corrective = final_M_corrective # Fixed warp matrix from Phase 1

            # --- Forward Pass for Phase 2 ---
            # 1. Simulate camera capture (includes projection and misalignment) - MUST be recalculated with current beamer map
            misaligned_perceived = camera.capture(current_beamer_map)

            # 2. Apply FIXED corrective warp
            corrected_perceived = cv2.warpAffine(misaligned_perceived, M_corrective, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

            # --- Compute Loss ---
            loss, rgb_L, ssim_L = lossfn.compute_loss_components(corrected_perceived)

            # --- Compute Gradients & Update Beamer ---
            grad_corrected_perceived = lossfn.compute_rgb_gradient_wrt_input(corrected_perceived)
            beamer_clip_mask = (current_beamer_map > 0.0) & (current_beamer_map < 1.0)
            grad_bmr = (grad_corrected_perceived * surf * beamer_clip_mask).astype(np.float32) # Calculate beamer gradient
            # Update Beamer parameters
            beamer.update(color_opt.step(grad_bmr, current_beamer_map))

            # Read current warp parameters (fixed) only for logging consistency
            current_warp_params = spatial_warp.get_params()


        # --- Store final corrected frame ---
        last_corrected_perceived = corrected_perceived

        # --- Logging ---
        phase_str = f"P{current_phase}"
        step_in_phase_str = f"{phase1_step}/{PHASE1_ITERATIONS}" if current_phase == 1 else f"{total_steps - PHASE1_ITERATIONS}"
        step_str = f"Step {total_steps} ({step_in_phase_str})"
        loss_str = f"{phase_str} {step_str}: L={loss:.4f}"
        if RGB_LOSS_W > 0: loss_str += f" RGB={rgb_L:.4f}"
        if SSIM_LOSS_W > 0: loss_str += f" SSIM={ssim_L:.4f}"
        # Read warp params directly from spatial_warp object which stores unscaled values
        warp_str = f"Warp: θ={spatial_warp.theta:.2f}, tx={spatial_warp.tx:.1f}, ty={spatial_warp.ty:.1f}"

        # Show relevant info based on phase
        if current_phase == 1:
             # Log CMA-ES standard deviations (these are in the SCALED space)
             std_devs_scaled = cma_optimizer.get_scaled_stds()
             if std_devs_scaled is not None:
                 # Display scaled std deviations for insight into optimizer state
                 info_str = f"Warp σ (scaled)=[{std_devs_scaled[0]:.1e}, {std_devs_scaled[1]:.1e}, {std_devs_scaled[2]:.1e}]"
             else: # Handle initial state or if stds aren't available
                 info_str = f"Warp σ (scaled)=(init)" # Indicate initial state
        else: # Phase 2
             grad_bmr_mean_abs = np.mean(np.abs(grad_bmr))
             info_str = f"|∇Beamer|={grad_bmr_mean_abs:.2e}"

        print(f"{loss_str} | {warp_str} | {info_str}")


        # --- Visualization ---
        vis_surf = (surf * 255).astype(np.uint8)
        # Display the beamer map being USED in the current phase/iteration
        vis_beamer = (current_beamer_map * 255).astype(np.uint8)
        vis_misaligned = (np.clip(misaligned_perceived, 0, 1) * 255).astype(np.uint8)
        vis_corrected = (np.clip(corrected_perceived, 0, 1) * 255).astype(np.uint8)
        vis_target = (target_bgr_f32 * 255).astype(np.uint8)

        # Label beamer differently in phase 1
        beamer_label = "Beamer (Target Fixed)" if current_phase == 1 else "Beamer (Optimizing)"

        vis = composite_frame([
            vis_surf, vis_beamer, vis_misaligned, vis_corrected, vis_target
        ], ["Surface", beamer_label, "Misaligned", "Corrected (P1: Mean / P2: Final)", "Target"]) # Updated label for clarity
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
        vis_last_corrected = (np.clip(last_corrected_perceived, 0, 1) * 255).astype(np.uint8) # This uses the result from the *last* iteration (potentially mean params if P1 ended)
        vis_target = (target_bgr_f32 * 255).astype(np.uint8)

        # Regenerate corrected view using the *final best* warp params for accurate comparison
        if final_M_corrective is not None:
             # Use the misaligned view corresponding to the *final* beamer map if available, else use default
             final_beamer_misaligned = camera.capture(final_beamer_map) # What the camera sees with the final beamer output
             vis_optimized_corrected_final = (np.clip(cv2.warpAffine(final_beamer_misaligned, final_M_corrective, (W,H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), 0, 1)*255).astype(np.uint8)
        else:
             # Fallback if phase 1 didn't finish or something went wrong
             vis_optimized_corrected_final = vis_last_corrected


        comparison_img = composite_frame([
            vis_final_beamer,
            vis_default_misaligned,
            vis_optimized_corrected_final, # Show corrected view using final best warp
            vis_target
        ], [
            "Final Beamer",
            "Default View (Misaligned)", # Show what happens if you just project target with misalignment
            "Optimized View (Corrected)",
            "Target"
        ])

        # Ensure config module is accessible or COMPARISON_FILENAME is defined directly
        try:
            comparison_filename = config.COMPARISON_FILENAME
        except AttributeError:
            comparison_filename = "comparison_result.jpg" # Fallback filename
            print(f"Warning: config.COMPARISON_FILENAME not found, using default: {comparison_filename}")

        cv2.imwrite(comparison_filename, comparison_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"Saved final comparison image to {comparison_filename}")


if __name__ == "__main__":
    # Ensure config is imported if COMPARISON_FILENAME is used from it in the save block
    import config # Keep config import if needed, e.g., for COMPARISON_FILENAME

    #main('../assets/surface.jpg', 0.5)
    main('../assets/surface.jpg', '../assets/target.jpg')
import cv2
import numpy as np
import time
import os

# Constants
TARGET_SIZE = 1024
FPS = 10
DT = 1.0 / FPS
LEARNING_RATE = 0.1
AMBIENT_LIGHT_STRENGTH = 0.0    # Strength of ambient light relative to max beamer brightness
INITIAL_BEAMER_OUT_VALUE = 0.25 # Initial value for beamer out map (0 to 1)
DEFAULT_TARGET_VALUE = 0.5      # Default target brightness if none specified (0 to 1)
GRADIENT_LOSS_WEIGHT = 2000.0      # Weight for the gradient difference loss term
OUTPUT_FILENAME = "projection_evolution.mp4"
COMPARISON_FILENAME = "comparison_output.png"
WINDOW_NAME = "Projection Mapping Simulator"

# --- Visualization Helpers ---
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.2
FONT_THICKNESS = 3
TEXT_COLOR = (105, 195, 105) # BGR Greenish

def composite_frame(images, labels, font=FONT, font_scale=FONT_SCALE, font_thickness=FONT_THICKNESS, text_color=TEXT_COLOR, text_y_offset=50):
    """Stacks images horizontally and adds labels below each image."""
    if not images or not labels or len(images) != len(labels):
        print("Error: Mismatch between images and labels or empty lists.")
        return None # Or raise an error

    # Ensure all images are 3-channel BGR for stacking and text
    processed_images = []
    for img in images:
        if len(img.shape) == 2: # Grayscale
            processed_images.append(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        elif len(img.shape) == 3 and img.shape[2] == 1: # Single channel
             processed_images.append(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        elif len(img.shape) == 3 and img.shape[2] == 3: # BGR
            processed_images.append(img)
        else:
            print(f"Error: Unsupported image shape: {img.shape}")
            return None

    # Assuming all images have the same height and width after processing
    h, w, _ = processed_images[0].shape
    combined_view = np.hstack(processed_images)
    frame_height, total_width, _ = combined_view.shape

    # Add Text Overlays
    text_y = frame_height - text_y_offset # Position from bottom
    shadow_offset = 2 # Offset for the shadow
    shadow_color = (0, 0, 0) # Black color for shadow

    for i, label in enumerate(labels):
        # Calculate text size to center it
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        # Center text horizontally within the column for that image
        text_x = (i * w) + (w - text_width) // 2

        # Draw shadow first
        shadow_x = text_x + shadow_offset
        shadow_y = text_y + shadow_offset
        cv2.putText(combined_view, label, (shadow_x, shadow_y), font, font_scale, shadow_color, font_thickness, cv2.LINE_AA)

        # Draw main text on top
        cv2.putText(combined_view, label, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    return combined_view

# --- Core Classes ---

class InputSurface:
    def __init__(self, image_path):
        self.original = cv2.imread(image_path).astype(np.float32) / 255.0
        self.surface_image = self.resize_max(self.original, TARGET_SIZE)

    @staticmethod
    def resize_max(image, max_size):
        h, w = image.shape[:2]
        scale = max_size / max(h, w)
        return cv2.resize(image, (int(w * scale), int(h * scale)))

    def get_surface(self):
        return self.surface_image

class BeamerOutLayer:
    def __init__(self, shape):
        self.beamer_out_map = np.ones(shape, dtype=np.float32) * INITIAL_BEAMER_OUT_VALUE

    def update(self, delta):
        self.beamer_out_map += delta
        self.beamer_out_map = np.clip(self.beamer_out_map, 0, 1)

    def get_beamer_out_map(self):
        return self.beamer_out_map


class LossFunction:
    def __init__(self, shape, target=None):
        target_dtype = np.float32
        self.target = None
        sh, sw, _ = shape # Get surface height and width
        self.normalization_calculated = False # Flag for dynamic weight calculation
        self.dynamic_gradient_weight = GRADIENT_LOSS_WEIGHT # Initialize with default

        if target is None:
            self.target = np.ones(shape, dtype=target_dtype) * DEFAULT_TARGET_VALUE
            return

        if isinstance(target, (float, int)):
            self.target = np.ones(shape, dtype=target_dtype) * float(target)
            return

        # --- Handle Image Path Target ---
        if isinstance(target, str) and os.path.exists(target):
            target_image = cv2.imread(target).astype(target_dtype) / 255.0
            th, tw, _ = target_image.shape

            # Calculate aspect ratios
            surface_aspect = sw / sh
            target_aspect = tw / th

            # Center crop to match surface aspect ratio
            if target_aspect > surface_aspect: # Target is wider, crop width
                new_tw = int(th * surface_aspect)
                crop_x = (tw - new_tw) // 2
                cropped_target = target_image[:, crop_x:crop_x + new_tw]
            elif target_aspect < surface_aspect: # Target is taller, crop height
                new_th = int(tw / surface_aspect)
                crop_y = (th - new_th) // 2
                cropped_target = target_image[crop_y:crop_y + new_th, :]
            else: # Aspect ratios match, no crop needed
                cropped_target = target_image

            # Resize cropped image to surface dimensions
            self.target = cv2.resize(cropped_target, (sw, sh))
            return
        # --- End Handle Image Path Target ---

        else:
            print(f"Error: Target is not None, a float/int, or a valid image path: {target} ({type(target)}) ")
            # Default to prevent crashes, could also raise an error
            self.target = np.ones(shape, dtype=target_dtype) * DEFAULT_TARGET_VALUE
            pass

    def compute(self, perceived_image, surface_image):
        # Convert images to CIELAB for perceptual loss calculation
        perceived_lab = cv2.cvtColor(perceived_image.astype(np.float32), cv2.COLOR_BGR2LAB)
        target_lab = cv2.cvtColor(self.target.astype(np.float32), cv2.COLOR_BGR2LAB)

        # Calculate squared Euclidean distance in LAB space (proportional to Delta E 76 squared)
        error_lab = target_lab - perceived_lab
        lab_loss = np.mean(np.sum(error_lab**2, axis=-1)) # Sum squares across L, a, b channels, then mean

        # Calculate raw gradient difference loss
        raw_gradient_loss = self._compute_gradient_loss(perceived_image, self.target)

        total_loss = lab_loss + GRADIENT_LOSS_WEIGHT * raw_gradient_loss
        
        # --- Gradient Calculation (Approximation in RGB) ---
        # This gradient is based on the RGB difference, acting as a heuristic direction
        # for minimizing the combined loss.
        # dL/d_beamer_out = dL/d_perceived_rgb * d_perceived_rgb/d_beamer_out
        # We approximate dL/d_perceived_rgb with -2 * (target_rgb - perceived_rgb)
        # From the simulator: perceived = surface * (ambient + beamer_out)
        # d_perceived_rgb/d_beamer_out = surface_image
        error_rgb = self.target - perceived_image # Error in the original BGR space
        # The gradient component related to beamer output change is influenced by surface reflectance
        gradient = -2 * error_rgb * surface_image
        # --- End Gradient Calculation ---
        # Print individual loss components for better insight
        print(f"Total loss: {total_loss:.4f} (LAB: {lab_loss:.4f}, Scaled Grad: {raw_gradient_loss:.4f})")

        return total_loss, gradient # Return total loss and original gradient

    def _compute_gradient_loss(self, perceived_image, target_image):
        """Computes loss based on the difference of image gradients."""
        # Convert to grayscale for simpler gradient calculation
        # Ensure input is float32 [0, 1] before converting
        perceived_gray = cv2.cvtColor(perceived_image.astype(np.float32), cv2.COLOR_BGR2GRAY)
        target_gray = cv2.cvtColor(target_image.astype(np.float32), cv2.COLOR_BGR2GRAY)

        # Calculate gradients using Sobel operator (using 64F for precision)
        grad_p_x = cv2.Sobel(perceived_gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_p_y = cv2.Sobel(perceived_gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_t_x = cv2.Sobel(target_gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_t_y = cv2.Sobel(target_gray, cv2.CV_64F, 0, 1, ksize=3)

        # Calculate loss as the mean squared error of the gradient differences (x and y separately)
        loss_x = np.mean((grad_p_x - grad_t_x)**2)
        loss_y = np.mean((grad_p_y - grad_t_y)**2)

        # Combined gradient loss (sum of MSE for x and y gradients)
        gradient_loss = loss_x + loss_y
        return gradient_loss

class ProjectionSimulator:
    def __init__(self, surface_image, beamer_out_layer, ambient_light_strength):
        self.surface_image = surface_image
        self.beamer_out_layer = beamer_out_layer
        self.ambient_light_strength = ambient_light_strength # Store ambient light strength

    def render(self):
        # Scale beamer out map by max projector brightness # Removed scaling
        # scaled_beamer_out = self.beamer_out_layer.get_beamer_out_map() * self.max_projector_brightness # Removed
        beamer_out_map = self.beamer_out_layer.get_beamer_out_map()
        # Calculate perceived image using the additive model with ambient light
        # perceived = self.surface_image + scaled_beamer_out # Old model
        # New model: Surface reflects ambient light + projector light
        perceived = self.surface_image * (self.ambient_light_strength + beamer_out_map)
        return np.clip(perceived, 0, 1) # Clip to valid range [0, 1]

# --- Main Execution ---

def main(surface_path, target = None):
    input_surface = InputSurface(surface_path)
    surface_image = input_surface.get_surface()
    h, w, _ = surface_image.shape

    beamer_out_layer = BeamerOutLayer(surface_image.shape)
    # Pass ambient light strength instead of max brightness
    simulator = ProjectionSimulator(surface_image, beamer_out_layer, ambient_light_strength=AMBIENT_LIGHT_STRENGTH)
    # Get the target image from the loss function
    loss_fn = LossFunction(shape=surface_image.shape, target=target)
    target_image = loss_fn.target # Access the target image

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # --- Video Saving Setup ---
    # Use width from the first composite frame to ensure correct video dimensions
    # Generate one frame first to get dimensions
    _vis_surface = (surface_image * 255).astype(np.uint8)
    _vis_beamer_out = (beamer_out_layer.get_beamer_out_map() * 255).astype(np.uint8)
    _vis_perceived_init = (simulator.render() * 255).astype(np.uint8) # Initial perceived
    _vis_target = (target_image * 255).astype(np.uint8)
    _initial_frame = composite_frame(
        [_vis_surface, _vis_beamer_out, _vis_perceived_init, _vis_target],
        ["Surface", "Beamer Out", "Perceived", "Target"]
    )
    if _initial_frame is None:
        print("Error creating initial frame for video setup. Exiting.")
        return

    frame_height, frame_width, _ = _initial_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(OUTPUT_FILENAME, fourcc, FPS, (frame_width, frame_height))
    if not video_writer.isOpened():
        print(f"Error: Could not open video writer for {OUTPUT_FILENAME}")
        # Optionally, continue without saving video or exit
        # return

    # Write the first frame we already composed
    if video_writer.isOpened():
        video_writer.write(_initial_frame)

    # --- Optimization Loop ---
    while True:
        start = time.time()

        perceived_image = simulator.render()
        # Pass surface_image to compute for gradient calculation
        loss, grad = loss_fn.compute(perceived_image, surface_image)

        beamer_out_layer.update(-LEARNING_RATE * grad)

        # --- Visualization ---
        vis_surface = (surface_image * 255).astype(np.uint8)
        vis_beamer_out = (beamer_out_layer.get_beamer_out_map() * 255).astype(np.uint8)
        vis_perceived = (perceived_image * 255).astype(np.uint8)
        vis_target = (target_image * 255).astype(np.uint8)

        # Create the frame using the composite function
        combined_view = composite_frame(
            [vis_surface, vis_beamer_out, vis_perceived, vis_target],
            ["Surface", "Beamer Out", "Perceived", "Target"],
            text_y_offset=30 # Adjusted offset for potentially smaller video frames
        )

        if combined_view is not None:
            cv2.imshow(WINDOW_NAME, combined_view)
            if video_writer.isOpened():
                video_writer.write(combined_view)
            else:
                # Only print error once if writing fails
                if 'video_write_error' not in locals():
                    print("Error: Video writer is not open during loop.")
                    video_write_error = True # Flag to prevent repeated messages
        else:
            print("Error creating frame in loop.")
            break # Exit loop if frame creation fails

        key = cv2.waitKey(1)
        # Quit main loop on ESC key or 'q' key:
        if key == 27 or key == ord('q'):
            break

        elapsed = time.time() - start
        if elapsed < DT:
            time.sleep(DT - elapsed)

    if video_writer.isOpened():
        video_writer.release()
        print(f"Video saved to {OUTPUT_FILENAME}")

    # --- Save Comparison Image ---
    # Calculate default perceived image using the new ambient light model
    # Default projection: target image projected onto the surface, plus ambient light
    # default_projected = target_image * simulator.max_projector_brightness # Old
    # default_perceived = np.clip(surface_image + default_projected, 0, 1) # Old

    # New default perceived: ambient contribution + target projected onto surface
    default_perceived = surface_image * (simulator.ambient_light_strength + target_image)
    default_perceived = np.clip(default_perceived, 0, 1)

    # Get the final optimized perceived image from the last iteration
    # 'perceived_image' holds the result from the last simulator.render() call before the loop ended
    final_perceived = perceived_image

    # Convert images to uint8 for saving/display
    vis_default_perceived = (default_perceived * 255).astype(np.uint8)
    vis_final_perceived = (final_perceived * 255).astype(np.uint8)
    vis_target_final = (target_image * 255).astype(np.uint8)

    # Create the comparison image using the composite function
    comparison_image = composite_frame(
        [vis_default_perceived, vis_final_perceived, vis_target_final],
        ["Default Perceived", "Optimized Perceived", "Target"]
    )

    # Save the comparison image
    if comparison_image is not None:
        cv2.imwrite(COMPARISON_FILENAME, comparison_image)
        print(f"Comparison image saved to {COMPARISON_FILENAME}")
    else:
        print("Error creating final comparison image.")
    # --- End Save Comparison Image ---

    cv2.destroyAllWindows()


if __name__ == "__main__":
    surface_path = '../assets/surface.jpg'
    target = '../assets/target.jpg'
    #target = 0.5 # Example: use a uniform target value

    main(surface_path, target=target)

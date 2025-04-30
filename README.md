# Projection Mapping Color Correction Optimizer

This repository contains a Python script (`color_correction/opt_loop_advanced.py`) that optimizes the output of a projector (`beamer`) to make a projected image appear as close as possible to a target image, considering the color and texture of the projection surface and ambient light.

It uses an optimization loop (Adam optimizer) to adjust the projector's RGB output per pixel, minimizing the difference between the perceived image (simulated based on surface, ambient light, and projector output) and the desired target image.

## How it Works

1.  **Simulation:** The script simulates how the projector's light interacts with the projection surface (`surface.jpg`) and ambient light.
2.  **Loss Calculation:** It calculates the difference (currently using Mean Squared Error in RGB space) between the simulated perceived image and the `target.jpg`.
3.  **Optimization:** It uses the Adam optimizer to calculate the gradient of the loss with respect to the projector's output and updates the projector's output image (`beamer`) to minimize this loss.
4.  **Visualization:** It displays the surface, the current beamer output, the simulated perceived image, and the target image side-by-side during the optimization process.
5.  **Output:**
    *   Saves a video (`projection_evolution.mp4`) showing the evolution of the optimization.
    *   Saves a final comparison image (`comparison.jpg`) showing the optimized result alongside the target and a default projection attempt.

## Requirements

*   Python 3
*   OpenCV (`opencv-python`)
*   NumPy

You can install the required libraries using pip:
```bash
pip install opencv-python numpy
```

## Usage

1.  Place your projection surface image (e.g., a photo of the wall or object you're projecting onto) as `assets/surface.jpg`.
2.  Place your desired target image as `assets/target.jpg`.
3.  Navigate to the `color_correction` directory.
4.  Run the script:
    ```bash
    cd color_correction
    python opt_loop_advanced.py
    ```
5.  The script will open a window showing the optimization process. Press `q` or `Esc` to stop.
6.  The output video (`projection_evolution.mp4`) and comparison image (`comparison.jpg`) will be saved in the `color_correction` directory.

## Configuration

You can adjust various hyper-parameters within the `opt_loop_advanced.py` script, such as:

*   `TARGET_SIZE`: Resolution for processing.
*   `FPS`: Output video frame rate.
*   `INITIAL_BEAMER_OUT_VALUE`: Starting brightness for the projector simulation.
*   `AMBIENT_LIGHT_STRENGTH`: Simulated ambient light level.
*   `LEARNING_RATE`, `BETA1`, `BETA2`: Adam optimizer parameters.
*   `RGB_LOSS_W`: Weight for the RGB loss component. 
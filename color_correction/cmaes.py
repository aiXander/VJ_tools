import cma
import numpy as np
import cv2
import random

from components import Loss # Assuming Loss is needed indirectly via lossfn evaluation
from spatial_warp import SpatialWarp # Import from the new file

class MultiStartCMAESOptimizer:
    """Multi-start CMA-ES optimizer with adaptive restarts to escape local optima."""
    
    def __init__(self, initial_params, W, H, img_center,
                 max_rotation, max_shift, population_size, verbose=-9,
                 max_restarts=5, min_improvement_threshold=0.001):
        self.W = W
        self.H = H
        self.img_center = img_center
        self.max_restarts = max_restarts
        self.min_improvement_threshold = min_improvement_threshold
        self.verbose = verbose
        
        # Track restart state
        self.current_restart = 0
        self.best_overall_loss = float('inf')
        self.best_overall_params = None
        self.restart_history = []
        self.stagnation_counter = 0
        self.last_improvement_loss = float('inf')
        
        # Define scaling factors based on typical ranges/limits
        self.scale_factors = np.array([max_rotation, max_shift * W, max_shift * H], dtype=np.float32)
        # Avoid division by zero if a scale factor is zero
        self.scale_factors[self.scale_factors == 0] = 1.0
        
        # Define bounds in the original space first
        self.warp_bounds_orig = [
            [-max_rotation * 1.5, -max_shift * W * 1.1, -max_shift * H * 1.1], # Lower bounds [theta, tx, ty]
            [ max_rotation * 1.5,  max_shift * W * 1.1,  max_shift * H * 1.1]  # Upper bounds [theta, tx, ty]
        ]
        
        # Scale the bounds
        self.warp_bounds_scaled = [
            (np.array(self.warp_bounds_orig[0]) / self.scale_factors).tolist(),
            (np.array(self.warp_bounds_orig[1]) / self.scale_factors).tolist()
        ]
        
        # Base CMA-ES options
        self.base_cma_options = {
            'popsize': population_size,
            'bounds': self.warp_bounds_scaled,
            'verbose': verbose
        }
        
        # Initialize first CMA-ES instance
        self._initialize_cmaes(initial_params)
        
    def _initialize_cmaes(self, initial_params=None):
        """Initialize a new CMA-ES instance with potentially randomized starting point."""
        if initial_params is None or self.current_restart > 0:
            # Generate random starting point for restarts
            initial_params = self._generate_random_start()
        
        # Scaled initial guess
        initial_params_scaled = np.array(initial_params, dtype=np.float32) / self.scale_factors
        
        # Adaptive sigma based on restart number
        initial_sigma0 = 1.0 * (1.2 ** self.current_restart)  # Increase exploration with each restart
        
        self.es = cma.CMAEvolutionStrategy(initial_params_scaled.tolist(), initial_sigma0, self.base_cma_options)
        self.solutions_scaled = None
        
        if self.verbose >= 0:
            print(f"CMA-ES restart {self.current_restart}: initial_params={initial_params}, sigma0={initial_sigma0:.3f}")
    
    def _generate_random_start(self):
        """Generate a random starting point within bounds for restarts."""
        bounds_lower = np.array(self.warp_bounds_orig[0])
        bounds_upper = np.array(self.warp_bounds_orig[1])
        
        # Generate random point within bounds with some bias toward center for first few restarts
        if self.current_restart < 2:
            # Bias toward center for early restarts
            center_bias = 0.3
            random_params = bounds_lower + (bounds_upper - bounds_lower) * (
                center_bias + (1 - 2 * center_bias) * np.random.random(3)
            )
        else:
            # Full random for later restarts
            random_params = bounds_lower + (bounds_upper - bounds_lower) * np.random.random(3)
        
        return random_params
    
    def _should_restart(self):
        """Determine if we should restart CMA-ES."""
        if self.current_restart >= self.max_restarts:
            return False
            
        # Check if CMA-ES has stopped
        if self.es.stop():
            return True
            
        # Check for stagnation (no improvement for several generations)
        if self.stagnation_counter > 10:  # Reduced stagnation threshold for faster convergence
            return True
            
        return False
    
    def ask(self):
        """Get the next population of candidate solutions (scaled)."""
        self.solutions_scaled = self.es.ask()
        return self.solutions_scaled

    def evaluate_and_tell(self, misaligned_perceived, lossfn, camera_simulator=None):
        """
        Evaluates the current population and updates the CMA-ES state.
        Includes restart logic and best solution tracking.
        """
        if self.solutions_scaled is None:
            raise RuntimeError("Must call ask() before evaluate_and_tell()")

        losses = []
        best_loss_this_gen = float('inf')
        best_params_this_gen = None
        
        for i, params_scaled in enumerate(self.solutions_scaled):
            # Unscale parameters before using them
            params_unscaled = np.array(params_scaled) * self.scale_factors
            # Calculate the corrective warp matrix for this candidate
            M_corrective_candidate = SpatialWarp.get_matrix_from_params(
                params_unscaled[0], params_unscaled[1], params_unscaled[2], self.img_center
            )
            # Apply candidate warp to the *fixed* misaligned image captured for this iteration
            corrected_candidate = cv2.warpAffine(
                misaligned_perceived, M_corrective_candidate, (self.W, self.H),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )
            # Compute loss for this candidate (use blur for smoothing in Phase 1)
            loss_candidate, _, _ = lossfn.compute_loss_components(corrected_candidate, use_blur=True)
            losses.append(loss_candidate)
            
            # Track best in this generation
            if loss_candidate < best_loss_this_gen:
                best_loss_this_gen = loss_candidate
                best_params_this_gen = params_unscaled.copy()

        # Update overall best
        if best_loss_this_gen < self.best_overall_loss:
            improvement = self.best_overall_loss - best_loss_this_gen
            self.best_overall_loss = best_loss_this_gen
            self.best_overall_params = best_params_this_gen.copy()
            
            # Check for significant improvement
            if improvement > self.min_improvement_threshold:
                self.stagnation_counter = 0
                self.last_improvement_loss = best_loss_this_gen
            else:
                self.stagnation_counter += 1
        else:
            self.stagnation_counter += 1

        # Update CMA-ES state with SCALED solutions and their losses
        self.es.tell(self.solutions_scaled, losses)
        self.solutions_scaled = None
        
        # Check if we should restart
        if self._should_restart() and self.current_restart < self.max_restarts:
            self.restart_history.append({
                'restart_num': self.current_restart,
                'best_loss': self.best_overall_loss,
                'generations': self.es.countiter if hasattr(self.es, 'countiter') else 0,
                'stop_reason': self.es.stop()
            })
            
            if self.verbose >= 0:
                print(f"Restarting CMA-ES (restart {self.current_restart + 1}/{self.max_restarts})")
                print(f"Best loss so far: {self.best_overall_loss:.6f}")
            
            self.current_restart += 1
            self.stagnation_counter = 0
            self._initialize_cmaes()

    def should_stop(self):
        """Check if optimization should stop (all restarts exhausted or converged)."""
        return self.current_restart >= self.max_restarts and self.es.stop()

    def get_mean_unscaled_params(self):
        """Get the current mean of the distribution (unscaled)."""
        return self.es.mean * self.scale_factors

    def get_best_unscaled_params(self):
        """Get the best solution found across all restarts (unscaled)."""
        if self.best_overall_params is not None:
            return self.best_overall_params
        else:
            return self.es.result.xbest * self.scale_factors

    def get_stop_reason(self):
        """Get comprehensive stop information including restart history."""
        base_stop = self.es.stop()
        return {
            'cmaes_stop': base_stop,
            'restart_count': self.current_restart,
            'best_overall_loss': self.best_overall_loss,
            'restart_history': self.restart_history,
            'stagnation_counter': self.stagnation_counter
        }

    def get_scaled_stds(self):
        """Get the standard deviations in the scaled space."""
        if hasattr(self.es.result, 'stds') and self.es.result.stds is not None:
            return self.es.result.stds
        else:
            return None

class CMAESOptimizer:
    def __init__(self, initial_params, W, H, img_center,
                 max_rotation, max_shift, population_size, verbose=-9):
        self.W = W
        self.H = H
        self.img_center = img_center

        # Define scaling factors based on typical ranges/limits
        self.scale_factors = np.array([max_rotation, max_shift * W, max_shift * H], dtype=np.float32)
        # Avoid division by zero if a scale factor is zero
        self.scale_factors[self.scale_factors == 0] = 1.0

        # Scaled initial guess
        initial_params_scaled = np.array(initial_params, dtype=np.float32) / self.scale_factors

        # Define bounds in the original space first
        warp_bounds_orig = [
            [-max_rotation * 1.5, -max_shift * W * 1.1, -max_shift * H * 1.1], # Lower bounds [theta, tx, ty]
            [ max_rotation * 1.5,  max_shift * W * 1.1,  max_shift * H * 1.1]  # Upper bounds [theta, tx, ty]
        ]

        # Scale the bounds
        warp_bounds_scaled = [
            (np.array(warp_bounds_orig[0]) / self.scale_factors).tolist(),
            (np.array(warp_bounds_orig[1]) / self.scale_factors).tolist()
        ]

        cma_options = {
            'popsize': population_size,
            'bounds': warp_bounds_scaled, # Use scaled bounds
            'verbose': verbose
        }

        # Initialize CMA-ES with scaled parameters and sigma=1.0
        # Use a single sigma0=1.0 in the scaled space as recommended
        initial_sigma0 = 1.0
        self.es = cma.CMAEvolutionStrategy(initial_params_scaled.tolist(), initial_sigma0, cma_options)
        self.solutions_scaled = None # To store current population

    def ask(self):
        """Get the next population of candidate solutions (scaled)."""
        self.solutions_scaled = self.es.ask()
        return self.solutions_scaled

    def evaluate_and_tell(self, misaligned_perceived, lossfn, camera_simulator=None):
        """
        Evaluates the current population and updates the CMA-ES state.
        Requires the pre-captured misaligned_perceived image (using the fixed target).
        """
        if self.solutions_scaled is None:
            raise RuntimeError("Must call ask() before evaluate_and_tell()")

        losses = []
        for params_scaled in self.solutions_scaled:
            # Unscale parameters before using them
            params_unscaled = np.array(params_scaled) * self.scale_factors
            # Calculate the corrective warp matrix for this candidate
            M_corrective_candidate = SpatialWarp.get_matrix_from_params(
                params_unscaled[0], params_unscaled[1], params_unscaled[2], self.img_center
            )
            # Apply candidate warp to the *fixed* misaligned image captured for this iteration
            corrected_candidate = cv2.warpAffine(
                misaligned_perceived, M_corrective_candidate, (self.W, self.H),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )
            # Compute loss for this candidate (use blur for smoothing in Phase 1)
            loss_candidate, _, _ = lossfn.compute_loss_components(corrected_candidate, use_blur=True)
            losses.append(loss_candidate)

        # Update CMA-ES state with SCALED solutions and their losses
        self.es.tell(self.solutions_scaled, losses)
        self.solutions_scaled = None # Reset for next iteration

    def should_stop(self):
        """Check if CMA-ES recommends stopping."""
        return self.es.stop()

    def get_mean_unscaled_params(self):
        """Get the current mean of the distribution (unscaled)."""
        return self.es.mean * self.scale_factors

    def get_best_unscaled_params(self):
        """Get the best solution found so far (unscaled)."""
        return self.es.result.xbest * self.scale_factors

    def get_stop_reason(self):
        """Get the stop criteria dictionary."""
        return self.es.stop() # The stop() method returns the criteria dictionary

    def get_scaled_stds(self):
        """Get the standard deviations in the scaled space."""
        if hasattr(self.es.result, 'stds') and self.es.result.stds is not None:
            return self.es.result.stds
        else:
            return None # Not available yet or not calculated 
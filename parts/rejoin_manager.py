import math
import numpy as np
from scipy.interpolate import CubicHermiteSpline

class RejoinManager:
    def __init__(self, wheelbase_m, steer_max_rad, activation_dist=2.0, deactivation_dist=1.0, lookahead_m=10.0):
        self.wheelbase_m = wheelbase_m
        self.steer_max_rad = steer_max_rad
        
        # Absolute maximum theoretical curvature from bicycle model
        term = max(1e-6, abs(steer_max_rad))
        self.max_curvature = abs(math.tan(term) / wheelbase_m)
        # Apply a safety margin so we don't plan paths right at the physical limit
        self.max_curvature *= 0.8  

        # State fields
        self.in_rejoin_mode = False
        self.activation_dist = activation_dist
        self.deactivation_dist = deactivation_dist
        self.lookahead_m = lookahead_m

        self.active_x = None
        self.active_y = None
        self.active_psi = None
        self.active_kappa = None
        self.active_speeds = None

    @staticmethod
    def calculate_heading(path_xy):
        """Calculates track heading (psi) using central differences for a global track loop."""
        n = len(path_xy)
        psi_rad = np.zeros(n, dtype=float)
        for i in range(n):
            prev_i = (i - 1) % n
            next_i = (i + 1) % n
            dx = float(path_xy[next_i, 0] - path_xy[prev_i, 0])
            dy = float(path_xy[next_i, 1] - path_xy[prev_i, 1])
            if np.hypot(dx, dy) < 1e-6:
                dx = float(path_xy[next_i, 0] - path_xy[i, 0])
                dy = float(path_xy[next_i, 1] - path_xy[i, 1])
            psi_rad[i] = np.arctan2(dy, dx)
        return psi_rad

    def get_target_point(self, start_idx, lookahead_m, global_x, global_y):
        n = len(global_x)
        dist = 0.0
        idx = start_idx
        
        for _ in range(n):
            next_idx = (idx + 1) % n
            dx = global_x[next_idx] - global_x[idx]
            dy = global_y[next_idx] - global_y[idx]
            seg_len = math.hypot(dx, dy)
            
            dist += seg_len
            if dist >= lookahead_m:
                return next_idx
            idx = next_idx
            
        return idx

    def _calculate_curvature(self, x, y):
        """Calculates curvature along a discrete parametric path."""
        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        
        denom = dx**2 + dy**2
        denom[denom == 0] = 1e-6
        curvature = (dx * ddy - dy * ddx) / (denom**1.5)
        return curvature
        
    def _build_path_heading(self, x, y):
        """Calculates instantaneous heading along a discrete parametric path."""
        dx = np.gradient(x)
        dy = np.gradient(y)
        return np.arctan2(dy, dx)

    def generate_rejoin_path(self, current_x, current_y, current_yaw_rad, 
                             global_closest_idx, global_x, global_y, global_psi, global_erpm,
                             base_lookahead_m=10.0, step_m=2.0, max_lookahead_m=50.0):
        """
        Generates a smooth cubic Hermite spline path from the current position back to the global path.
        If the resulting curvature exceeds capabilities, it stretches the target lookahead forward until valid.
        """
        target_lookahead = base_lookahead_m
        
        # P0: Current state
        p0 = np.array([current_x, current_y])
        t0 = np.array([math.cos(current_yaw_rad), math.sin(current_yaw_rad)])
        
        best_spline = None
        min_max_k = float('inf')
        
        while target_lookahead <= max_lookahead_m:
            target_idx = self.get_target_point(global_closest_idx, target_lookahead, global_x, global_y)
            
            # P1: Target state
            p1 = np.array([global_x[target_idx], global_y[target_idx]])
            target_psi = global_psi[target_idx]
            t1 = np.array([math.cos(target_psi), math.sin(target_psi)])
            
            # Distance scales the tangents to provide a natural curve behavior without harsh buckling
            dist = np.linalg.norm(p1 - p0)
            if dist < 1e-3:
                # Fallback if somehow perfectly matched
                dist = 1.0
                
            scale = dist * 1.0  
            
            spline = CubicHermiteSpline([0, 1], np.vstack([p0, p1]), np.vstack([t0*scale, t1*scale]))
            
            # Discretize path roughly every 0.5m
            N = max(10, int(dist * 2)) 
            t_vals = np.linspace(0, 1, N)
            points = spline(t_vals)
            
            spline_x = points[:, 0]
            spline_y = points[:, 1]
            
            kappa = self._calculate_curvature(spline_x, spline_y)
            
            max_k = np.max(np.abs(kappa))
            
            # Track the closest-to-valid path just in case we never find a completely valid one
            if max_k < min_max_k:
                min_max_k = max_k
                best_spline = (spline_x, spline_y, kappa, target_idx, N)
            
            if max_k <= self.max_curvature:
                break
                
            target_lookahead += step_m
            
        # Unpack the best finding (whether it hit physical limits completely or we just had to fall back)
        spline_x, spline_y, kappa, target_idx, N = best_spline
        headings = self._build_path_heading(spline_x, spline_y)
        
        # Give a slight speed penalty to ERPM when rejoining to be safe
        erpm = np.full(N, float(global_erpm[target_idx])) * 0.8
        
        return spline_x, spline_y, headings, kappa, erpm

    def update_state(self, x, y, yaw_rad, global_closest_idx, global_closest_dist_m,
                     global_x, global_y, global_psi, global_speeds, global_kappa=None):
        """
        Evaluates distance thresholds and triggers path generation or restoration.
        Returns a flag indicating if rejoin is active, and the 5 trajectory property arrays to follow.
        """
        if not self.in_rejoin_mode and global_closest_dist_m > self.activation_dist:
            print(f"REJOIN ACTIVATED: Dist = {global_closest_dist_m:.2f}m")
            self.in_rejoin_mode = True
            
            self.active_x, self.active_y, self.active_psi, self.active_kappa, self.active_speeds = self.generate_rejoin_path(
                x, y, yaw_rad, 
                global_closest_idx, global_x, global_y, global_psi, global_speeds,
                base_lookahead_m=self.lookahead_m
            )
            
        elif self.in_rejoin_mode and global_closest_dist_m <= self.deactivation_dist:
            print(f"REJOIN DEACTIVATED: Dist = {global_closest_dist_m:.2f}m")
            self.in_rejoin_mode = False
            self.active_x, self.active_y, self.active_psi, self.active_kappa, self.active_speeds = None, None, None, None, None

        if self.in_rejoin_mode:
            return True, self.active_x, self.active_y, self.active_psi, self.active_kappa, self.active_speeds
        
        return False, global_x, global_y, global_psi, global_kappa, global_speeds

import numpy as np
import time
from donkeycar.la import Line3D, Vec3
from donkeycar.utils import dist
import logging

from parts.predictive_model import TinyPredictiveModel as PredictiveModel


# Helper Classes

def normalize_angle_deg(angle):
    return (angle + 180.0) % 360.0 - 180.0

# donkeycar/parts/transform.py
class PIDController:
    """ Performs a PID computation and returns a control value.
        This is based on the elapsed time (dt) and the current value
        of the process variable
        (i.e. the thing we're measuring and trying to change).
        https://github.com/chrisspen/pid_controller/blob/master/pid_controller/pid.py
    """

    def __init__(self, p=0, i=0, d=0, debug=False):

        # initialize gains
        self.Kp = p
        self.Ki = i
        self.Kd = d

        # The value the controller is trying to get the system to achieve.
        self.target = 0

        # initialize delta t variables
        self.prev_tm = time.time()
        self.prev_feedback = 0
        self.error = None
        self.integral = 0

        # initialize the output
        self.alpha = 0

        # debug flag (set to True for console output)
        self.debug = debug

    def run(self, target_value, feedback):
        curr_tm = time.time()

        self.target = target_value
        error = self.error = self.target - feedback

        # Calculate time differential.
        dt = curr_tm - self.prev_tm

        # Initialize output variable.
        curr_alpha = 0

        # Add proportional component.
        curr_alpha += self.Kp * error

        # Add integral component.
        self.integral += error * dt
        curr_alpha += self.Ki * self.integral

        # Add differential component (avoiding divide-by-zero).
        if dt > 0:
            # D term should be opposing 
            # feed - prev instead of prev - feed
            curr_alpha -= self.Kd * ((feedback - self.prev_feedback) / float(dt))

        # Maintain memory for next loop.
        self.prev_tm = curr_tm
        self.prev_feedback = feedback

        # Update the output
        self.alpha = curr_alpha

        if (self.debug):
            print('PID target value:', round(target_value, 4))
            print('PID feedback value:', round(feedback, 4))
            print('PID output:', round(curr_alpha, 4))

        return curr_alpha

# donkeycar/parts/path.py
class CTE(object):

    def __init__(self, look_ahead=1, look_behind=1, num_pts=None) -> None:
        self.num_pts = num_pts
        self.look_ahead = look_ahead
        self.look_behind = look_behind

    #
    # Find the index of the path element with minimal distance to (x,y).
    # This prefers the first element with the minimum distance if there
    # are more then one.
    #
    def nearest_pt(self, path, x, y, from_pt=0, num_pts=None):
        from_pt = from_pt if from_pt is not None else 0
        num_pts = num_pts if num_pts is not None else len(path)
        num_pts = min(num_pts, len(path))
        if num_pts < 0:
            logging.error("num_pts must not be negative.")
            return None, None, None

        min_pt = None
        min_dist = None
        min_index = None
        for j in range(num_pts):
            i = (j + from_pt) % len(path)
            p = path[i]
            d = dist(p[0], p[1], x, y)
            if min_dist is None or d < min_dist:
                min_pt = p
                min_dist = d
                min_index = i
        return min_pt, min_index, min_dist


    # TODO: update so that we look for nearest two points starting from a given point
    #       and up to a given number of points.  This will speed things up
    #       but more importantly it can be used to handle crossing paths.
    def nearest_two_pts(self, path, x, y):
        if path is None or len(path) < 2:
            logging.error("path is none; cannot calculate nearest points")
            return None, None

        distances = []
        for iP, p in enumerate(path):
            d = dist(p[0], p[1], x, y)
            distances.append((d, iP, p))
        distances.sort(key=lambda elem : elem[0])

        # get the prior point as start of segment
        iA = (distances[0][1] - 1) % len(path)
        a = path[iA]

        # get the next point in the path as the end of the segment
        iB = (iA + 2) % len(path)
        b = path[iB]
        
        return a, b

    def nearest_waypoints(self, path, x, y, look_ahead=1, look_behind=1, from_pt=0, num_pts=None):
        """
        Get the path elements around the closest element to the given (x,y)
        :param path: list of (x,y) points
        :param x: horizontal coordinate of point to check
        :param y: vertical coordinate of point to check
        :param from_pt: index start start search within path
        :param num_pts: maximum number of points to search in path
        :param look_ahead: number waypoints to include ahead of nearest point.
        :param look_behind: number of waypoints to include behind nearest point.
        :return: index of first point, nearest point and last point in nearest path segments
        """
        if path is None or len(path) < 2:
            logging.error("path is none; cannot calculate nearest points")
            return None, None

        if look_ahead < 0:
            logging.error("look_ahead must be a non-negative number")
            return None, None
        if look_behind < 0:
            logging.error("look_behind must be a non-negative number")
            return None, None
        if (look_ahead + look_behind) > len(path):
            logging.error("the path is not long enough to supply the waypoints")
            return None, None

        _pt, i, _distance = self.nearest_pt(path, x, y, from_pt, num_pts)

        # get  start of segment
        a = (i + len(path) - look_behind) % len(path)

        # get the end of the segment
        b = (i + look_ahead) % len(path)

        return a, i, b

    def nearest_track(self, path, x, y, look_ahead=1, look_behind=1, from_pt=0, num_pts=None):
        """
        Get the line segment around the closest point to the given (x,y)
        :param path: list of (x,y) points
        :param x: horizontal coordinate of point to check
        :param y: vertical coordinate of point to check
        :param from_pt: index start start search within path
        :param num_pts: maximum number of points to search in path
        :param look_ahead: number waypoints to include ahead of nearest point.
        :param look_behind: number of waypoints to include behind nearest point.
        :return: start and end points of the nearest track and index of nearest point
        """

        a, i, b = self.nearest_waypoints(path, x, y, look_ahead, look_behind, from_pt, num_pts)

        return (path[a], path[b], i) if a is not None and b is not None else (None, None, None)

    def run(self, path, x, y, look_ahead, look_behind, from_pt=None):
        """
        Run cross track error algorithm
        :return: cross-track-error, index of nearest point, and segment endpoints
        """
        cte = 0.
        i = from_pt

        a, b, i = self.nearest_track(path, x, y, 
                                     look_ahead=look_ahead, look_behind=look_behind, 
                                     from_pt=from_pt, num_pts=None)
        
        print(f"[CTE]a:{a},b:{b}")
        if type(a) == np.ndarray and type(b) == np.ndarray:
            logging.info(f"nearest: ({a[0]}, {a[1]}) to ({x}, {y})")
            a_v = Vec3(a[0], 0., a[1])
            b_v = Vec3(b[0], 0., b[1])
            p_v = Vec3(x, 0., y)
            line = Line3D(a_v, b_v)
            err = line.vector_to(p_v)
            sign = 1.0
            cp = line.dir.cross(err.normalized())
            if cp.y > 0.0 :
                sign = -1.0
            cte = err.mag() * sign            
        else:
            logging.info(f"no nearest point to ({x},{y}))")
        return cte, i, a, b

class CTEController:
    def __init__(
        self,
        path_csv,
        throttle=1000,
        kp=0.5,
        ki=0.0,
        kd=0.0,
    ):  
        self.lookahead, self.lookbehind = 3, 1
        self.cte = CTE(look_ahead=self.lookahead, look_behind=self.lookbehind)
        self.pid = PIDController(p=kp, i=ki, d=kd, debug=False)
        a = np.genfromtxt(path_csv, delimiter=',', dtype=float, encoding='utf-8', skip_header=0)
        if a.ndim == 1:
            a = np.reshape(a, (1, -1))
        if a.shape[1] >= 3 and np.isfinite(a[0, 0]) and np.isfinite(a[0, 1]):
            self.path_xy = a[:, :2]
            self.pwm_table = np.asarray(a[:, -1], dtype=int)
        else:
            a = np.genfromtxt(path_csv, delimiter=',', dtype=float, encoding='utf-8', skip_header=1)
            self.path_xy = a[:, :2]
            self.pwm_table = None
        self.throttle = throttle
        self.prev_cte = None
        self.pred_model = PredictiveModel()

        self.K_STEER = -0.18
        self.K_BIAS = 0 # 0.018
        self.ff_lookahead_m = 1.0

        self.path_s = self.compute_path_s(self.path_xy)
        self.path_curvature = self.compute_path_curvature(self.path_xy)

    @staticmethod
    def compute_path_s(path_xy):
        n = len(path_xy)
        s = np.zeros(n, dtype=float)
        for i in range(1, n):
            s[i] = s[i - 1] + np.linalg.norm(path_xy[i] - path_xy[i - 1])
        return s

    @staticmethod
    def compute_path_curvature(path_xy, stride=2):
        n = len(path_xy)
        kappa = np.zeros(n, dtype=float)

        for i in range(n):
            p0 = path_xy[(i - stride) % n]
            p1 = path_xy[i]
            p2 = path_xy[(i + stride) % n]

            a = np.linalg.norm(p1 - p0)
            b = np.linalg.norm(p2 - p1)
            c = np.linalg.norm(p2 - p0)
            if a < 1e-6 or b < 1e-6 or c < 1e-6:
                continue

            cross = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
            kappa[i] = 2.0 * cross / (a * b * c)

        window = 5
        pad = window // 2
        padded = np.r_[kappa[-pad:], kappa, kappa[:pad]]
        kernel = np.ones(window) / window
        return np.convolve(padded, kernel, mode="valid")

    def index_ahead(self, idx, meters):
        n = len(self.path_xy)
        dist = 0.0
        i = int(idx) % n

        while dist < meters:
            j = (i + 1) % n
            dist += np.linalg.norm(self.path_xy[j] - self.path_xy[i])
            i = j

        return i

    def run(self, x, y, yaw, gps_speed, gps_heading):
        if self.prev_cte is not None and abs(self.prev_cte) > 1.5:
            self.lookahead = 4
        else:
            self.lookahead = 3

        # One/N step pseudo MPC - predict X/Y N steps into the future
        N=0
        x_future, y_future = x, y
        x_future, y_future = self.pred_model.run(
            x_future, y_future, gps_speed, gps_heading, dt=N*0.02
        )

        cte, idx, a, b = self.cte.run(self.path_xy, x_future, y_future, look_ahead=self.lookahead, look_behind=self.lookbehind)
        self.prev_cte = cte
        cte_steer = self.pid.run(0.0, cte) # we desire 0 cte
        
        cte_steer *= -1


        # feedforward from precomputed path curvature ahead of current index
        if idx is None:
            curvature = 0.0
        else:
            curv_idx = self.index_ahead(idx, self.ff_lookahead_m)
            curvature = float(self.path_curvature[curv_idx])
        steer_ff = (curvature - self.K_BIAS) / self.K_STEER
        steer_ff = float(np.clip(steer_ff, -0.4, 0.4))

        steer =  -1 * steer_ff +  cte_steer
        steer = np.clip(steer,-1,1)
        # if abs(steer) < 0.04:
        #     steer = 0
        # print(f"[CTEController] Reversing steer")

        if self.pwm_table is not None:
            i = 0 if idx is None else int(idx) % len(self.pwm_table)
            throttle = int(self.pwm_table[i])
        else:
            STEER_THRESHOLD = 0.2
            HIGH = 1000 # 2600
            LOW = 1000 # 1400
            if np.abs(steer) < STEER_THRESHOLD:
                throttle = int(HIGH - ((HIGH - LOW) / STEER_THRESHOLD * np.abs(steer)))
            else:
                throttle = LOW
            throttle = -1

        if self.pid.debug:
            print('CTE:', round(cte, 4))
        
        return throttle, steer

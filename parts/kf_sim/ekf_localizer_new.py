import time
import math
import numpy as np

DEFAULT_HZ = 50
NUM_STATES = 5
# TODO: find these offsets
# NOTE: these are the position of each GPS in the KART BODY FRAME
# (x forward along the kart heading, y to the left), relative to the
# point the state vector tracks. If the state tracks the midpoint of the
# two GPSs and the baseline runs along the kart's forward axis, these are
# just (-dist_between_gps/2, 0) and (+dist_between_gps/2, 0).
ZED_GPS_OFFSET_M = (0.1, 0.25) # offset of the ZED gps from where position is measured from
GEM_GPS_OFFSET_M = (0.1, -0.25) # offset of the GEM gps from where position is measured from

def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

# Rotates a 2D vector counter-clockwise by a given angle in radians
def rotate_2d_vector(vector, angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array(((c, -s), (s, c)))
    return R @ vector

class EKF_Localizer:
    """
    state vector: 
        position_x (eastward meters relative to first waypoint)
        position_y (northward meters relative to first waypoint)
        heading (radians according to unit circle, with east being 0rad)
        forward_vel (meters/sec velocity in the direction of the heading of the kart)
        gyro_bias (rad/sec representing the estimated bias of the gyro)

    position_x and position_y represent position of center of 2 gps's
    angular velocity of kart is not in state vector, it is just stored separately as instance data

    the 2 GPSs on the kart
    ZED-F9P
    GEM1305
    """
    def __init__(self, start_x, start_y, start_heading, dist_between_gps):
        self.dist_between_gps = dist_between_gps
        self.x = np.array([start_x, start_y, start_heading, 0, 0], dtype=float)

        # H matricies to convert state vector x into sensor measurement
        # gps angle measurement
        self.H_ang = np.array([
            [0, 0, 1, 0, 0]
        ])
        # drive encoder measurement
        self.H_enc = np.array([
            [0, 0, 0, 1, 0]
        ])
        # H matrix for gps position is calculated dynamically b/c 
        # h(x) for gps is nonlinear
        # and its linearization H_gps changes based on heading

        # R matricies representing covariance of each sensor measurement
        # CEP of ZED_F9P = 0.01 m, standard dev of ZED_F9P = 0.00849 m
        self.R_gps_zed = np.array([
            [0.00849 ** 2, 0],
            [0, 0.00849 ** 2]
        ])
        # CEP of GEM1305 = 0.02 m, standard dev of GEM1305 = 0.01698
        self.R_gps_gem = np.array([
            [0.01698 ** 2, 0],
            [0, 0.01698 ** 2]
        ])

        # 1x1 covariance matrix for heading from GPS
        self.R_heading = self.make_R_heading(self.R_gps_zed, self.R_gps_gem, start_heading, dist_between_gps)

        
        # TODO: need to find drive encoder velocity covariance with data
        self.R_enc = np.zeros((1, 1))
        self.R_enc[0,0] = 0.01  # this is just placeholder value so it doesn't throw error

        # TODO: find Q matrix
        # Q matrix representing covariance of model
        #self.Q = np.zeros((NUM_STATES, NUM_STATES))
        self.Q = np.diag([0.01, 0.01, 0.01, 0.01, 0.01])

        # matrix reprsenting covariance of error between current estimated state and true state
        #self.est_error_cov = np.zeros((NUM_STATES, NUM_STATES))
        self.est_error_cov = np.diag([0.1**2, 0.1**2, math.radians(5)**2, 0.05, math.radians(0.5)**2])

        self.prev_update_time = None
        self.last_gyro_heading_vel = 0.0

    # the covariance of the gps angle measurement can be directly derived from
    # the coviarance of the gps position
    # R_gps1 and R_gps2 should both have shape (2,2)
    # returns a 1x1 matrix
    def make_R_heading(self, R_gps1, R_gps2, expected_heading: float, dist_between_gps:float):
        assert R_gps1.shape == R_gps2.shape == (2,2)
        variance_x = R_gps1[0,0] + R_gps2[0,0]
        variance_y = R_gps1[1,1] + R_gps2[1,1]
        expected_delta_x = math.cos(expected_heading) * dist_between_gps
        expected_delta_y = math.sin(expected_heading) * dist_between_gps

        a = -expected_delta_y / (dist_between_gps ** 2)
        b = expected_delta_x / (dist_between_gps ** 2)

        return np.array([[(a ** 2) * variance_x + (b ** 2) * variance_y]])
    
    # h(x) = [pos_x, pos_y] + R(heading) @ offset_body
    # this returns the predicted gps position and the H matrix b/c h(x) is a nonlinear function
    # so it is better to directly evaluate predicted_gps_position than to use H to find linearized version
    # kart_to_gps_offset should be a tuple with 2 elements
    def make_h_and_H_gps(self, kart_to_gps_offset: tuple[float, float]):
        heading = self.x[2]
        c, s = math.cos(heading), math.sin(heading)
        lx, ly = kart_to_gps_offset

        predicted_gps_pos = self.x[0:2] + rotate_2d_vector(np.array(kart_to_gps_offset, dtype=float), heading)

        H = np.zeros((2, NUM_STATES))
        H[0,0] = 1.0
        H[1,1] = 1.0
        H[0,2] = -lx * s - ly * c
        H[1,2] =  lx * c - ly * s
        return predicted_gps_pos, H

    # predicts the next state with euler integration
    def predict_next_state(self, x, dt: float, accel_fwd: float, ang_vel_gyro: float):
        assert x.shape == (NUM_STATES,)
        """
        Euler integration:
            position += velocity * dt
            velocity += accel * dt
        """
        pos_x, pos_y, heading, speed_fwd, bias_gyro = x

        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        vx = speed_fwd * cos_h
        vy = speed_fwd * sin_h

        pos_x += vx * dt
        pos_y += vy * dt
        speed_fwd += accel_fwd * dt
        heading += (ang_vel_gyro - bias_gyro) * dt
        heading = wrap(heading)
        return np.array([pos_x, pos_y, heading, speed_fwd, bias_gyro])


    # F is the jacobian of function f(x, u) with respect to x taken at x
    # f(x, u) is the function specifying the euler integration used in predict_next_state
    # this needs to be recalculated every loop because F is a linearization of partial derivative of f(x, u)
    # because f(x, u) is a nonlinear function (as seen below with the cos and sin)
    def build_F(self, heading: float, velocity: float, dt: float):
        F = np.eye(NUM_STATES)
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        F[0,2] = -velocity * sin_h * dt
        F[0,3] = cos_h * dt
        F[1,2] = velocity * cos_h * dt
        F[1,3] = sin_h * dt
        F[2,4] = -dt
        return F

    # uses F and Q to update the error covariance
    # (the covariance would normally grow)
    # this is a standard formula
    def propogate_error_through_model(self, x, dt: float):
        assert x.shape == (NUM_STATES,)
        pos_x, pos_y, heading, velocity, bias_gyro = x
        F = self.build_F(heading, velocity, dt)
        return F @ self.est_error_cov @ F.T + self.Q

    # calculates the kalman gain matrix of a specific measurement given:
    # the estimated state error covariance matrix (nxn matrix)
    # H (matrix to convert x into measurement dimensions)
    # R (covariance of measurement, this is property of sensor)
    # returns an nxm matrix where n is NUM_STATES and m is num sensor measurements
    def calculate_kalman_gain(self, est_error_cov, H, R):
        assert est_error_cov.shape == (NUM_STATES,NUM_STATES)
        assert H.shape[0] == R.shape[0] == R.shape[1]
        assert H.shape[1] == NUM_STATES
        #          nxn         nxm                mxn       nxn        nxm  mxm
        return est_error_cov @ H.T @ np.linalg.inv(H @ est_error_cov @ H.T + R)

    # updates the state when a measurement comes in according to the kalman_gain
    # added predicted_measurement so a nonlinear h(x) can be
    # passed in; when it is None this behaves exactly as before)
    # kalman_gain: nxm
    # measurement: m
    # H: mxn
    # returns new state vector (vector w n entries)
    def update_state_with_measurement(self, x, kalman_gain, measurement, H, angular:bool=False, predicted_measurement=None):
        assert x.shape == (NUM_STATES,)
        num_measurements = measurement.shape[0]
        assert kalman_gain.shape == (NUM_STATES,num_measurements)
        assert H.shape == (num_measurements,NUM_STATES)
        
        if predicted_measurement is None:
            predicted_measurement = H @ x
        innovation = measurement - predicted_measurement
        if angular:
            innovation[0] = wrap(innovation[0])
        x_new = x + kalman_gain @ innovation
        x_new[2] = wrap(x_new[2])
        return x_new

    # updates the covariance of the estimated state after recieving a measurement
    # (the covariance would normally shrink)
    # kalman gain: nxm
    # H: mxn
    # prev_error: nxn
    # returns new error, so nxn
    def update_error_after_measurement(self, kalman_gain, H, prev_error):
        assert kalman_gain.shape == (NUM_STATES,H.shape[0])
        assert H.shape[1] == NUM_STATES
        assert prev_error.shape == (NUM_STATES,NUM_STATES)
        new_error = (np.eye(NUM_STATES) - kalman_gain @ H) @ prev_error
        return 0.5 * (new_error + new_error.T)

    # imu_accel is [ax, ay], (control input to the prediction)
    # gyro_heading_vel is scalar (control input to the prediction)
    # gps_position is [eastward meters, northward meters] 
    # gps_heading is scalar (measurement)
    # drive_encoder_vel is scalar (measurement)
    def update(self, imu_accel, gyro_heading_vel, gps_zed_position, gps_gem_position, gps_heading, drive_encoder_vel, cur_time=None):
        cur_time = cur_time if cur_time is not None else time.monotonic()
        dt = cur_time - self.prev_update_time if self.prev_update_time is not None else 1. / DEFAULT_HZ

        self.R_heading = self.make_R_heading(self.R_gps_gem, self.R_gps_zed, self.get_heading_rad(), self.dist_between_gps)

        # update state with model and update error covariance when either accel or gyro is not null
        if imu_accel is not None or gyro_heading_vel is not None:
            self.prev_update_time = cur_time
            # if imu accel data is None, assume 0 accel (not sure if there is a better way to do this)
            accel_x, accel_y = imu_accel if imu_accel is not None else (0, 0)
            # if gyro data is None, assume previous gyro data (not sure if there is a better way to do this)
            self.last_gyro_heading_vel = gyro_heading_vel if gyro_heading_vel is not None else self.last_gyro_heading_vel
            self.est_error_cov = self.propogate_error_through_model(self.x, dt)
            self.x = self.predict_next_state(self.x, dt, accel_x, self.last_gyro_heading_vel)

        # update state with measurements and update error covariance
        # read in gps position measurement(s)
        # CLAUDE CHANGED THIS
        # the raw sensor R is used now (R_gps_zed / R_gps_gem), not an inflated
        # "kart position" R. the heading-induced uncertainty is no longer our job
        # to precompute - it falls out of H @ P @ H.T. each GPS also updates
        # independently on its own arrival, so the two never need to be synced.
        gps_info = [] # list of tuples (position, R, offset from kart)
        if gps_zed_position is not None:
            gps_info.append((gps_zed_position, self.R_gps_zed, ZED_GPS_OFFSET_M))
        if gps_gem_position is not None:
            gps_info.append((gps_gem_position, self.R_gps_gem, GEM_GPS_OFFSET_M))
        for gps_pos, gps_R, gps_offset in gps_info:
            # CLAUDE CHANGED THIS
            # build h(x) and H for THIS antenna, then fuse the raw GPS reading.
            # no pre-rotation of the measurement, so the measurement is no longer
            # a function of the state estimate.
            predicted_gps_pos, H_gps = self.make_h_and_H_gps(gps_offset)
            kalman_gain = self.calculate_kalman_gain(self.est_error_cov, H_gps, gps_R)
            self.x = self.update_state_with_measurement(
                self.x,
                kalman_gain,
                np.asarray(gps_pos, dtype=float),
                H_gps,
                predicted_measurement=predicted_gps_pos,
            )
            self.est_error_cov = self.update_error_after_measurement(kalman_gain, H_gps, self.est_error_cov)

        # update state with gps heading measurement
        # NOTE: with nonzero lever arms the GPS position updates above already
        # carry some heading information, and gps_heading is derived from those
        # same two receivers - so this channel partially double counts them.
        # left in for now; see if it fights the gyro once the offsets are measured.
        if gps_heading is not None:
            kalman_gain = self.calculate_kalman_gain(self.est_error_cov, self.H_ang, self.R_heading)
            self.x = self.update_state_with_measurement(self.x, kalman_gain, np.array([gps_heading]), self.H_ang, angular=True)
            self.est_error_cov = self.update_error_after_measurement(kalman_gain, self.H_ang, self.est_error_cov)

        # update state with drive encoder measurement
        if drive_encoder_vel is not None:
            kalman_gain = self.calculate_kalman_gain(self.est_error_cov, self.H_enc, self.R_enc)
            self.x = self.update_state_with_measurement(self.x, kalman_gain, np.array([drive_encoder_vel]), self.H_enc)
            self.est_error_cov = self.update_error_after_measurement(kalman_gain, self.H_enc, self.est_error_cov)

    # get x, y of the kart in meters relative to first waypoint
    def get_position(self):
        return self.x[0], self.x[1]

    # get heading in radians of the kart according to unit circle convention
    # 0 rad is east, increasing counter clockwise
    def get_heading_rad(self):
        return self.x[2]

    # get heading in degrees of kart according to compass direction
    # 0 deg is north, increasing clockwise
    def get_compass_heading_deg(self):
        rad_from_north = wrap(self.get_heading_rad() - 0.5 * math.pi) # [-pi, pi]
        deg_from_north = math.degrees(rad_from_north) # [-180, 180]
        deg_from_north_clockwise = -deg_from_north # [-180, 180]
        return deg_from_north_clockwise

   # get the velocity in meters/sec of kart that is in direction of kart heading
    def get_tangential_vel(self):
        return self.x[3]

    # get the angular velocity in rad/sec of kart with positive being counter clockwise rotation
    def get_angular_vel_rad(self):
        return self.last_gyro_heading_vel - self.x[4]

    # get the angular velocity in deg/sec of kart with positive being clockwise
    def get_angular_vel_deg(self):
        rad = self.get_angular_vel_rad()
        return -math.degrees(rad)
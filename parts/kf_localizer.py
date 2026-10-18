import time
import math
import numpy as np

R_EARTH = 6378137.0
DEFAULT_HZ = 50
NUM_STATES = 5
def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class EK_Localizer:
    """
    state vector: 
        position_x (eastward meters relative to first waypoint)
        position_y (northward meters relative to first waypoint)
        heading (radians according to unit circle, with east being 0rad)
        forward_accel (meters/sec^2 acceleration in the direction of the heading of the kart)
        gyro_bias (rad/sec representing the estimated bias of the gyro)

    angular velocity of kart is not in state vector, it is just stored separately as instance data

    the 2 GPSs on the kart
    ZED-F9P
    GEM1305
    """
    def __init__(self, start_x, start_y, start_heading, dist_between_gps):
        self.dist_between_gps = dist_between_gps
        self.x = np.array([start_x, start_y, start_heading, 0, 0])

        # H matricies to convert state vector x into same dim as sensor measurement
        # gps position measurement
        self.H_gps = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0]
        ])
        # gps angle measurement
        self.H_ang = np.array([
            [0, 0, 1, 0, 0]
        ])
        # drive encoder measurement
        self.H_enc = np.array([
            [0, 0, 0, 1, 0]
        ])

        # R matricies representing covariance of each sensor measurement
        # TODO: find the gps covariances online
        self.R_gps = np.zeros((2, 2))
        self.R_ang = self.make_R_ang(self.R_gps, start_heading, dist_between_gps)
        # TODO: need to find drive encoder velocity covariance
        self.R_enc = np.zeros((1, 1))

        # TODO: find imu accel and gyro covariances online
        # Q matrix representing covariance of model
        self.Q = np.zeros((NUM_STATES, NUM_STATES))

        # matrix reprsenting covariance of error between current estimated state and true state
        self.est_error_cov = np.zeros((NUM_STATES, NUM_STATES))

        self.prev_update_time = None
        self.last_gyro_heading_vel = None

    # the covariance of the gps angle measurement can be directly derived from
    # the coviarance of the gps position
    # so this does not need to be measured, it can be calculated
    def make_R_ang(self, R_gps, expected_heading, dist_between_gps):
        variance_x = R_gps[0,0]
        variance_y = R_gps[1,1]
        expected_delta_x = math.cos(expected_heading) * dist_between_gps
        expected_delta_y = math.sin(expected_heading) * dist_between_gps

        a = -expected_delta_y / (dist_between_gps ** 2)
        b = expected_delta_x / (dist_between_gps ** 2)

        return np.array([[(a ** 2) * variance_x + (b ** 2) * variance_y]])



    # predicts the next state with euler integration
    def predict_next_state(self, x, dt, accel_fwd, ang_vel_gyro):
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
    def build_F(self, heading, velocity, dt):
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
    def propogate_error_through_model(self, x, dt):
        pos_x, pos_y, heading, velocity, bias_gyro = x
        F = self.build_F(heading, velocity, dt)
        return F @ self.est_error_cov @ F.T + self.Q

    # calculates the kalman gain matrix of a specific measurement given:
    # the estimated state error covariance matrix
    # H (matrix to convert x into measurement dimensions)
    # R (covariance of measurement, this is property of sensor)
    # this is a standard formula
    def calculate_kalman_gain(self, est_error_cov, H, R):
        return est_error_cov @ H.T @ np.linalg.inv(H @ est_error_cov @ H.T + R)

    # updates the state when a measurement comes in according to the kalman_gain
    # this is a standard formula
    def update_state_with_measurement(self, x, kalman_gain, measurement, H, angular=False):
        innovation = measurement - (H @ x)
        if angular:
            innovation[4] = wrap(innovation[4])
        x_new = x + kalman_gain @ innovation
        x_new[2] = wrap(x_new[2])
        return x_new

    # updates the covariance of the estimated state after recieving a measurement
    # (the covariance would normally shrink)
    def update_error_after_measurement(self, kalman_gain, H, prev_error):
        return (np.eye(NUM_STATES) - kalman_gain @ H) @ prev_error

    # imu_accel is [ax, ay], (control input to the prediction)
    # gyro_heading_vel is scalar (control input to the prediction)
    # gps_position is [latitude, longitude] (measurement, need to convert to xy relative to first waypoint on track)
    # gps_heading is scalar (measurement)
    # drive_encoder_vel is scalar (measurement)
    def update(self, imu_accel, gyro_heading_vel, gps_position, gps_heading, drive_encoder_vel):
        new_time = time.monotonic()
        dt = new_time - self.prev_update_time if self.prev_update_time is not None else 1. / DEFAULT_HZ

        # update state with model and update error covariance when either accel or gyro is not null
        if imu_accel is not None or gyro_heading_vel is not None:
            self.prev_update_time = new_time
            # if imu accel data is None, assume 0 accel (not sure if there is a better way to do this)
            accel_x, accel_y = imu_accel if imu_accel is not None else (0, 0)
            # if gyro data is None, assume previous gyro data (not sure if there is a better way to do this)
            self.last_gyro_heading_vel = gyro_heading_vel if gyro_heading_vel is not None else self.last_gyro_heading_vel
            self.est_error_cov = self.propogate_error_through_model(self.x, dt)
            self.x = self.predict_next_state(self.x, dt, accel_x, self.last_gyro_heading_vel)

        # update state with measurements and updat error covariance
        if gps_position is not None:
            kalman_gain = self.calculate_kalman_gain(self.est_error_cov, self.H_gps, self.R_gps)
            self.x = self.update_state_with_measurement(self.x, kalman_gain, gps_position, self.H_gps)
            self.est_error_cov = self.update_error_after_measurement(kalman_gain, self.H_gps, self.est_error_cov)

        if gps_heading is not None:
            kalman_gain = self.calculate_kalman_gain(self.est_error_cov, self.H_ang, self.R_ang)
            self.x = self.update_state_with_measurement(self.x, kalman_gain, gps_heading, self.H_ang, angular=True)
            self.est_error_cov = self.update_error_after_measurement(kalman_gain, self.H_ang, self.est_error_cov)

        if drive_encoder_vel is not None:
            kalman_gain = self.calculate_kalman_gain(self.est_error_cov, self.H_enc, self.R_enc)
            self.x = self.update_state_with_measurement(self.x, kalman_gain, drive_encoder_vel, self.H_enc)
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









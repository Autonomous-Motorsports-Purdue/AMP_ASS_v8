import donkeycar as dk
import pickle
from parts.zed_frame_publisher import Zed_Frame_Publisher
from parts.zed_viewer import Zed_Viewer
from parts.camera import Camera

V = dk.Vehicle()
V.add(Camera(), outputs=['left'])
V.add(Zed_Frame_Publisher(), outputs=['left', 'right', 'depth'])

with open("zed_calibration_params.bin", "rb") as f:
    b = pickle.load(f)
V.mem['zed_calibration_params'] = b

V.add(Zed_Viewer(), inputs=['left', 'zed_calibration_params'])

V.start(rate_hz=20)

import static_donkeycar.donkeycar.donkeycar as dk

from parts.image_publisher import Image_Publisher
from parts.image_cv import Object_Detection
from parts.log import Logger

if __name__ == "__main__":
    V = dk.vehicle.Vehicle()
    print("starting")

    V.add(Image_Publisher(), inputs=[], outputs=['image'])
    V.add(Object_Detection(), inputs=['image'], outputs=['image_cv', 'object_x', 'object_y', 'contour_area'])
    V.add(Logger(), inputs=['object_x', 'object_y', 'contour_area'], outputs=[])

    V.start(rate_hz=30)
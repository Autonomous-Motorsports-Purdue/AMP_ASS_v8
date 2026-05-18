import datetime
import cv2
import os
import numpy as np
import csv
import json

from parts.pure_pursuit_controller import PP_DEBUG_FIELDS


class Logger_GPS():
    def __init__(self):
        start_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        # self.image_directory = "data/images/" + start_time
        # self.segmented_directory = "data/segmented_images/" +start_time
        self.information_directory= "data/information/"
        # # self.depth_directory = "data/depth/"  + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + "/"
        # if not os.path.exists(self.image_directory):
        #     os.makedirs(self.image_directory)
        # if not os.path.exists(self.segmented_directory):
        #     os.makedirs(self.segmented_directory)
        if not os.path.exists(self.information_directory):
            os.makedirs(self.information_directory)
        self.info_csv = "data/information/" + start_time + ".csv"
        # Writing to csv file
        self.csvfile = open(self.info_csv, 'w', newline='')

        self.base_fields = [
            'timestamp',
            'latitude',
            'longitude',
            'steering',
            'throttle',
            'fix',
            'gps_heading',
            'gps_speed',
            'imu_heading',
            'imu_accuracy_deg',
            'fused_x',
            'fused_y',
            'fused_yaw',
            'pp_debug_json',
        ]
        self.pp_debug_fields = [f'pp_{field}' for field in PP_DEBUG_FIELDS]
        self.fields = self.base_fields + self.pp_debug_fields
        self.csvwriter = csv.DictWriter(self.csvfile, fieldnames=self.fields)

        # Writing the fields
        self.csvwriter.writeheader()
        
    def run(self, lat, lon, steering, throttle, fix, gps_heading, gps_speed, imu_heading, imu_accuracy_deg, fused_x, fused_y, fused_yaw, pp_debug):
        """
        Logs the current image, segmented Image, centroid, steering, and throttle values.
        Saves the images in their respective directory and logs the image paths and other data into a CSV.
        """
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S.%f')
        row = {
            'timestamp': timestamp,
            'latitude': lat,
            'longitude': lon,
            'steering': steering,
            'throttle': throttle,
            'fix': fix,
            'gps_heading': gps_heading,
            'gps_speed': gps_speed,
            'imu_heading': imu_heading,
            'imu_accuracy_deg': imu_accuracy_deg,
            'fused_x': fused_x,
            'fused_y': fused_y,
            'fused_yaw': fused_yaw,
            'pp_debug_json': json.dumps(pp_debug, sort_keys=True) if isinstance(pp_debug, dict) else '',
        }

        if isinstance(pp_debug, dict):
            for field in PP_DEBUG_FIELDS:
                row[f'pp_{field}'] = pp_debug.get(field, '')
        else:
            for field in PP_DEBUG_FIELDS:
                row[f'pp_{field}'] = ''

        self.csvwriter.writerow(row)
            
            
        


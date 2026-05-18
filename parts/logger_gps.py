import datetime
import cv2
import os
import numpy as np
import csv
class Logger_GPS():
    def __init__(self):
        start_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
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
        self.csvfile = open(self.info_csv, 'w') 
        # Creating a csv writer object
        self.csvwriter = csv.writer(self.csvfile)

        # fields = ['timestamp', 'image', 'segmented', 'centroid', 'steering', 'throttle']
        fields = ['timestamp', 'latitude', 'longitude', 'steering', 'throttle', 'fix', 'gps_heading', 'gps_speed', 'imu_heading', 'imu_accuracy_deg', 'fused_x', 'fused_y', 'fused_yaw']
        # Writing the fields
        self.csvwriter.writerow(fields)
        
    def run(self, lat, lon, steering, throttle, fix, gps_heading, gps_speed, imu_heading, imu_accuracy_deg, fused_x, fused_y, fused_yaw):
        """
        Logs the current image, segmented Image, centroid, steering, and throttle values.
        Saves the images in their respective directory and logs the image paths and other data into a CSV.
        """
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S.%f')
        rows = [timestamp, lat, lon, steering, throttle, fix, gps_heading, gps_speed, imu_heading, imu_accuracy_deg, fused_x, fused_y, fused_yaw]
        self.csvwriter.writerow(rows)
        self.csvfile.flush()

    def shutdown(self):
        if getattr(self, "csvfile", None) is not None and not self.csvfile.closed:
            self.csvfile.flush()
            self.csvfile.close()
            
            
        


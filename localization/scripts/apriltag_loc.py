#!/usr/bin/env python3

import rospy
from cv_bridge import CvBridge
from dtc_msgs.msg import CasualtyFix, CasualtyFixArray
from std_msgs.msg import Header, Time
from sensor_msgs.msg import NavSatFix, Imu, MagneticField,  CompressedImage, Image
from std_msgs.msg import Float64, Float32MultiArray
import cv2
import os
from threading import Thread, Lock
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from pyproj import Proj, Transformer
from tf.transformations import (
    euler_from_quaternion, quaternion_from_euler, quaternion_matrix
)
from shapely.geometry import Point, Polygon

from detector import PersonDetector
from utils.converter import LLtoUTM, UTMtoLL

import numpy as np

# TODO: Modify this paths when running on drone
ckpt = 'default'
run_mode = 'default'
# ckpt = '/ws/src/11x_ft.pt'

# ckpt = '/ws/src/epoch20.pt'
camera_cfg = '/ws/src/localization/scripts/camchain.yaml'
geofence_txt = '/ws/src/geofence.txt'

SAVE_RESULT = False # Save result for debugging
save_path = '/home/rex/data/catkin_ws/src/my_bag_tools/scripts/result.txt'
save_crops = '/home/rex/data/catkin_ws/src/my_bag_tools/scripts/crops.npz'
save_path2 = '/home/rex/data/catkin_ws/src/my_bag_tools/scripts/result_cluster.txt'


class _Adapter:
    def __init__(self, func):
        self._f = func
    def transform(self, *args, **kwargs):
        return self._f(*args, **kwargs)
    __call__ = transform

def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    # Read image (supports both grayscale and 16-bit thermal images)
    
    if img is None:
        raise FileNotFoundError(f"Image not found")

    # Normalize 16-bit thermal image to 8-bit if needed
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype(np.uint8)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    img_clahe = clahe.apply(img)
    return img_clahe

class Localization:
    def __init__(self):
        rospy.init_node('robot_localization')
        
        # self.result = LocalizationResult()
        rospy.loginfo(f"Initializing PersonDetector with {ckpt}")
        rospy.loginfo(f"Run mode set to {run_mode}")
        self.detector = PersonDetector(
            model_path=ckpt,
            camera_config = camera_cfg,
            confidence_threshold=0.65,
            run_mode=run_mode,
            distance_threshold=2.5,
            clustering_method='centroid',
            device='cuda',
            min_points_threshold=10
        )
        geofence = np.loadtxt(geofence_txt)
        geofence = geofence.reshape(-1,2)
        vertices = [(geo[0], geo[1]) for geo in geofence]
        self.geofence = Polygon(vertices)

        self.transformer = Transformer.from_crs("epsg:4326", "epsg:32610", always_xy=True)
        self.inv_transformer = Transformer.from_crs("epsg:32610", "epsg:4326", always_xy=True)
        # self.init_geo_tools()

        self.pub_locations = rospy.Publisher('/casualty_info', CasualtyFixArray, queue_size=10)

        rospy.Subscriber('/mavros/global_position/raw/fix', NavSatFix, self.gps_callback)
        rospy.Subscriber('/mavros/global_position/rel_alt', Float64, self.alt_callback)
        rospy.Subscriber('/mavros/global_position/global', NavSatFix, self.abs_alt_callback)
        rospy.Subscriber('/imu/data', Imu, self.imu_callback)
        rospy.Subscriber('/camera/image_color/compressed', CompressedImage, self.frame_callback, queue_size=1)
        self.init_param()
        self.bridge = CvBridge()

        #TODO: Load calibration from file
        T_ci =  np.array([
            [0.011106298412152327,  0.9999324199187616,   0.0034359468839849595, -0.036802732375442404],
            [-0.999832733821092,    0.01115499451474039,  -0.014493808237225339, -0.008332238900780303],
            [-0.014531156713131006, -0.0032743996068676073, 0.9998890557415823,  -0.08775357009176091],
            [0.0,                   0.0,                   0.0,                   1.0]
        ])
        if run_mode == 'night':
            T_ci =  np.array([
                [0.011106298412152327,  0.9999324199187616,   0.0034359468839849595, -0.11002732375442404],
                [-0.999832733821092,    0.01115499451474039,  -0.014493808237225339, -0.008332238900780303],
                [-0.014531156713131006, -0.0032743996068676073, 0.9998890557415823,  -0.08775357009176091],
                [0.0,                   0.0,                   0.0,                   1.0]
            ])
        self.T_ic = np.linalg.inv(T_ci)

    def init_geo_tools(self):
        self._geo_zone = None
        ellipsoid = 23
        
        def _ll2utm(long, lat):
            zone, easting, northing = LLtoUTM(ellipsoid, lat, long)
            self._geo_zone = zone
            return easting, northing
        def _utm2ll(easting, northing):
            if not self._geo_zone:
                raise ValueError("UTM zone unknown. Call forward transform once (or set self._geo_state['zone']).")
            lat, long = UTMtoLL(ellipsoid, northing, easting, self._geo_zone)
            return lat, long
        self.transformer = _Adapter(_ll2utm)
        self.inv_transformer = _Adapter(_utm2ll)

    def save_result(self):
        self.result.save_objects()

    def pose_callback(self,msg):
        self.imu_time = msg.header.stamp.to_sec()
        q = msg.pose.orientation
        # Quaternion to Euler (roll, pitch, yaw)
        self.rtm = quaternion_matrix([q.x, q.y, q.z, q.w])[:3,:3]


    def init_param(self):
        self.x = None
        self.y = None
        self.z = None
        self.mx = None
        self.my = None
        self.mz = None
        self.rel_alt = None
        self.abs_alt = None
        self.row = None
        self.pitch = None
        self.yaw = None
        self.rtm = None
        self.imu_time = None
        self.gps_time = None
        self.frame_time = None
        self.image = None
        self.processing = False

    def frame_callback(self, msg):
        
        if self.processing:
            return  # Skip frame if still processing previous
        
        
        
        if self.x is None or self.y is None or self.z is None or self.rtm is None :
            return
        self.processing = True
        self.frame_time = msg.header.stamp.to_sec()
        img_np = np.frombuffer(msg.data, np.uint8)
        
        if run_mode == 'night':
            filter_min = 28000
            filter_max = 38000
            self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono16')
            self.image = np.clip(self.image, filter_min, filter_max)
            self.image = apply_clahe(self.image)
            self.image = self.image[...,None]

            self.image = np.repeat(self.image, 3, axis=-1)  
        else:
            self.image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        detect_input = {
            'image':self.image,
            'translation':np.array([self.x, self.y, self.z]),
            'rotation':self.rtm
        }

        # Run detection in a separate thread
        Thread(target=self.run_detection, args=(detect_input,)).start()


    def run_detection(self, detect_input):
        translation = detect_input['translation']
        image = detect_input['image']
        rotation = detect_input['rotation']

        # Process frame using PersonDetector
        result = self.detector.process_frame(
            frame=image,
            translation=translation,
            rotation=rotation,
            enable_localization=True,
            draw_annotations=False,
            return_full_frame=False,
            toGPS=False
        )
        print(result.localizations)
        # Publish results if detections found
        if len(result.localizations) > 0:
            self.publish_detections(result)
            rospy.loginfo(f"Published {len(result.localizations)} detections")


        self.processing = False


    def gps_callback(self, msg):
        self.gps_time = msg.header.stamp.to_sec()
        x_gps, y_gps = self.transformer.transform(msg.longitude, msg.latitude)
        z_gps = self.rel_alt  # EKF still uses rel_alt for Z

        self.x = y_gps # - T_world[0]
        self.y = x_gps #- T_world[1]
        if z_gps is not None:
            self.z = -z_gps #- T_world[2]

    def alt_callback(self, msg):
        self.rel_alt = msg.data

    def abs_alt_callback(self, msg):
        self.abs_alt = msg.altitude
    
    def mag_callback(self, msg):
        self.mx = msg.magnetic_field.x
        self.my = msg.magnetic_field.y
        self.mz = msg.magnetic_field.z
        # self.mx = np.arctan2(-my, mx)

    def imu_callback(self, msg):
        self.imu_time = msg.header.stamp.to_sec()
        # Original IMU orientation
        q_orig = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]

        self.roll, self.pitch, self.yaw = euler_from_quaternion(q_orig)
        self.rtm = quaternion_matrix(q_orig)[:3,:3]



    def publish_detections(self, result):
        """Publish detection results: all casualties (time, GPS, optional image) in one CasualtyFixArray."""
        if not result.localizations:
            return

        # cluster_info = self.detector.get_cluster_info()
        cluster_info = self.detector.tracker.clusterer.get_valid_cluster_info()


        # One common timestamp so entries in the array are coherent
        now = rospy.Time.now()

        wrapper = CasualtyFixArray()
        wrapper.header = Header(stamp=now, frame_id="world")

        casualties = []

        for cluster_id, info in cluster_info.items():
            cf = CasualtyFix()
            cf.header = Header(stamp=now, frame_id="world")
            cf.casualty_id = int(cluster_id)

            # --- NED -> WGS84 ---
            # Assume centroid = [N, E, ...]; swap if yours is [E, N].
            N = float(info['centroid'][0])
            E = float(info['centroid'][1])
            lon, lat = self.inv_transformer.transform(E, N)   # returns (lon, lat)


            point = Point(lat, lon)
            if not self.geofence.contains(point):
                print('Object',cluster_id,' Not in geofence')
            cf.location = NavSatFix()
            cf.location.latitude  = lat
            cf.location.longitude = lon
            cf.location.altitude  = 0.0  # set actual alt if you have it


            ##############################################
            # cf.view_point = result.view_points[cluster_id]
            ##############################################


            cf.time_ago = Time()
            cf.time_ago.data = now

            # --- Optional crop image (leave empty if not available) ---
            # Prefer an explicit mapping if your detector provides one
            cv_img = info['crop']
            img_msg = self.bridge.cv2_to_imgmsg(cv_img, encoding="bgr8")
            img_msg.header = Header(stamp=now, frame_id=f"casualty_{cf.casualty_id}")
            cf.image = img_msg
            # else: leave cf.image at default (empty)

            casualties.append(cf)

        wrapper.casualties = casualties
        self.pub_locations.publish(wrapper)



def extract_lat_lon_from_file(file_path):
    lat, lon = None, None
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip().startswith('latitude:'):
                lat = float(line.strip().split(':')[1])
            elif line.strip().startswith('longitude:'):
                lon = float(line.strip().split(':')[1])
    return [lat, lon] if lat is not None and lon is not None else None

def read_all_casualty_coords(directory):
    coords = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            lat_lon = extract_lat_lon_from_file(file_path)
            if lat_lon:
                coords.append(lat_lon)
    return coords

def save_results_on_exit():
    global detection_raw
    if len(detection_raw)>=1:
        np.savetxt(save_path, np.array(detection_raw))


if __name__ == '__main__':
    try:
        detect = Localization()
        run_mode = rospy.get_param("/mode", 'day')  # "~" means private parameter
        if run_mode == 'day':
            ckpt = '/ws/src/11x_ft.pt'
        elif run_mode == 'night':
            ckpt = '/ws/src/epoch20.pt'
        else:
            print(f"Run mode not implemented: {run_mode}")
            raise NameError
            
        print(f"Run mode set to: {run_mode}")
        if SAVE_RESULT:
            rospy.on_shutdown(save_results_on_exit)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass



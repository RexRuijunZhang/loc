#!/usr/bin/env python3

import rospy
from cv_bridge import CvBridge
from target_localization.msg.dtc_msgs import CasualtyFix, CasualtyFixArray
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

from camera import Camera

import numpy as np

# TODO: Modify this paths when running on drone
ckpt = '/home/rex/data/catkin_ws/src/my_bag_tools/scripts/11x_ft.pt'
camera_cfg = '/home/rex/Downloads/data_07_02_2025/camchain.yaml'

SAVE_RESULT = True # Save result for debugging
save_path = '/home/rex/data/catkin_ws/src/my_bag_tools/scripts/result.txt'
save_crops = '/home/rex/data/catkin_ws/src/my_bag_tools/scripts/crops.npz'
save_path2 = '/home/rex/data/catkin_ws/src/my_bag_tools/scripts/result_cluster.txt'

model = YOLO(ckpt)
model.to('cuda')

last_result = []
last_crops = []
detection_raw = []
result_lock = Lock()

def box_center(
        boxes: Boxes, 
        center: bool = True
) -> np.ndarray:
    """Get the center of the bounding boxes.
    
    Args:
        boxes (Boxes): Bounding boxes from the YOLO model.
        center (bool): If True, the origin is the image center, else top-left corner.
    Returns:
        np.ndarray: Array of shape (N, 2) containing the centers of the boxes.
    """
    box_centers = boxes.xywh[:, :2]
    
    if not center:
        box_centers = box_centers.cpu().numpy()
    else:
        H, W = boxes.orig_shape[:2]
        box_centers = box_centers.cpu().numpy() - np.array([W / 2, H / 2])

    return box_centers.astype(int)

class LocalizationResult:
    def __init__(self):
        self.objects = []
        self.id = 0
        self.threshold = 3 # reject radius 2.5 m
        self.min_poins = 10

    def get_id(self):
        id = self.id
        self.id += 1
        return id
    
    def create_objects(self, center, crop):
        obj = {
            'id': self.get_id(),
            'center': center,
            'image': crop,
            'time': rospy.Time.now()
        }
        self.objects.append(obj)

    def get_object(self, id):
        return self.objects[id]

    def filter_detections(self, detections):
        detections = np.array(detections)
        obj_center = np.array([obj['center'] for obj in self.objects]).reshape(1,-1,2)
        distance = np.linalg.norm(detections.reshape(-1,1,2)-obj_center, axis=-1)
        mask = np.all(distance > self.threshold,axis=-1)
        detections = detections[mask]
        return detections.tolist(), mask
    
    def cluster_detections_once(self, unhandled_detections, crops):
        unhandled_detections = np.array(unhandled_detections)
        distance_mat = np.linalg.norm(unhandled_detections.reshape(-1,1,2) - unhandled_detections.reshape(1,-1, 2), axis=-1)
        thres = distance_mat < self.threshold
        flag = np.count_nonzero(thres, axis=-1) > self.min_poins
        last_true_idx = np.where(flag)[0][-1] if np.any(flag) else -1
        if last_true_idx != -1:
            crop = crops[last_true_idx]
            valuable_detects = unhandled_detections[distance_mat[last_true_idx]< self.threshold]
            self.create_objects(np.mean(valuable_detects,axis=0), crop)
            # self.create_objects(unhandled_detections[last_true_idx], crop)

    def save_objects(self):
        obj_center = np.array([obj['center'] for obj in self.objects])
        image_crops = np.array([obj['image'] for obj in self.objects])
        np.savetxt(save_path2, obj_center)
        np.savez(save_crops, image_crops)

class Localization:
    def __init__(self):
        rospy.init_node('robot_localization')
        self.proj_utm = Proj(proj='utm', zone=33, ellps='WGS84')
        self.proj_wgs84 = Proj(proj='latlong', datum='WGS84')
        self.proj_ll = Transformer.from_proj(self.proj_utm, self.proj_wgs84)
        self.result = LocalizationResult()

        self.pub = rospy.Publisher('/casualty_info', CasualtyFixArray, queue_size=10)

        # self.loc_pub = rospy.Publisher('/loc/locations', Float32MultiArray, queue_size=10)
        self.crop_pub = rospy.Publisher('/loc/crops', Image, queue_size=10)

        rospy.Subscriber('/mavros/global_position/raw/fix', NavSatFix, self.gps_callback)
        rospy.Subscriber('/mavros/global_position/rel_alt', Float64, self.alt_callback)
        rospy.Subscriber('/mavros/global_position/global', NavSatFix, self.abs_alt_callback)
        rospy.Subscriber('/imu/imu', Imu, self.imu_callback)
        rospy.Subscriber('/imu/magnetic_field', MagneticField, self.mag_callback)
        rospy.Subscriber('/camera/image_color/compressed', CompressedImage, self.frame_callback, queue_size=1)
        self.init_param()
        self.bridge = CvBridge()

        #TODO: Load calibration from file
        T_ci =  np.array([
            [0.011106298412152327,  0.9999324199187616,   0.0034359468839849595,  0.036802732375442404],
            [-0.999832733821092,    0.01115499451474039,  -0.014493808237225339, -0.008332238900780303],
            [-0.014531156713131006, -0.0032743996068676073, 0.9998890557415823,  -0.08775357009176091],
            [0.0,                   0.0,                   0.0,                   1.0]
        ])
        self.T_ic = np.linalg.inv(T_ci)
        self.cam = Camera.load_config(camera_cfg)

    def save_result(self):
        self.result.save_objects()

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
        self.image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        # print(self.gps_time, self.imu_time, self.frame_time)
        print('Max time diff:',max(abs(self.frame_time - self.imu_time), abs(self.frame_time - self.gps_time)))
        detect_input = {
            'image':self.image,
            'translation':np.array([self.x, self.y, self.z]),
            'rotation':self.rtm,
            'T_ic': self.T_ic,
            'cam': self.cam
        }

        # Run detection in a separate thread
        Thread(target=self.run_detection, args=(detect_input,)).start()

    def run_detection(self, detect_input):
        # heavy computation here
        global last_result, detection_raw, last_crops
        result, crops = YOLO_detection(detect_input)
        with result_lock:
            for res in result:
                last_result.append(res)
                detection_raw.append(res)
            for cp in crops:
                last_crops.append(cp)
            print(last_result)
            last_result, mask = self.result.filter_detections(last_result)
            last_crops = [d for d, m in zip(last_crops, mask) if m]
            if len(last_result) > 5:
                self.result.cluster_detections_once(last_result, last_crops)
                if SAVE_RESULT:
                    self.result.save_objects()
                self.publish_pose()
        self.processing = False

    def gps_callback(self, msg):
        self.gps_time = msg.header.stamp.to_sec()
        x_gps, y_gps = self.proj_utm(msg.longitude, msg.latitude)
        z_gps = self.rel_alt  # EKF still uses rel_alt for Z

        self.x = x_gps # - T_world[0]
        self.y = -y_gps #- T_world[1]
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

        # TODO: Add global rotation calculation after fixing imu
        # if self.mx is not None:

            # Compensate for roll and pitch on magnetometer
            # mag_x_comp = self.mx * cos_p + self.mz * sin_p
            # mag_y_comp = self.mx * sin_r * sin_p + self.my * cos_r - self.mz * sin_r * cos_p

            # B_world = rtm @ np.array([self.mx, self.my, self.mz])
    def publish_pose(self):
        if len(self.result.objects)<1:
            return
        stitched_image = [obj['image'] for obj in self.result.objects]
        stitched_image = np.concatenate(stitched_image, axis=1)
        # print(stitched_image.shape)
        img_msg = self.bridge.cv2_to_imgmsg(stitched_image, encoding='bgr8')
        self.crop_pub.publish(img_msg)

        # points = np.array([obj['center'] for obj in self.result.objects])

        # point_msg = Float32MultiArray(data = points.flatten().tolist())
        # self.loc_pub.publish(point_msg)

        wrapper_msg = CasualtyFixArray()
        wrapper_msg.header = Header()
        wrapper_msg.header.stamp = rospy.Time.now()
        wrapper_msg.header.frame_id = "world"
        casualties = []
        for obj in self.result.objects:
            msg = CasualtyFix()
            msg.header = Header()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "world"

            msg.casualty_id = obj['id']

            msg.location = NavSatFix()
            lon, lat = self.proj_ll(obj['center'][0], -obj['center'][1])
            msg.location.latitude = lat
            msg.location.longitude = lon
            msg.location.altitude = 0

            msg.time_ago = Time()
            msg.time_ago.data = obj['time']
            casualties.append(msg)
        self.pub.publish(wrapper_msg)


        

def YOLO_detection(input):
    translation = input['translation']
    frame = input['image']
    T_ic = input['T_ic']
    R_wi = input['rotation']
    cam = input['cam']
    R_ic = T_ic[:3,:3]
    t_ic = T_ic[:3,3]

    # TODO: Delete hard code
    translation[-1] = translation[-1]-7
    theta = np.radians(-46)
    Rz = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])
    R_wi = Rz @ R_wi

    result = model.predict(frame, verbose=False, conf=0.65)[0]
    coords = []
    crops = []

    for box in result.boxes:
        img_coord = box_center(box, center=False)
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        
        crop = frame[y1:y2, x1:x2].copy()

        crop = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_NEAREST)
        crops.append(crop)

        img_coord = img_coord.reshape(-1, 2)
        ray_cam = (cam.reproject(img_coord)).reshape(-1, 1)
        ray_cam_homo = np.ones((4,1))
        ray_cam_homo[:3] = ray_cam
        ray_world = R_wi @ R_ic @ (ray_cam)
        
        ray_world = ray_world.flatten()[:3]
        translation = translation + R_wi @ t_ic
        s = -(translation[2] / ray_world[2])
        
        world_coord = translation + s * ray_world 

        print(f"World Coord: {[world_coord[0], world_coord[1], world_coord[2]]}")
        coords.append(world_coord[:2].tolist())

    return coords, crops


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
        if SAVE_RESULT:

            rospy.on_shutdown(save_results_on_exit)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass



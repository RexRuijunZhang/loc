#!/usr/bin/env python3
"""
Visualize published detections from /loc/locations topic
"""

import rospy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dtc_msgs.msg import CasualtyFixArray
import numpy as np
from collections import defaultdict
import time
import cv2
from cv_bridge import CvBridge
from PIL import Image
import io
import os
from sensor_msgs.msg import NavSatFix

from loguru import logger

GT = False

# import sys
# sys.path.insert(0, "/home/darpa/ros_ws2/src/localization/scripts")

def extract_lat_lon_from_file(file_path):
    """Extract latitude and longitude from GT files."""
    lat, lon = None, None
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip().startswith('latitude:'):
                lat = float(line.strip().split(':')[1])
            elif line.strip().startswith('longitude:'):
                lon = float(line.strip().split(':')[1])
    return [lat, lon] if lat is not None and lon is not None else None


def read_all_casualty_coords(directory):
    """Read all casualty coordinates from GT directory."""
    coords = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            lat_lon = extract_lat_lon_from_file(file_path)
            if lat_lon:
                coords.append(lat_lon)
    return coords

class DetectionVisualizer:
    def __init__(self):
      rospy.init_node('detection_visualizer')

      self.bridge = CvBridge()
      self.detections = defaultdict(list)  # cluster_id -> list of (lat, lon, timestamp)
      self.images = {}  # cluster_id -> latest image
      self.drone_trajectory = []  # (lat, lon, timestamp) for drone path
      self.colors = plt.cm.tab10(np.linspace(0, 1, 10))  # Colors for different clusters

      # Set up the plot
      self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 7))

      # Map subplot
      self.ax1.set_title('Detections & Drone Trajectory (GPS)', fontsize=14)
      self.ax1.set_xlabel('Longitude')
      self.ax1.set_ylabel('Latitude')
      self.ax1.grid(True, alpha=0.3)
      

      # Image subplot
      self.ax2.set_title('Latest Detection Images', fontsize=14)
      self.ax2.axis('off')

      # Subscribe to topics
      rospy.Subscriber('/dione/casualty_info', CasualtyFixArray, self.detection_callback)
      rospy.Subscriber('/dione/mavros/global_position/raw/fix', NavSatFix, self.drone_gps_callback)

      # Animation
      self.ani = animation.FuncAnimation(self.fig, self.update_plot, interval=1000, blit=False)

      logger.info("DetectionVisualizer started. Listening to /loc/locations and drone GPS...")
      logger.info("Close the plot window to stop.")
      
      if GT:
        directory_path = '/home/rex/Downloads/gt'
        self.casualty_gps = np.array(read_all_casualty_coords(directory_path))
        self.ax1.scatter(self.casualty_gps[:,1], self.casualty_gps[:,0], color = 'r')

      

    def drone_gps_callback(self, msg):
      """Handle incoming drone GPS messages"""

      if msg.status.status >= 0:  # Valid GPS fix
          timestamp = time.time()
          lat = msg.latitude
          lon = msg.longitude

          self.drone_trajectory.append((lat, lon, timestamp))

          # Keep only last 1000 points to avoid memory issues
        #   if len(self.drone_trajectory) > 1000:
        #       self.drone_trajectory.pop(0)

          rospy.logdebug(f"Drone GPS: ({lat:.6f}, {lon:.6f})")

          
    def detection_callback(self, msg):
        """Handle incoming detection messages"""
        timestamp = time.time()
        
        for casualty in msg.casualties:
            cluster_id = casualty.casualty_id
            lat = casualty.location.latitude
            lon = casualty.location.longitude
            
            # Store detection
            self.detections[cluster_id].append((lat, lon, timestamp))
            
            # Store image if available - FIX THIS PART
            if casualty.image.data and len(casualty.image.data) > 0:
                try:
                    # Convert ROS Image message to OpenCV image
                    cv_image = self.bridge.imgmsg_to_cv2(casualty.image, desired_encoding='bgr8')
                    
                    # Validate image
                    if cv_image is not None and cv_image.size > 0:
                        self.images[cluster_id] = cv_image
                        # Add debugging
                        print(f"Cluster {cluster_id} image: shape={cv_image.shape}, dtype={cv_image.dtype}")
                        if cv_image.shape[0] < 10 or cv_image.shape[1] < 10:
                            print(f"Warning: Very small image for cluster {cluster_id}")
                    else:
                        print(f"Invalid image data for cluster {cluster_id}")
                        
                except Exception as e:
                    rospy.logwarn(f"Failed to convert image for cluster {cluster_id}: {e}")
        
        rospy.loginfo(f"Received {len(msg.casualties)} casualties. Total clusters: {len(self.detections)}")
    
    
    def update_plot(self, frame):
      """Update the visualization"""
      if not self.detections and not self.drone_trajectory:
          return

      # Clear previous plots
      self.ax1.clear()
      self.ax2.clear()
      

      # Plot GPS locations
      self.ax1.set_title('Detections & Drone Trajectory (GPS)', fontsize=14)
      self.ax1.set_xlabel('Longitude')
      self.ax1.set_ylabel('Latitude')
      self.ax1.grid(True, alpha=0.3)
      if GT:
        all_lats = [item[0] for item in self.casualty_gps]
        all_lons = [item[1] for item in self.casualty_gps]
      else:
        all_lats = []
        all_lons = []
      # Plot drone trajectory
      if self.drone_trajectory:
          drone_lats = [d[0] for d in self.drone_trajectory]
          drone_lons = [d[1] for d in self.drone_trajectory]
          drone_times = [d[2] for d in self.drone_trajectory]

          all_lats.extend(drone_lats)
          all_lons.extend(drone_lons)

          # Plot trajectory as a line with gradient (older = lighter)
          if len(drone_lats) > 1:
              # Create segments for gradient effect
              for i in range(len(drone_lats) - 1):
                  alpha = (i + 1) / len(drone_lats) * 0.7 + 0.3  # Alpha from 0.3 to 1.0
                  self.ax1.plot([drone_lons[i], drone_lons[i+1]],
                               [drone_lats[i], drone_lats[i+1]],
                               'b-', alpha=alpha, linewidth=1)

          # Mark current drone position
          if drone_lats:
              current_lat, current_lon = drone_lats[-1], drone_lons[-1]
              self.ax1.scatter([current_lon], [current_lat], c='blue', s=100,
                             marker='^', edgecolors='darkblue', linewidth=2,
                             label='Drone', zorder=5)

          # Mark start position
          if len(drone_lats) > 5:  # Only show start if we have some trajectory
              start_lat, start_lon = drone_lats[0], drone_lons[0]
              self.ax1.scatter([start_lon], [start_lat], c='green', s=80,
                             marker='s', edgecolors='darkgreen', linewidth=2,
                             label='Start', zorder=5)

      # Plot detections (existing code)
      for cluster_id, detections in self.detections.items():
          if not detections:
              continue

          lats = [d[0] for d in detections]
          lons = [d[1] for d in detections]
          timestamps = [d[2] for d in detections]

          all_lats.extend(lats)
          all_lons.extend(lons)

          color = self.colors[cluster_id % len(self.colors)]

          # Plot all detections for this cluster
          self.ax1.scatter(lons, lats, c=[color], alpha=0.8, s=60,
                          label=f'Cluster {cluster_id}', zorder=4)

          # Highlight the latest detection
          if detections:
              latest_lat, latest_lon, _ = detections[-1]
              self.ax1.scatter([latest_lon], [latest_lat], c=[color], s=120, marker='*',
                             edgecolors='black', linewidth=2, zorder=6)

          # Add cluster ID text
          if detections:
              centroid_lat = np.mean(lats)
              centroid_lon = np.mean(lons)
              self.ax1.annotate(f'C{cluster_id}', (centroid_lon, centroid_lat),
                              xytext=(5, 5), textcoords='offset points', fontsize=10,
                              bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8),
                              zorder=7)

      # Set axis limits with some padding
      if all_lats and all_lons:
          lat_range = max(all_lats) - min(all_lats)
          lon_range = max(all_lons) - min(all_lons)
          padding = max(lat_range, lon_range) * 0.1 or 0.001

          self.ax1.set_xlim(min(all_lons) - padding, max(all_lons) + padding)
          self.ax1.set_ylim(min(all_lats) - padding, max(all_lats) + padding)
      if GT:
         self.ax1.scatter(self.casualty_gps[:,1], self.casualty_gps[:,0], color = 'r')
      # Add legend
      self.ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

      # Display images
      self.ax2.set_title('Latest Detection Images', fontsize=14)
      self.ax2.axis('off')

      if self.images:
          # Arrange images in a grid
          n_images = len(self.images)
          if n_images == 0:
              return
          elif n_images == 1:
              rows, cols = 1, 1
          elif n_images <= 4:
              rows, cols = 2, 2
          else:
              rows = int(np.ceil(np.sqrt(n_images)))
              cols = int(np.ceil(n_images / rows))
          
          # Create a combined image with proper size
          img_size = 150  # Increase size for better visibility
          combined_img = np.ones((rows * img_size, cols * img_size, 3), dtype=np.uint8) * 255
          
          for i, (cluster_id, img) in enumerate(self.images.items()):
              if i >= rows * cols:
                  break
                  
              row = cluster_id // cols
              col = cluster_id % cols
              
              # Fix image processing
              try:
                  # Ensure img is valid
                  if img is None or img.size == 0:
                      continue
                  
                  # Handle different image formats
                  if len(img.shape) == 3 and img.shape[2] == 3:
                      # Already RGB/BGR, good
                      img_to_resize = img.copy()
                  elif len(img.shape) == 3 and img.shape[2] == 4:
                      # RGBA, convert to RGB
                      img_to_resize = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                  elif len(img.shape) == 2:
                      # Grayscale, convert to RGB
                      img_to_resize = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                  else:
                      continue
                  
                  # Resize image with proper interpolation
                  img_resized = cv2.resize(img_to_resize, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
                  
                  # Ensure correct data type
                  if img_resized.dtype != np.uint8:
                      img_resized = img_resized.astype(np.uint8)
                  
                  # Place in combined image
                  y1, y2 = row * img_size, (row + 1) * img_size
                  x1, x2 = col * img_size, (col + 1) * img_size
                  combined_img[y1:y2, x1:x2] = img_resized
                  
                  # Add cluster ID text with better visibility
                  cv2.rectangle(combined_img, (x1, y1), (x1 + 60, y1 + 25), (255, 255, 255), -1)  # White background
                  cv2.putText(combined_img, f'C{cluster_id}', (x1 + 5, y1 + 18), 
                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)  # Black text
                             
              except Exception as e:
                  print(f"Error processing image for cluster {cluster_id}: {e}")
                  continue
          
          # Convert BGR to RGB for matplotlib (OpenCV uses BGR, matplotlib uses RGB)
          combined_img_rgb = cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB)
          self.ax2.imshow(combined_img_rgb)

      plt.tight_layout()

    
    def run(self):
        """Start the visualization"""
        try:
            plt.show()
        except KeyboardInterrupt:
            print("Shutting down...")
        finally:
            plt.close()

def print_detection_info():
    """Simple callback to print detection info without visualization"""
    def callback(msg):
        logger.info(f"\n=== Detection Update ===")
        logger.info(f"Timestamp: {msg.header.stamp}")
        logger.info(f"Number of casualties: {len(msg.casualties)}")
        
        for casualty in msg.casualties:
            logger.info(f"  Cluster {casualty.casualty_id}:")
            logger.info(f"    GPS: ({casualty.location.latitude:.6f}, {casualty.location.longitude:.6f})")
            logger.info(f"    Has image: {'Yes' if casualty.image.data else 'No'}")
    
    rospy.init_node('detection_listener')
    rospy.Subscriber('/loc/locations', CasualtyFixArray, callback)
    
    logger.info("Listening to /loc/locations (text output)...")
    logger.info("Press Ctrl+C to stop.")
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        logger.info("Shutting down...")

def main():
    import sys
    
    mode = 'visual'
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    
    if mode == 'text':
        logger.info("Running in text mode...")
        logger.info()
    else:
        logger.info("Running in visual mode...")
        logger.info("Use 'python3 visualize_detections.py text' for text-only output")
        
        try:
            visualizer = DetectionVisualizer()
            visualizer.run()
        except Exception as e:
            logger.info(f"Visualization failed: {e}")
            logger.info("Falling back to text mode...")
            print_detection_info()

if __name__ == '__main__':
    main()
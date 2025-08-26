cmake_minimum_required(VERSION 3.0.2)
project(dtc_msgs)

find_package(catkin REQUIRED COMPONENTS
  std_msgs
  sensor_msgs
  nav_msgs
  geometry_msgs
  message_generation
)

add_message_files(
  FILES
  NavSatFixArray.msg
  NavSatFixTimeAgo.msg
  RateTimeAgo.msg
  BoolTimeAgo.msg
  Uint8TimeAgo.msg
  ScoreCard.msg
  RespirationRate.msg
  HeartRate.msg
  GroundDetection.msg
  GroundImage.msg
  JackalStatus.msg
  FalconStatus.msg
  CasualtyFix.msg
  CasualtyFixArray.msg
)
catkin_python_setup()

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  nav_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS 
    std_msgs 
    sensor_msgs 
    nav_msgs 
    geometry_msgs 
    message_runtime
)

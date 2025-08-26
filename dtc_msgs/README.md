# DTC Msgs

This repo should contain any custom message types used for the DTC project. This will make `MOCHA` configuration significantly easier. 

## Parser 
I wrote a python parser for the `ScoreCard` to conver it to a dict. I still need to implement the time conversion but I'm not sure what format DARPA wants. Here is the usage:
```
from dtc_msgs.msg import ScoreCard
from dtc_msgs.scorecard_parser import ScoreCardParser

msg = ScoreCard()
d = ScoreCardParser.parse_to_dict(msg)
```

## Message Definitions

### ScoreCard
This uses the exact layout of the scorecard from DARPA
```
std_msgs/Header header

uint32 casualty_id

dtc_msgs/NavSatFixTimeAgo location

dtc_msgs/BoolTimeAgo severe_hemorrhage
dtc_msgs/BoolTimeAgo respiratory_distress

dtc_msgs/RateTimeAgo heart_rate
dtc_msgs/RateTimeAgo respiratory_rate

# 0 => Normal, 1 => Wound, 2 => Not Testable (NT)
dtc_msgs/Uint8TimeAgo trauma_head
# 0 => Normal, 1 => Wound, 2 => Not Testable (NT)
dtc_msgs/Uint8TimeAgo trauma_torso
# 0 => Normal, 1 => Wound, 2 => Amputation, 3 => NT
dtc_msgs/Uint8TimeAgo trauma_upper_extremity
# 0 => Normal, 1 => Wound, 2 => Amputation, 3 => NT
dtc_msgs/Uint8TimeAgo trauma_lower_extremity

# 0 => Open, 1 => Closed, 2 => NT
dtc_msgs/Uint8TimeAgo alertness_ocular
# 0 => Normal, 1 => Abnormal, 2 => Absent, 3 => NT
dtc_msgs/Uint8TimeAgo alterness_verbal
# 0 => Normal, 1 => Abnormal, 2 => Absent, 3 => NT
dtc_msgs/Uint8TimeAgo alterness_motor
```
### BoolTimeAgo
```
bool value
std_msgs/Time time_ago
```
### RateTimeAgo
```
float64 value
std_msgs/Time time_ago
```
### Uint8TimeAgo
```
uint8 value
std_msgs/Time time_ago
```
### NavSatFixTimeAgo
```
sensor_msgs/NavSatFix location
std_msgs/Time time_ago
```
### NavSatFixArray
```
std_msgs/Header header
sensor_msgs/NavSatFix[] coordinates
```
### RespirationRate
```
std_msgs/Header header
std_msgs/UInt64 casualty_id
std_msgs/Float64 rate
```
### HeartRate
```
std_msgs/Header header
std_msgs/UInt64 casualty_id
std_msgs/Float64 rate
```
### JackalStatus
```
std_msgs/Header header

# 0 is not running, 1 is running
bool controller
bool rgb
bool thermal
bool ouster
bool gps
bool rtk
bool radar
```
### FalconStatus
```
std_msgs/Header header

# 0 is not running, 1 is running
bool rgb
bool thermal
bool gps
bool rtk
```
### GroundDetection
```
std_msgs/Header header

sensor_msgs/NavSatFix gps

std_msgs/UInt64 casualty_id

std_msgs/String whisper

dtc_msgs/RespirationRate acconeer
dtc_msgs/RespirationRate mtts_resp

dtc_msgs/HeartRate mtts_hr
```
### GroundImage
```
std_msgs/Header header

sensor_msgs/NavSatFix gps

sensor_msgs/CompressedImage image1
sensor_msgs/CompressedImage image2
sensor_msgs/CompressedImage image3

std_msgs/UInt64 casualty_id
```

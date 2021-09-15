#!/bin/bash
rosservice call /pylon_camera_node/set_image_encoding “value: mono8”
rosservice call /pylon_camera_node/set_balance_white_auto “value: 1”


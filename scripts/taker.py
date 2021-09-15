#!/usr/bin/env python
import rospy
from std_msgs import msg
from std_msgs.msg import String
from teal_camera.msg import ur5_camera

def talker():
    pub = rospy.Publisher('chatter', ur5_camera, queue_size=10)
    rospy.init_node('camera_ur5_talker', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        location =  ur5_camera()
        _x = int(input("Input your x:"))
        _y = int(input("Input your y:"))
        _angle = float(input("Angle"))
        location.x = _x
        location.y = _y
        location.angle = _angle
        rospy.loginfo(location)
        pub.publish(location)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass

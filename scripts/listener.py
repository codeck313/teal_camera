#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from teal_camera.msg import ur5_camera
def callback(data):
    rospy.loginfo(rospy.get_caller_id() + "I heard %d %d %f", data.x,data.y,data.angle)

def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('camera_ur5_listener', anonymous=True)

    rospy.Subscriber("chatter", ur5_camera, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()

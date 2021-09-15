#!/usr/bin/env python
from __future__ import print_function

import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from linetimer import CodeTimer

import actionlib
from teal_camera.msg import ur5_cameraAction, ur5_cameraResult, ur5_cameraFeedback


class image_converter:

    def __init__(self, func, img_small, img_template):
        self.functionImage = func
        self.img0 = img_small
        self.template = img_template
        self.image_pub = rospy.Publisher("vision_ur5", Image, queue_size=50)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            "pylon_camera_node/image_raw", Image, self.callback)

        self.a_server = actionlib.SimpleActionServer(
            "camera_server", ur5_cameraAction, execute_cb=self.callbackexec, auto_start=False)
        self.a_server.start()

    def callback(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "mono8")
            # cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
            # if cv2.waitKey(0) == 32:
            #     cv2.imshow("Output", self.functionImage(
            #         self.img0, self.template, cv_image))
        except CvBridgeError as e:
            print(e)

    def callbackexec(self, goal):

        success = True
        feedback = ur5_cameraFeedback()
        result = ur5_cameraResult()
        rospy.loginfo('Executing, as order is %i' % (goal.start))
        r = rospy.Rate(1)

        while (goal.start == 1) and success is True:
            if self.a_server.is_preempt_requested():
                self.a_server.set_preempted()
                success = False
                break
            with CodeTimer():
                self.cv_image, _x, _y, _angle = self.functionImage(
                    self.img0, self.template, self.cv_image)
            feedback.x = _x
            feedback.y = _y
            feedback.angle = _angle
            result.x = _x
            result.y = _y
            result.angle = _angle
            self.a_server.publish_feedback(feedback)
            if success:
                self.a_server.set_succeeded(result)
                goal.start = 0
            try:
                self.image_pub.publish(
                    self.bridge.cv2_to_imgmsg(self.cv_image, "mono8"))
            except CvBridgeError as e:
                print(e)
            r.sleep()


def main(args):
    ic = image_converter()
    rospy.init_node('image_converter', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)

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


class image_converter:

    def __init__(self, func, img_small, img_template):
        self.functionImage = func
        self.img0 = img_small
        self.template = img_template
        self.image_pub = rospy.Publisher("image_topic_2", Image, queue_size=50)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            "pylon_camera_node/image_raw", Image, self.callback)

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "mono8")
            # cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
            # if cv2.waitKey(0) == 32:
            #     cv2.imshow("Output", self.functionImage(
            #         self.img0, self.template, cv_image))
            with CodeTimer():
                cv_image = self.functionImage(
                    self.img0, self.template, cv_image)
        except CvBridgeError as e:
            print(e)
        try:
            self.image_pub.publish(
                self.bridge.cv2_to_imgmsg(cv_image, "mono8"))
        except CvBridgeError as e:
            print(e)


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

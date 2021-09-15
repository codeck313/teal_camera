#! /usr/bin/env python

import rospy

import actionlib

from teal_camera.msg import ur5_cameraAction,ur5_cameraResult,ur5_cameraFeedback

class cameraActionServer():

    def __init__(self,func):
        self.function = func
        self.a_server = actionlib.SimpleActionServer(
            "camera_server", ur5_cameraAction, execute_cb=self.callbackexec, auto_start=False)
        self.a_server.start()

    def callbackexec(self, goal):

        success = True
        feedback = ur5_cameraFeedback()
        result = ur5_cameraResult()
        rospy.loginfo('Executing, as order is %i' % ( goal.start))
        r = rospy.Rate(1)

        while (goal.start == 1) and success is True:
            if self.a_server.is_preempt_requested():
                self.a_server.set_preempted()
                success = False
                break
            _x,_y,_angle = self.function()
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
            r.sleep()

def printFunc():
    return 233,21,785.522

if __name__ == "__main__":
    rospy.init_node("camera_server")
    s = cameraActionServer(printFunc)
    rospy.spin()


# Camera - Machine tending robot using Opencv using Ros framework
## Overview 
The aim of the R&D project was to get the UR5 to pick the object from the workspace given in any orientation and position and drop it in the desired drop location.
The training for the object will be done using just "one" master image and then it needs to be able to detect the orientation. To achieve this we had to use location, scale and angle invariant method. 

## Flow of Operations

![sequence-dig-teal-ur5](https://user-images.githubusercontent.com/23121752/133966064-0b372d6c-b72e-4159-8811-79fbe0d3e7ac.png)

## Node Setup
![rosgraph](https://user-images.githubusercontent.com/23121752/133967200-32a19bc5-a691-4771-944f-db765f04240a.png)


## Setup
### Pre-requisite 
1. Have ROS Melodic installed on Ubuntu 18.
2. Download the following repositories <br>
Updated RobotIQ 2f Gripper: 	`git clone https://github.com/codeck313/ur5-robotiq` <br>
Robot Controller Package:  `git clone https://github.com/codeck313/teal-ur5`<br>
Camera Node Package: `git clone https://github.com/codeck313/teal_camera`
3. Use the *requirements.txt* file from [*teal_camera*](https://github.com/codeck313/teal_camera/blob/master/requirements.txt) to install all needed python packages.
`pip install -r requirements.txt` 
4. Install [UR5_driver](https://github.com/UniversalRobots/Universal_Robots_ROS_Driver) in your system
5. Update the firmware of PolyScope to 3.7+

6. [External control package](https://github.com/UniversalRobots/Universal_Robots_ROS_Driver/blob/master/ur_robot_driver/resources/externalcontrol-1.0.5.urcap) in URSoftware
7. Install [Pylon Camera Suite](https://www.baslerweb.com/en/products/software/basler-pylon-camera-software-suite/)
8. Install [pylon drivers](https://github.com/basler/pylon-ros-camera) for ROS
9. Do  `catkin build`  in your [catkin workspace](https://wiki.ros.org/catkin/Tutorials/create_a_workspace)

### Configuration and starting the nodes
 
1. Configure IP settings in Installation setup for external control

3. Create a function in PolyScope such that it is calling the external control function in a loop.

4. Now launching the UR5Drivers in ROS  
`roslaunch ur_robot_driver <robot_type>_bringup.launch robot_ip:=192.1.1.2`

5. Start the program in PolyScope

6. For jogging joints manually use: 
`rosrun rqt_joint_trajectory_controller rqt_joint_trajectory_controller`

7. For jogging using MoveIt use
`roslaunch ur5_moveit_config ur5_moveit_planning_execution.launch'` and <br>
`roslaunch ur5_moveit_config moveit_rviz.launch`

8.  To run everything together use ur5_teal_bringup package. To start the whole process you can use the `roslaunch ur5_teal_bringup ur5_teal.launch` command.<br>
Output will be like this:
![launch](https://user-images.githubusercontent.com/23121752/133966154-d0fe8ac1-9f41-4c32-9733-67c85db46398.png)


9. Next we need to start the pylon camera node `rosrun pylon_camera pylon_camera_node` <br>
Output will be like this:
![pylon](https://user-images.githubusercontent.com/23121752/133966219-5cefdcc1-32e3-4cd1-8e60-1d3e77326257.png)


10. Now we can start the camera detection node `rosrun teal_camera camera_detect.py` <br>
Output will be like this:
![camera](https://user-images.githubusercontent.com/23121752/133966256-aff6fa59-58d3-469d-b563-4ecf0439b39d.png)


11. Lastly we need to start the UR5 controller node `rosrun ur5_teal_bringup robo.py`<br>
Output will be like this:
![robo_controller](https://user-images.githubusercontent.com/23121752/133966278-4957626d-0c9d-445a-947e-87a93b98399d.png)

 

## Things to change for a new gripper
The following changes need to be done in case you are attaching a different gripper. For example case we are trying to robotiq 2f gripper 

1. Add a param `connected_to` in *robotiq_arg2f_85_model_macro.xacro* file. This will help us link the base of gripper to tool0 of the arm.
Like this:
``` xml
<xacro:macro name="robotiq_arg2f_85" params="prefix connected_to">
<xacro:robotiq_arg2f_base_link prefix="${prefix}" connected_to="${connected_to}"/>
<xacro:finger_links prefix="${prefix}" fingerprefix="left" stroke="85"/>
<xacro:finger_links prefix="${prefix}" fingerprefix="right" stroke="85"/>
<xacro:finger_joint prefix="${prefix}"/>
<xacro:right_outer_knuckle_joint prefix="${prefix}"/>
<xacro:robotiq_arg2f_transmission prefix="${prefix}"/>
</xacro:macro>
</robot>
```
	
2. Add `connected_to` param in *robotiq_arg2f.xacro* and  *robotiq_arg2f_85_model.xacro* files in *robotiq_2f_85_gripper_visualization*  . And create a joint to join the base link and robot’s tool0 aka the `connected_to` param value.
Like this:
- *robotiq_arg2f.xacro* 

```xml
<xacro:macro name="robotiq_arg2f_base_link" params="prefix connected_to">
<joint name="${prefix}_tool_joint" type="fixed">
<!-- The parent link must be read from the robot model it is attached to. -->
<parent link="${connected_to}"/>
<child link="${prefix}robotiq_arg2f_base_link"/>
<!-- The tool is directly attached to the flange. -->
<origin rpy="0 0 0" xyz="0 0 0"/>
</joint>
....
```

- *robotiq_arg2f_85_model.xacro*
```xml
<xacro:robotiq_arg2f_85 prefix="" connected_to="world"/>
```
	
4. Create the file ur5_arm.urdf.xacro in to attach the gripper to the arm. Pass `connected_to`  param with `tool0` as the value.

## Additional things to tweak if required
1. To be able to run the robo code at 50% speed you need to add `allowed_execution_duration_scaling: 2.0`  in the *ur5_teal_config/config/ros_controllers.yaml*  file to get rid of time out.

2. Add `<arg name="execution_type" default="interpolate"/>` in *robot_moveit_controller_manager.launch.xml* in *ur5_teal_config/launch* to resolve not being able to open demo_gazebo.launch

3.  Install MoveIt to reconfigure moveit files 
`sudo apt install ros-melodic-moveit` 

4. In case you aren't able to run a script make it is executable. If not you need to `chmod +x robo.py`

5. To find the coordinates your TCP is at currently you can use `rosrun tf tf_echo tool0 flange` or use Rviz > Motion Planning > Scene Robot > Links > robotiq_arg2f_base_link.
![tcp](https://user-images.githubusercontent.com/23121752/133966423-27409bfb-00bc-4cf3-88da-6de1f4addbf2.png)


6. If you want to rectify wrong planning you can put path constrains in rviz to make sure it doesn't happen.
7. You can use the provided perspective file in teal-ur5 to get the testing rviz setup

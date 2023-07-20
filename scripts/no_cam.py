#!/usr/bin/env python3

import numpy as np
import rospy
from CMU_UNet_Node.msg import line_2pts, line_list
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import tf

USE_ODOM = True

forward_line_list = line_list()
forward_line_list.lines.append(line_2pts(x1=10, y1=1.5, x2=0, y2=1.5))
forward_line_list.lines.append(line_2pts(x1=10, y1=-1.5, x2=0, y2=-1.5))


class Node:
    '''
    When initialized, store the lines as perfectly forward wrt the robot

    For each frame, update the line positions with the odometry (adjust the angle wrt the robot)
    and add some noise to the positions
    '''
    def __init__(self):
        self.sub_odom = rospy.Subscriber(
            "/odometry/filtered", Odometry, self.odom_callback, queue_size=1)

        self.pub_lines = rospy.Publisher(
            "/unet_lines", line_list, queue_size=1)

        self.pub_left_line = rospy.Publisher(
            "/left_line", Marker, queue_size=1)

        self.pub_right_line = rospy.Publisher(
            "/right_line", Marker, queue_size=1)

        # Line directly to the left and to the right of the robot
        self.lines = [[10, 1.5, 0, 1.5], [10, -1.5, 0, -1.5]]

        self.last_yaw = None
        self.lastx, self.lasty = None, None

        self.frame_count = 0

    def odom_callback(self, msg):
        self.frame_count += 1
        if self.frame_count != 5:
            return

        self.frame_count = 0

        if not USE_ODOM:
            msg = forward_line_list
        else:
            # Get the yaw angle
            euler = tf.transformations.euler_from_quaternion(
                [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                    msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
            yaw = euler[2]
            delta_yaw = 0 if self.last_yaw is None else yaw - self.last_yaw
            self.last_yaw = yaw

            # Get the x, y position
            x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
            delta_x = 0 if self.lastx is None else x - self.lastx
            delta_y = 0 if self.lasty is None else y - self.lasty
            self.lastx, self.lasty = x, y

            # Rotate the lines, which are in x1, y1, x2, y2 format
            for line in self.lines:
                # Swap for angle
                sy1, sx1, sy2, sx2 = line
                line[1] = sx1 * np.cos(delta_yaw) - sy1 * np.sin(delta_yaw)
                line[0] = sx1 * np.sin(delta_yaw) + sy1 * np.cos(delta_yaw)
                line[3] = sx2 * np.cos(delta_yaw) - sy2 * np.sin(delta_yaw)
                line[2] = sx2 * np.sin(delta_yaw) + sy2 * np.cos(delta_yaw)

                # Translate
                line[0] -= delta_x
                line[1] -= delta_y
                line[2] -= delta_x
                line[3] -= delta_y

            print(delta_yaw, delta_x, delta_y)

            # msg = forward_line_list

            # Add some noise to the lines
            for line in self.lines:
                x1, y1, x2, y2 = line
                line[0] = x1 + np.random.normal(0, 0.03)
                line[1] = y1 + np.random.normal(0, 0.03)
                line[2] = x2 + np.random.normal(0, 0.03)
                line[3] = y2 + np.random.normal(0, 0.03)

            msg = line_list()

            for line in self.lines:
                new_line = line_2pts(x1=line[0], y1=line[1], x2=line[2], y2=line[3])
                msg.lines.append(new_line)

        msg.header.stamp = rospy.Time.now()
        msg.num_lines = len(msg.lines)

        self.pub_lines.publish(msg)

        # Only display one marker for the left line
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.type = marker.LINE_STRIP
        marker.action = marker.ADD
        marker.scale.x = 0.05
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.id = 0
        marker.pose.position.x = 0
        marker.pose.position.y = 0
        marker.pose.position.z = 0
        marker.pose.orientation.x = 0
        marker.pose.orientation.y = 0
        marker.pose.orientation.z = 0
        marker.pose.orientation.w = 1
        marker.points.append(Point(x=msg.lines[0].x1, y=msg.lines[0].y1, z=0))
        marker.points.append(Point(x=msg.lines[0].x2, y=msg.lines[0].y2, z=0))
        self.pub_left_line.publish(marker)

        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.type = marker.LINE_STRIP
        marker.action = marker.ADD
        marker.scale.x = 0.05
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.id = 0
        marker.pose.position.x = 0
        marker.pose.position.y = 0
        marker.pose.position.z = 0
        marker.pose.orientation.x = 0
        marker.pose.orientation.y = 0
        marker.pose.orientation.z = 0
        marker.pose.orientation.w = 1
        marker.points.append(Point(x=msg.lines[1].x1, y=msg.lines[1].y1, z=0))
        marker.points.append(Point(x=msg.lines[1].x2, y=msg.lines[1].y2, z=0))
        self.pub_right_line.publish(marker)

        # # Remove all previous markers
        # marker_array = MarkerArray()
        # marker = Marker()
        # marker.header.frame_id = "base_link"
        # marker.action = marker.DELETEALL
        # marker_array.markers.append(marker)
        # self.pub_vis_lines.publish(marker_array)

        # # # Publish the lines as markers
        # # marker_array = MarkerArray()
        # # for i, line in enumerate(msg.lines):
        # #     marker = Marker()
        # #     marker.header.frame_id = "base_link"
        # #     marker.type = marker.LINE_STRIP
        # #     marker.action = marker.ADD
        # #     marker.scale.x = 0.05
        # #     marker.color.a = 1.0
        # #     marker.color.r = 1.0
        # #     marker.color.g = 0.0
        # #     marker.color.b = 0.0
        # #     marker.id = i
        # #     marker.lifetime = rospy.Duration(0.1)
        # #     marker.pose.position.x = 0
        # #     marker.pose.position.y = 0
        # #     marker.pose.position.z = 0
        # #     marker.pose.orientation.x = 0
        # #     marker.pose.orientation.y = 0
        # #     marker.pose.orientation.z = 0
        # #     marker.pose.orientation.w = 1

        # #     # NOTE: Swap X and Y for visualization (for some reason, they are swapped in the code)
        # #     marker.points.append(Point(x=line.y1, y=line.x1, z=0))
        # #     marker.points.append(Point(x=line.y2, y=line.x2, z=0))

        # #     marker_array.markers.append(marker)

        # # print("Publishing markers: ", marker_array)

        # # self.pub_vis_lines.publish(marker_array)


if __name__ == "__main__":
    rospy.init_node('dummy', log_level=rospy.INFO)

    node = Node()

    rospy.spin()

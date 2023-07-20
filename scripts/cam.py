#!/usr/bin/env python3

import numpy as np
import rospy
from CMU_UNet_Node.msg import line_2pts, line_list
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import tf
from sensor_msgs.msg import Image
from message_filters import ApproximateTimeSynchronizer, Subscriber
from config import IMAGE_TOPIC, DEPTH_TOPIC, CAMERA_INFO, DEPTH_CAMERA_INFO
from transforms import TfBuffer, Transformer
import cv2
from cv_bridge import CvBridge
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sensor_msgs.msg import PointCloud2
import sensor_msgs
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header


class Node:
    '''
    When initialized, store the lines as perfectly forward wrt the robot

    For each frame, update the line positions with the odometry (adjust the angle wrt the robot)
    and add some noise to the positions
    '''
    def __init__(self):
        self.pub_pointcloud = rospy.Publisher('/row_points', PointCloud2, queue_size=1)

        self.pub_lines = rospy.Publisher('/lines', line_list, queue_size=1)

        self.cv_bridge = CvBridge()

        self.transformer = Transformer(TfBuffer().get_tf_buffer())

        self.camera_info = rospy.wait_for_message(
            CAMERA_INFO, CameraInfo, timeout=1)
        self.depth_info = rospy.wait_for_message(
            DEPTH_CAMERA_INFO, CameraInfo, timeout=1)

        self.image_width = self.camera_info.width
        self.image_height = self.camera_info.height

        self.sub_image = Subscriber(IMAGE_TOPIC, Image)
        self.sub_depth = Subscriber(DEPTH_TOPIC, Image)
        self.ts = ApproximateTimeSynchronizer(
            [self.sub_image, self.sub_depth], queue_size=5, slop=0.1)
        self.ts.registerCallback(self.image_depth_callback)

    def image_depth_callback(self, image, depth):
        # Detect the black lines in the image
        cv_image = self.cv_bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')

        # region Line detection
        # gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        # edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        # minLineLength = 300
        # maxLineGap = 10
        # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)
        # print('lines', lines)
        # for x1, y1, x2, y2 in lines[0]:
        #     cv2.line(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # cv2.imwrite('houghlines.jpg', cv_image)
        # endregion

        # Segment out the black pixels (close to black)
        black_pixels = np.where(np.logical_and(
            cv_image[:, :, 0] < 10, cv_image[:, :, 1] < 10, cv_image[:, :, 2] < 10))

        if len(black_pixels[0]) == 0:
            return

        # Cluster the pixels
        db = DBSCAN(eps=10, min_samples=10).fit(np.array(black_pixels).T)

        # Color the pixels based on the cluster
        # separated_image = cv_image.copy()
        # colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
        # for i, (x, y) in enumerate(zip(*black_pixels)):
        #     separated_image[x, y] = colors[db.labels_[i]]
        # cv2.imwrite('separated_image.jpg', separated_image)

        # # Run PCA on each cluster
        # pca_image = cv_image.copy()
        # for i in range(max(db.labels_) + 1):
        #     cluster = np.array(black_pixels)[:, db.labels_ == i]

        #     if cluster.shape[0] < 2:
        #         continue

        #     pca = PCA(n_components=1)
        #     pca.fit(cluster.T)

        #     start_y, end_y = np.min(cluster[1]), np.max(cluster[1])
        #     start_x = pca.mean_[0] + pca.components_[0][0] * (start_y - pca.mean_[1]) / pca.components_[0][1]
        #     end_x = pca.mean_[0] + pca.components_[0][0] * (end_y - pca.mean_[1]) / pca.components_[0][1]
        #     cv2.line(pca_image, (int(start_y), int(start_x)), (int(end_y), int(end_x)), (0, 255, 0), 2)

        # cv2.imwrite('pca_image.jpg', pca_image)

        # Get the depth of the black lines
        depth_image = self.cv_bridge.imgmsg_to_cv2(depth, desired_encoding='passthrough')
        depth_image = np.array(depth_image, dtype=np.float32)

        # List of 3d points for each line
        lines_3d = []
        for i in range(max(db.labels_) + 1):
            cluster = np.array(black_pixels)[:, db.labels_ == i]

            if cluster.shape[0] < 2:
                continue

            cluster_depth = depth_image[cluster[0], cluster[1]]

            points = np.array([self.image_width - cluster[1], cluster[0], cluster_depth]).T

            # Remove elements with NAN
            points = points[~np.isnan(points).any(axis=1)]

            if len(points) == 0:
                continue

            points3d = self.transformer.transform_cam_to_base_arr(points)
            lines_3d.append(points3d)

        points = np.concatenate(lines_3d)
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'base_link'
        pointcloud = pc2.create_cloud_xyz32(header=header, points=points)
        self.pub_pointcloud.publish(pointcloud)

        # Publish the lines
        lines = line_list()
        lines.header.frame_id = 'base_link'
        lines.header.stamp = rospy.Time.now()

        # print('===========')
        lines.lines = []
        for line in lines_3d:
            # Run PCA on the XY points
            pca = PCA(n_components=1)

            pca.fit(line[:, :2])

            # Print 3 random points
            # print(line[np.random.randint(0, len(line))])
            # print(line[np.random.randint(0, len(line))])
            # print(line[np.random.randint(0, len(line))])

            start_x, end_x = 0, 10
            start_y = pca.mean_[1] + pca.components_[0][1] * (start_x - pca.mean_[0]) / pca.components_[0][0]
            end_y = pca.mean_[1] + pca.components_[0][1] * (end_x - pca.mean_[0]) / pca.components_[0][0]

            # print(start_x, start_y, end_x, end_y)

            lines.lines.append(line_2pts(x1=start_x, y1=start_y, x2=end_x, y2=end_y))

        lines.num_lines = len(lines.lines)
        self.pub_lines.publish(lines)


if __name__ == "__main__":
    rospy.init_node('dummy', log_level=rospy.INFO)

    node = Node()

    rospy.spin()

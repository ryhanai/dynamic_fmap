import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from pathlib import Path

from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy


URDF_FILE = str(Path.home() / 'miniconda3/envs/maniskill_ros2/lib/python3.10/site-packages/mani_skill/assets/robots/panda/panda_v2.urdf')


# For launching RViz2
# import os
# from ament_index_python.packages import get_package_share_directory
# from launch import LaunchDescription
# from launch_ros.actions import Node

# def generate_launch_description():
#     package_dir = get_package_share_directory("ros2_first_test")
#     urdf = os.path.join(package_dir, "urdf", "first_robot.urdf")

#     return LaunchDescription([

#         Node(
#             package='robot_state_publisher',
#             executable='robot_state_publisher',
#             name='robot_state_publisher',
#             arguments=[urdf]),

#         Node(
#             package='joint_state_publisher_gui',
#             executable='joint_state_publisher_gui',
#             name='joint_state_publisher_gui',
#             arguments=[urdf]),
            
#         Node(
#             package="rviz2",
#             executable="rviz2",
#             name="rviz2"),
#     ])


def main():
    rclpy.init()
    node = Node('maniskill_viewer_client')

    qos = QoSProfile(depth=1)
    qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
    qos.reliability = QoSReliabilityPolicy.RELIABLE
    publisher = node.create_publisher(String, 'robot_description', qos)

    msg = String()
    with open(URDF_FILE, "r") as f:
        msg.data = f.read()

    publisher.publish(msg)
    node.get_logger().info("Published URDF to /robot_description")
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
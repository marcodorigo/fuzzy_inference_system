import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32, Float32MultiArray, Float64MultiArray
import math
import numpy as np  # For matrix operations
from visualization_msgs.msg import Marker


class DistanceMetricsNode(Node):

    def __init__(self):
        super().__init__('ee_distance_metrics_node')

        # Initialize parameters with default values
        self.obstacle_center = [0.0, 0.0, 0.0]
        self.obstacle_radius = 0.0
        self.workspace_radius = 0.0
        self.target_position = [0.0, 0.0, 0.0]

        # Publishers
        self.distance_metrics_publisher = self.create_publisher(
            PoseStamped, '/distance_metrics', 10
        )
        self.manipulability_publisher = self.create_publisher(
            Float32, '/manipulability_metric', 10
        )

        # Subscribers to parameter topics
        self.create_subscription(
            Float32MultiArray,
            '/obstacle_center',
            self.obstacle_center_callback,
            10
        )
        self.create_subscription(
            Float32,
            '/obstacle_radius',
            self.obstacle_radius_callback,
            10
        )
        self.create_subscription(
            Float32,
            '/workspace_radius',
            self.workspace_radius_callback,
            10
        )
        self.create_subscription(
            Float32MultiArray,
            '/target_position',
            self.target_position_callback,
            10
        )

        # Subscribe to the pose topic
        self.create_subscription(
            PoseStamped,
            '/admittance_controller/pose_debug',
            self.pose_callback,
            10
        )

        # Subscribe to the Jacobian topic
        self.create_subscription(
            Float64MultiArray,
            '/virtual_wrench_commander/jacobian',
            self.jacobian_callback,
            10
        )

        # Log initialization
        self.get_logger().info("✅ EE Distance Metrics Node started and ready.")

    def obstacle_center_callback(self, msg: Float32MultiArray):
        self.obstacle_center = list(msg.data)
        self.get_logger().info(f"Updated obstacle_center: {[round(x, 3) for x in self.obstacle_center]}")

    def obstacle_radius_callback(self, msg: Float32):
        self.obstacle_radius = msg.data  # Directly assign the float value
        self.get_logger().info(f"Updated obstacle_radius: {round(self.obstacle_radius, 3)}")

    def workspace_radius_callback(self, msg: Float32):
        self.workspace_radius = msg.data  # Directly assign the float value
        self.get_logger().info(f"Updated workspace_radius: {round(self.workspace_radius, 3)}")

    def target_position_callback(self, msg: Float32MultiArray):
        self.target_position = list(msg.data)
        self.get_logger().info(f"Updated target_position: {[round(x, 3) for x in self.target_position]}")

    def pose_callback(self, msg: PoseStamped):
        ee_position = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]

        # Distance to obstacle surface
        dist_to_obstacle_center = self.euclidean_distance(ee_position, self.obstacle_center)
        dist_to_obstacles = max(0.0, dist_to_obstacle_center - self.obstacle_radius)

        # Distance to workspace boundary
        dist_from_base = self.euclidean_distance(ee_position, [0.0, 0.0, 0.0])
        dist_to_workspace = max(0.0, self.workspace_radius - dist_from_base)

        # Distance to target
        dist_to_target = self.euclidean_distance(ee_position, self.target_position)

        # Normalize and clamp distances
        dist_to_obstacles = min(1.0, dist_to_obstacles / 0.1)
        dist_to_workspace = min(1.0, dist_to_workspace / 0.1)
        dist_to_target = min(1.0, dist_to_target / 0.1)

        # Debugging log
        # self.get_logger().debug(
        #     f"Computed distances - Obstacles: {dist_to_obstacles}, Workspace: {dist_to_workspace}, Target: {dist_to_target}"
        # )

        # Publish the distances
        distances_msg = PoseStamped()
        distances_msg.header.stamp = self.get_clock().now().to_msg()
        distances_msg.header.frame_id = 'base_link'
        distances_msg.pose.position.x = dist_to_obstacles
        distances_msg.pose.position.y = dist_to_workspace
        distances_msg.pose.position.z = dist_to_target
        self.distance_metrics_publisher.publish(distances_msg)

    def jacobian_callback(self, msg: Float32MultiArray):
        # Convert the Jacobian data into a 6x6 matrix
        jacobian = np.array(msg.data).reshape((6, 6))

        # Compute the manipulability metric (determinant of J * J^T) - Volume of the manipulability ellipsoid
        manipulability = np.sqrt(np.linalg.det(jacobian @ jacobian.T))

        # Normalize and clamp the manipulability metric to the range [0, 1] for values 0.02 to 0.08 (found experimentally)
        normalized_manipulability = (manipulability - 0.02) / (0.08 - 0.02)
        normalized_manipulability = min(1.0, max(0.0, normalized_manipulability))

        # Compute manipulability as the ratio of the smallest to largest singular value - Condition number approach
        # singular_values = np.linalg.svd(jacobian, compute_uv=False)
        # if singular_values[0] > 1e-6:  # Avoid division by zero
        #     manipulability = singular_values[-1] / singular_values[0]
        # else:
        #     manipulability = 0.0
        # normalized_manipulability = 

        # Publish the normalized manipulability metric
        manipulability_msg = Float32()
        manipulability_msg.data = normalized_manipulability
        self.manipulability_publisher.publish(manipulability_msg)

        # Debugging log
        self.get_logger().debug(f"Computed manipulability: {manipulability}, Normalized: {normalized_manipulability}")

    @staticmethod
    def euclidean_distance(p1, p2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def main(args=None):
    rclpy.init(args=args)
    node = DistanceMetricsNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

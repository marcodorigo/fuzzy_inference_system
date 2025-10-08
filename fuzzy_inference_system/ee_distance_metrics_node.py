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
        self.spherical_obstacles = []  # List of spherical obstacles
        self.cylinder_base = {"center": [0.0, 0.0, 0.0], "radius": 0.0, "height": 0.0}
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
            '/spherical_obstacles',
            self.spherical_obstacles_callback,
            10
        )
        self.create_subscription(
            Float32MultiArray,
            '/cylinder_base',
            self.cylinder_base_callback,
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
            Float32MultiArray,
            '/virtual_wrench_commander/jacobian',
            self.jacobian_callback,
            10
        )

        # Log initialization
        self.get_logger().info("Distance Metrics Node started and ready.")

    def spherical_obstacles_callback(self, msg: Float32MultiArray):
        data = list(msg.data)
        self.spherical_obstacles = [
            {"center": data[i:i+3], "radius": data[i+3]}
            for i in range(0, len(data), 4)
        ]

    def cylinder_base_callback(self, msg: Float32MultiArray):
        data = list(msg.data)
        self.cylinder_base = {
            "center": data[:3],
            "radius": data[3],
            "height": data[4]
        }

    def workspace_radius_callback(self, msg: Float32):
        self.workspace_radius = msg.data

    def target_position_callback(self, msg: Float32MultiArray):
        self.target_position = list(msg.data)

    def pose_callback(self, msg: PoseStamped):
        ee_position = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]

        # Distance to spherical obstacles
        dist_to_obstacles = float('inf')
        for obstacle in self.spherical_obstacles:
            dist_to_center = self.euclidean_distance(ee_position, obstacle["center"])
            dist_to_surface = max(0.0, dist_to_center - obstacle["radius"])
            dist_to_obstacles = min(dist_to_obstacles, dist_to_surface)

        # Distance to cylindrical base
        dist_to_cylinder = self.distance_to_cylinder(ee_position, self.cylinder_base)

        # Combine distances to obstacles and cylinder
        dist_to_obstacles = min(dist_to_obstacles, dist_to_cylinder)

        # Distance to workspace boundary
        dist_from_base = self.euclidean_distance(ee_position, [0.0, 0.0, 0.0])
        dist_to_workspace = max(0.0, self.workspace_radius - dist_from_base)

        # Distance to target
        dist_to_target = self.euclidean_distance(ee_position, self.target_position)

        # Normalize and clamp distances
        dist_to_obstacles = min(1.0, dist_to_obstacles / 0.1)
        dist_to_workspace = min(1.0, dist_to_workspace / 0.1)
        dist_to_target = min(1.0, dist_to_target / 0.1)

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

        # Publish the normalized manipulability metric
        manipulability_msg = Float32()
        manipulability_msg.data = float(normalized_manipulability)  # Convert numpy scalar to Python float
        self.manipulability_publisher.publish(manipulability_msg)

    @staticmethod
    def euclidean_distance(p1, p2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

    @staticmethod
    def distance_to_cylinder(point, cylinder):
        # Calculate the distance from a point to the surface of a cylinder
        base_center = cylinder["center"]
        radius = cylinder["radius"]
        height = cylinder["height"]

        # Project the point onto the cylinder's axis
        dx, dy = point[0] - base_center[0], point[1] - base_center[1]
        dist_to_axis = math.sqrt(dx**2 + dy**2)
        dist_to_side = max(0.0, dist_to_axis - radius)

        # Check height bounds
        z = point[2]
        if z < base_center[2]:
            dist_to_top_bottom = base_center[2] - z
        elif z > base_center[2] + height:
            dist_to_top_bottom = z - (base_center[2] + height)
        else:
            dist_to_top_bottom = 0.0

        # Return the minimum distance
        return math.sqrt(dist_to_side**2 + dist_to_top_bottom**2)


def main(args=None):
    rclpy.init(args=args)
    node = DistanceMetricsNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

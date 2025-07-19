import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import math
from visualization_msgs.msg import Marker


class DistanceMetricsNode(Node):

    def __init__(self):
        super().__init__('ee_distance_metrics_node')

        # Define static obstacle as a sphere
        self.obstacle_center = [0.5, 0.15, 0.7]  # in meters
        self.obstacle_radius = 0.05  # meters

        # Define workspace radius (850 mm)
        self.workspace_radius = 0.85  # meters

        # Define target position
        self.target_position = [0.7, 0.15, 0.5]  # in meters

        # Publisher for distance metrics
        self.distance_metrics_publisher = self.create_publisher(
            PoseStamped, '/distance_metrics', 10
        )

        # Subscribe to the pose topic
        self.subscription = self.create_subscription(
            PoseStamped,
            '/admittance_controller/pose_debug',
            self.pose_callback,
            10
        )

        # Log initialization
        self.get_logger().info("✅ EE Distance Metrics Node started and ready.")
        self.get_logger().info(f"Obstacle at {self.obstacle_center} with radius {self.obstacle_radius}")
        self.get_logger().info(f"Target position set to {self.target_position}")
        self.get_logger().info(f"Workspace radius: {self.workspace_radius}")

        '''
        # Marker publisher for visualization
        self.marker_pub = self.create_publisher(Marker, 'visualization_markers', 10)
        self.timer = self.create_timer(1.0, self.publish_markers)  # Publish markers every second
        '''

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

        # Publish the distances
        distances_msg = PoseStamped()
        distances_msg.header.stamp = self.get_clock().now().to_msg()
        distances_msg.header.frame_id = 'base_link'
        distances_msg.pose.position.x = dist_to_obstacles
        distances_msg.pose.position.y = dist_to_workspace
        distances_msg.pose.position.z = dist_to_target
        self.distance_metrics_publisher.publish(distances_msg)


    @staticmethod
    def euclidean_distance(p1, p2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

    '''
    def publish_markers(self):
        frame_id = "base_link"  # Or your robot's base frame
        ns = "visualization"

        # === Obstacle marker ===
        obstacle_marker = Marker()
        obstacle_marker.header.frame_id = frame_id
        obstacle_marker.header.stamp = self.get_clock().now().to_msg()
        obstacle_marker.ns = ns
        obstacle_marker.id = 0
        obstacle_marker.type = Marker.SPHERE
        obstacle_marker.action = Marker.ADD
        obstacle_marker.pose.position.x = self.obstacle_center[0]
        obstacle_marker.pose.position.y = self.obstacle_center[1]
        obstacle_marker.pose.position.z = self.obstacle_center[2]
        obstacle_marker.scale.x = self.obstacle_radius * 2
        obstacle_marker.scale.y = self.obstacle_radius * 2
        obstacle_marker.scale.z = self.obstacle_radius * 2
        obstacle_marker.color.r = 1.0
        obstacle_marker.color.g = 0.0
        obstacle_marker.color.b = 0.0
        obstacle_marker.color.a = 0.8

        # === Target marker ===
        target_marker = Marker()
        target_marker.header.frame_id = frame_id
        target_marker.header.stamp = self.get_clock().now().to_msg()
        target_marker.ns = ns
        target_marker.id = 1
        target_marker.type = Marker.SPHERE
        target_marker.action = Marker.ADD
        target_marker.pose.position.x = self.target_position[0]
        target_marker.pose.position.y = self.target_position[1]
        target_marker.pose.position.z = self.target_position[2]
        target_marker.scale.x = 0.05
        target_marker.scale.y = 0.05
        target_marker.scale.z = 0.05
        target_marker.color.r = 0.0
        target_marker.color.g = 1.0
        target_marker.color.b = 0.0
        target_marker.color.a = 1.0
        
        # === Workspace boundary marker (wireframe sphere) ===
        workspace_marker = Marker()
        workspace_marker.header.frame_id = frame_id
        workspace_marker.header.stamp = self.get_clock().now().to_msg()
        workspace_marker.ns = ns
        workspace_marker.id = 2
        workspace_marker.type = Marker.SPHERE
        workspace_marker.action = Marker.ADD
        workspace_marker.pose.position.x = 0.0
        workspace_marker.pose.position.y = 0.0
        workspace_marker.pose.position.z = 0.0
        workspace_marker.scale.x = self.workspace_radius * 2
        workspace_marker.scale.y = self.workspace_radius * 2
        workspace_marker.scale.z = self.workspace_radius * 2
        workspace_marker.color.r = 0.0
        workspace_marker.color.g = 0.0
        workspace_marker.color.b = 1.0
        workspace_marker.color.a = 0.1  # Transparent blue
        
        # Publish all markers
        self.marker_pub.publish(obstacle_marker)
        self.marker_pub.publish(target_marker)
        self.marker_pub.publish(workspace_marker)
    '''

def main(args=None):
    rclpy.init(args=args)
    node = DistanceMetricsNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

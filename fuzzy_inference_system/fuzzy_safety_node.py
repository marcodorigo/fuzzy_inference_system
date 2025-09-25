import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32
from fuzzylogic.classes import Domain, Rule
from fuzzylogic.functions import gauss, triangular, trapezoid
from fuzzylogic.hedges import very
import numpy as np
from itertools import product

class FuzzySafetyNode(Node):
    def __init__(self):
        super().__init__('fuzzy_safety_node')

        # Initialize fuzzy safety system
        self.fuzzy_system = FuzzySafetySystem()

        # Initialize variables to store input values
        self.dist_to_obstacles = 0.0
        self.dist_to_workspace = 0.0
        self.dist_to_target = 0.0
        self.manipulability = 0.0

        # Subscriber to distances topic
        self.subscription = self.create_subscription(
            PoseStamped,
            '/distance_metrics',
            self.distance_callback,
            10
        )

        # Subscriber to manipulability metric topic
        self.manipulability_subscription = self.create_subscription(
            Float32,
            '/manipulability_metric',
            self.manipulability_callback,
            10
        )

        # Publisher for safety coefficient
        self.safety_publisher = self.create_publisher(Float32, '/safety_coefficient', 10)




        # Timer to periodically compute and publish the safety coefficient
        self.timer = self.create_timer(0.1, self.compute_and_publish_safety)

        # Flag to control remapping
        self.remap_flag = 0  # Set to 1 to enable remapping, 0 to disable

        self.get_logger().info("✅ Fuzzy Safety Node started and ready.")

    def manipulability_callback(self, msg: Float32):
        self.manipulability = msg.data
        # self.get_logger().debug(f"Updated manipulability: {self.manipulability}")

    def distance_callback(self, msg: PoseStamped):
        self.dist_to_obstacles = msg.pose.position.x
        self.dist_to_workspace = msg.pose.position.y
        self.dist_to_target = msg.pose.position.z
        # self.get_logger().debug(
        #     f"Updated distances - Obstacles: {self.dist_to_obstacles}, "
        #     f"Workspace: {self.dist_to_workspace}, Target: {self.dist_to_target}"
        # )

    def compute_and_publish_safety(self):
        # Compute safety coefficient using fuzzy logic
        values = {
            self.fuzzy_system.targ_dist: self.dist_to_target,
            self.fuzzy_system.ws_dist: self.dist_to_workspace,
            self.fuzzy_system.obs_dist: self.dist_to_obstacles,
            self.fuzzy_system.manipulability: self.manipulability
        }
        safety_coefficient = self.fuzzy_system.compute_safety(values)

        # Conditionally remap the safety coefficient
        if self.remap_flag == 1:
            # Remap safety coefficient from [0.16, 0.5] to [0.01, 0.99]
            remapped_safety_coefficient = 0.01 + (safety_coefficient - 0.16) * (0.99 - 0.01) / (0.5 - 0.16)
            remapped_safety_coefficient = max(0.01, min(0.99, remapped_safety_coefficient))  # Clamp to [0.01, 0.99]
        else:
            remapped_safety_coefficient = safety_coefficient  # No remapping

        # Publish the safety coefficient
        safety_msg = Float32()
        safety_msg.data = remapped_safety_coefficient
        self.safety_publisher.publish(safety_msg)

        self.get_logger().info(f"Published safety coefficient: {remapped_safety_coefficient}")


class FuzzySafetySystem:
    def __init__(self):
        # Initialize domains
        self.targ_dist = Domain("Distance to target", 0, 1, res=0.01)
        self.ws_dist = Domain("Distance to workspace limits", 0, 1, res=0.01)
        self.obs_dist = Domain("Distance to obstacles", 0, 1, res=0.01)
        self.manipulability = Domain("Manipulability", 0, 1, res=0.01)
        self.safety = Domain("Safety factor", 0, 1, res=0.01)

        # Define membership functions
        self._define_membership_functions()

        # Define rules
        self.rules = self._define_rules()

    def _define_membership_functions(self):
        # Target distance membership functions
        self.targ_dist.close = np.vectorize(gauss(0, 17))
        self.targ_dist.medium = np.vectorize(gauss(0.5, 100))
        self.targ_dist.far = np.vectorize(gauss(1, 17))

        # Workspace distance membership functions
        self.ws_dist.close = np.vectorize(gauss(0, 17))
        self.ws_dist.medium = np.vectorize(gauss(0.5, 20))
        self.ws_dist.far = np.vectorize(gauss(1, 17))

        # Obstacle distance membership functions
        self.obs_dist.close = np.vectorize(gauss(0, 17))
        self.obs_dist.medium = np.vectorize(gauss(0.5, 100))
        self.obs_dist.far = np.vectorize(gauss(1, 17))

        # Manipulability membership functions
        self.manipulability.low = np.vectorize(gauss(0, 17))
        self.manipulability.medium = np.vectorize(gauss(0.5, 100))
        self.manipulability.high = np.vectorize(gauss(1, 17))

        # Safety factor membership functions
        self.safety.acs = np.vectorize(triangular(-1, 0.1, c=0))
        self.safety.shared = np.vectorize(triangular(0.1, 0.9, c=0.5))
        self.safety.human = np.vectorize(triangular(0.9, 2, c=1))

    def _define_rules(self):
        # Define labels for each input
        target_labels = [self.targ_dist.close, self.targ_dist.medium, self.targ_dist.far]
        obstacle_labels = [self.obs_dist.close, self.obs_dist.medium, self.obs_dist.far]
        workspace_labels = [self.ws_dist.close, self.ws_dist.medium, self.ws_dist.far]
        manip_labels = [self.manipulability.low, self.manipulability.medium, self.manipulability.high]

        rules = {}

        # Generate all combinations
        for t, o, w, m in product(target_labels, obstacle_labels, workspace_labels, manip_labels):
            # Determine output based on logic
            if (t == self.targ_dist.close or
                o == self.obs_dist.close or
                w == self.ws_dist.close or
                m == self.manipulability.low):
                rules[(t, o, w, m)] = self.safety.acs
            elif (t == self.targ_dist.medium or
                  o == self.obs_dist.medium or
                  w == self.ws_dist.medium or
                  m == self.manipulability.medium):
                rules[(t, o, w, m)] = self.safety.shared
            else:
                # all far/high
                rules[(t, o, w, m)] = self.safety.human

        # Define fuzzy rules considering manipulability
        return Rule(rules)

    def compute_safety(self, values):
        """
        Compute the safety coefficient based on input values.
        :param values: Dictionary with input values for targ_dist, ws_dist, and obs_dist.
        :return: Safety coefficient.
        """
        return self.rules(values)


def main(args=None):
    rclpy.init(args=args)
    node = FuzzySafetyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
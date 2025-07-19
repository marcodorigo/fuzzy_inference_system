import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32
from fuzzylogic.classes import Domain, Rule
from fuzzylogic.functions import gauss, triangular, trapezoid
from fuzzylogic.hedges import very


class FuzzySafetyNode(Node):
    def __init__(self):
        super().__init__('fuzzy_safety_node')

        # Initialize fuzzy safety system
        self.fuzzy_system = FuzzySafetySystem()

        # Subscriber to distances topic
        self.subscription = self.create_subscription(
            PoseStamped,
            '/distance_metrics',
            self.distance_callback,
            10
        )

        # Publisher for safety coefficient
        self.safety_publisher = self.create_publisher(Float32, '/safety_coefficient', 10)

        self.get_logger().info("✅ Fuzzy Safety Node started and ready.")

    def distance_callback(self, msg: PoseStamped):
        # Extract distances from the PoseStamped message
        dist_to_obstacles = msg.pose.position.x
        dist_to_workspace = msg.pose.position.y
        dist_to_target = msg.pose.position.z

        # Compute safety coefficient using fuzzy logic
        values = {
            self.fuzzy_system.targ_dist: dist_to_target,
            self.fuzzy_system.ws_dist: dist_to_workspace,
            self.fuzzy_system.obs_dist: dist_to_obstacles
        }
        safety_coefficient = self.fuzzy_system.compute_safety(values)

        # Publish the safety coefficient
        safety_msg = Float32()
        safety_msg.data = safety_coefficient
        self.safety_publisher.publish(safety_msg)

        self.get_logger().info(f"Published safety coefficient: {safety_coefficient}")


class FuzzySafetySystem:
    def __init__(self):
        # Initialize domains
        self.targ_dist = Domain("Distance to target", 0, 1, res=0.01)
        self.ws_dist = Domain("Distance to workspace limits", 0, 1, res=0.01)
        self.obs_dist = Domain("Distance to obstacles", 0, 1, res=0.01)
        self.safety = Domain("Safety factor", 0, 1, res=0.01)

        # Define membership functions
        self._define_membership_functions()

        # Define rules
        self.rules = self._define_rules()

    def _define_membership_functions(self):
        # Target distance membership functions
        self.targ_dist.close = gauss(0, 50)
        self.targ_dist.far = gauss(1, 10)
        self.targ_dist.medium = gauss(0.5, 10)

        # Workspace distance membership functions
        self.ws_dist.close = gauss(0, 30)
        self.ws_dist.medium = gauss(0.5, 10)
        self.ws_dist.far = gauss(1, 30)

        # Obstacle distance membership functions
        self.obs_dist.close = gauss(0, 17)
        self.obs_dist.medium = gauss(0.5, 20)
        self.obs_dist.far = gauss(1, 17)

        # Safety factor membership functions
        self.safety.acs = gauss(0, 17)
        self.safety.shared = trapezoid(0.1, 0.2, 0.85, 0.95)
        self.safety.human = triangular(0.8, 2, c=1)

    def _define_rules(self):
        # Define fuzzy rules
        return Rule({
            (self.targ_dist.far, self.obs_dist.far, self.ws_dist.far): very(self.safety.human),
            (self.targ_dist.far, self.obs_dist.far, self.ws_dist.medium): self.safety.shared,
            (self.targ_dist.far, self.obs_dist.far, self.ws_dist.close): self.safety.acs,
            (self.targ_dist.far, self.obs_dist.medium, self.ws_dist.far): self.safety.shared,
            (self.targ_dist.far, self.obs_dist.medium, self.ws_dist.medium): self.safety.shared,
            (self.targ_dist.far, self.obs_dist.medium, self.ws_dist.close): self.safety.acs,
            (self.targ_dist.far, self.obs_dist.close, self.ws_dist.far): self.safety.acs,
            (self.targ_dist.far, self.obs_dist.close, self.ws_dist.medium): self.safety.acs,
            (self.targ_dist.far, self.obs_dist.close, self.ws_dist.close): self.safety.acs,
            (self.targ_dist.medium, self.obs_dist.far, self.ws_dist.far): self.safety.shared,
            (self.targ_dist.medium, self.obs_dist.far, self.ws_dist.medium): self.safety.shared,
            (self.targ_dist.medium, self.obs_dist.far, self.ws_dist.close): self.safety.acs,
            (self.targ_dist.medium, self.obs_dist.medium, self.ws_dist.far): self.safety.shared,
            (self.targ_dist.medium, self.obs_dist.medium, self.ws_dist.medium): self.safety.shared,
            (self.targ_dist.medium, self.obs_dist.medium, self.ws_dist.close): self.safety.acs,
            (self.targ_dist.medium, self.obs_dist.close, self.ws_dist.far): self.safety.acs,
            (self.targ_dist.medium, self.obs_dist.close, self.ws_dist.medium): self.safety.acs,
            (self.targ_dist.medium, self.obs_dist.close, self.ws_dist.close): self.safety.acs,
            (self.targ_dist.close, self.obs_dist.far, self.ws_dist.medium): self.safety.acs,
            (self.targ_dist.close, self.obs_dist.far, self.ws_dist.close): self.safety.acs,
            (self.targ_dist.close, self.obs_dist.medium, self.ws_dist.far): self.safety.acs,
            (self.targ_dist.close, self.obs_dist.medium, self.ws_dist.medium): self.safety.acs,
            (self.targ_dist.close, self.obs_dist.medium, self.ws_dist.close): self.safety.acs,
            (self.targ_dist.close, self.obs_dist.close, self.ws_dist.far): self.safety.acs,
            (self.targ_dist.close, self.obs_dist.close, self.ws_dist.medium): self.safety.acs,
            (self.targ_dist.close, self.obs_dist.close, self.ws_dist.close): self.safety.acs,
        })

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
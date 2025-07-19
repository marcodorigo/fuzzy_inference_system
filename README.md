Fuzzy Inference System contains two python nodes:
- ee_distance_metrics_node.py computes the distance metrics of the end-effector (dist. to target, obstacles and workspace)
- fuzzy_safety_node computes the safety coefficient

The first node reads from /admittance_controller/pose_debug and publishes in /distance_metrics.
The second node reads from /distance_metrics and publishes in /safety_coefficient

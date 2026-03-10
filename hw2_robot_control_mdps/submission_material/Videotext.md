# Question 1 Theoretical Questions

1. If you increase the width of the Lemniscate (increasing a), what issue can happen with the robot performing IK?

The issue that can happen is that the robot does not reach the target in the given maximum iterations for a keypoint, since the target is too far.

2. What can happen if you change the dt parameter in IK?

If you lower the dt parameter, this generally improves tracking precision since the path is discretized more finely, and if you increase it, it leads to larger lineraization error and the robot may drift away from the intended path. 

3. We implemented a simple numerical IK solver. What are the advantages and disadvantages compared to an analytical IK solver?

A numerical solver can be applied universally to any robot, regardless of the number of joints, and supports redundancy through secondary goals, however, it requires multiple iterations to converge to a solution, can get stuck at local minima. 

4. What are the limits of our IK solver compared to state-of-the-art IK solvers?

Our solver only cares about position error, ignoring joint limits, runs for many iterations which can be slow on complex robots, and can get stuck at local minima since it simply follows the gradient. 

# Question 2 Theoretical Questions

1. If you keep increasing $K_P$, what issue arises when tracking the waypoints?

With increased Kp the controller becoms more aggressive by trying to close the gap between current position and target waypoint, leading to overshoot. 

2. How does $K_D$ mitigate the effect you saw above when increasing $K_P$?

Kd mitigates this effect by producing a control signal in the opposite direction of the motion, such that the arm is slowed down when it gets closer to the target.

3. In what scenarios is a non-zero $K_I$ needed for the controller to perform well?

This is needed to compensate for constant disturbances such, adapting to system model mismatch and overcoming high friction.

# Question 3 PrintLog

Loading model from /home/arnav/workspace/RobotLearning/ethz-course-2026/hw2_robot_control_mdps/logs/so100_tracking/so100_tracking_4/model_400.zip...
Final EE tracking error: 0.0065
Final EE tracking error: 0.0027
Final EE tracking error: 0.0122
Final EE tracking error: 0.0065
Final EE tracking error: 0.0042
Final EE tracking error: 0.0205
Final EE tracking error: 0.0035
Final EE tracking error: 0.0058
Final EE tracking error: 0.0073
Final EE tracking error: 0.0061
Average final EE tracking error: 0.0075



# Question 3 Improved Policy

Final EE tracking error: 0.0035
Final EE tracking error: 0.0039
Final EE tracking error: 0.0034
Final EE tracking error: 0.0026
Final EE tracking error: 0.0030
Final EE tracking error: 0.0038
Final EE tracking error: 0.0028
Final EE tracking error: 0.0002
Final EE tracking error: 0.0027
Final EE tracking error: 0.0029
Average final EE tracking error: 0.0029

#!/usr/bin/env python3

import os
import rospy
import numpy as np

# import hj_reachability as hj
# import jax.numpy as jnp
# from threading import Lock
from std_msgs.msg import Bool
from refinecbf_ros.msg import ValueFunctionMsg, HiLoArray, Array
from refinecbf_ros.config import Config

# from refinecbf_ros.config_hr import Config     # TODO: Make me, see refine_cbf

# from refinecbf_ros.config import QuadraticCBF
# from refine_cbfs import HJControlAffineDynamics

# from refine_cbfs import (
#     HJControlAffineDynamics,
#     TabularControlAffineCBF,
#     TabularTVControlAffineCBF,
#     utils,
# )


class HopfReachabilityNode:
    """
    Node for Solving the Hamilton-Jacobi Reachable Control with the Hopf Formula
    """

    def __init__(self):

        ## Init
        rospy.loginfo("Using hopf reachability node")

        self.config = Config()
        self.safety_states_idis = self.config.safety_states
        self.safety_controls_idis = self.config.safety_controls

        # self.Model = config_hr.Model          # TODO: Implement me
        # self.HR_solver = config_hr.HR_solver  # TODO: Implement me
        # self.HR_solver.Model = self.Model
        # self.Target = config_hr.Target        # TODO: Implement me, static for now, subscription to rigid body later

        self.state = None
        self.safe_control_policy = None
        self.hopf_initialized = False
        self.use_hopf = True  # TODO: should be rosservice call triggered

        # self.HR_solver = ?

        ## Subscriptions

        self.state_topic = rospy.get_param("~topics/state", "/state_array")
        self.state_sub = rospy.Subscriber(self.state_topic, Array, self.callback_state)

        nom_control_topic = rospy.get_param(
            "~topics/nominal_control", "/control/nominal"
        )  # only used to signal when to feed control, could just raw publish it every time we see a state
        self.nominal_control_sub = rospy.Subscriber(
            nom_control_topic, Array, self.callback_overwrite_control
        )

        ## Publications

        filtered_control_topic = rospy.get_param(
            "~topics/filtered_control", "/control/filtered"
        )  # (just using this established topic name, not actually filter control)
        self.pub_filtered_control = rospy.Publisher(
            filtered_control_topic, Array, queue_size=1
        )  # not sure what the quese_size does

        ## Start Hopf Solver

        self.solve_hopf()  # continuously solves the control if use_hopf is true (maybe off when no target observed?)

    def callback_state(self, state_est_msg):
        self.state = np.array(state_est_msg.value)[self.safety_states_idis]

    def callback_overwrite_control(self, control_msg):
        nom_control = np.array(control_msg.value)
        if self.use_hopf and self.hopf_initialized:
            safe_control = nom_control
            rospy.loginfo(nom_control)
            # safe_control = self.safe_control_policy(self.state.copy())
            safe_control_msg = Array()
            safe_control_msg.value = safe_control.tolist()  # "Ensures compatibility"
        else:
            safe_control_msg = control_msg
        self.pub_filtered_control.publish(safe_control_msg)

    def solve_hopf(self):
        while not rospy.is_shutdown():
            if self.use_hopf:
                # self.safe_control_policy = self.HR_solver(
                #     self.state.copy()
                # )  # should return state/time-dependent policy (function)
                # self.safe_control_policy = self.
                self.hopf_initialized = True
            rospy.sleep(
                0.05
            )  # "To make sure that subscribers can run" (do we want this?)


if __name__ == "__main__":
    rospy.init_node("hopf_reachability_node")
    rospy.loginfo("Using hopf reachability node")
    safety_filter = HopfReachabilityNode()
    rospy.spin()



# Have to add uref_sim.txt into the file requested (if doing on local)
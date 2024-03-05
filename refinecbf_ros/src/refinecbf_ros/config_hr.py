import rospy
import numpy as np
import jax
import jax.numpy as jnp

# from cbf_opt import ControlAffineDynamics, ControlAffineCBF
from julia import Main


# have to redefine crazyflie? different dynamics
class CrazyflieDynamics(ControlAffineDynamics):
    """
    Simplified dynamics, and we need to convert controls from phi to tan(phi)
    (Taken from config.py)
    """

    STATES = ["x", "y", "z", "v_x", "v_y", "v_z", "d_th"]
    CONTROLS = ["tan(phi)", "T"]
    DISTURBANCES = []

    def __init__(self, params, test=True, **kwargs):
        super().__init__(params, test, **kwargs)

    def open_loop_dynamics(self, state, time: float = 0.0):
        return jnp.array([state[2], state[3], 0.0, -self.params["g"]])

    def control_matrix(self, state, time: float = 0.0):
        return jnp.array([[0.0, 0.0], [0.0, 0.0], [self.params["g"], 0.0], [0.0, 1.0]])

    def state_jacobian(self, state, control, disturbance=None, time: float = 0.0):
        return jax.jacfwd(lambda x: self.__call__(x, control, disturbance, time))(state)


class Model:
    """
    Class for the Koopman matrix K.
    """

    def __init__(self, K):
        if not isinstance(K, np.ndarray) or K.ndim != 2:
            raise ValueError("K must be a two-dimensional numpy array")
        self.K = K


class Target:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z


class Sphere(Target):
    """
    Defines a static, spherical target with a center at (x, y, z)
    """

    def __init__(self, x=0, y=0, z=0, radius=1):
        super().__init__(x, y, z)
        self.radius = radius


class HopfSolver:  # needs to take in the Koopman matrix
    def __init__(self):
        Main.include("HopfReachability.jl")  # path to julia script ?

    def setup_solver(self, K, target, x, **kwargs):
        julia_kwargs = Main.Dict(kwargs)

        self.system = K
        self.target = target
        self.x = x
        self.kwargs = julia_kwargs

    def query_solver(self):
        result = Main.Hopf_minT(self.system, self.target, self.x, **self.kwargs)

        # Unpack and return the results - GPT generated
        u_optimal, d_optimal, T_star, phi, dphi_dz = result
        return {
            "u_optimal": u_optimal,
            "d_optimal": d_optimal,
            "T_star": T_star,
            "phi": phi,
            "dphi_dz": dphi_dz,
        }

"""
Madgwick attitude estimation and coordinate transformation.

Implements Algorithm 2 / Equations (1)-(11) of:
  Zhong, Wang, Sun. "MadgwickFall-Net: A Lightweight Dual-Frame Feature
  Fusion Network for Pre-Impact Fall Detection Using Wearable IMUs."
  Bioengineering 2026, 13, 568.

All symbols (beta, dt, epsilon, quaternion convention, rotation-matrix
formula) are taken verbatim from the paper's Section 3.3 and Algorithm 2.
"""
import numpy as np


def quat_multiply(q1, q2):
    """Hamilton product q1 (x) q2, q = [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def rotation_matrix(q):
    """Eq. (10): 3x3 rotation matrix from quaternion q = [q0,q1,q2,q3]."""
    q0, q1, q2, q3 = q
    R = np.array([
        [1 - 2 * (q2 ** 2 + q3 ** 2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
        [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 ** 2 + q3 ** 2), 2 * (q2 * q3 - q0 * q1)],
        [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 ** 2 + q2 ** 2)],
    ])
    return R


class MadgwickAHRS:
    """
    Madgwick gradient-descent attitude filter (paper Section 3.3).

    Parameters
    ----------
    beta : float
        Accelerometer correction gain. Paper's ablation (Sec. 4.3.3) finds
        the optimum at beta = 0.1 (tested over {0.01, 0.05, 0.1, 0.2, 0.5}).
    dt : float
        Sample period. Paper uses 1/100 s (KFall sampling rate = 100 Hz).
    eps : float
        Free-fall gating threshold on ||a|| (paper: epsilon = 0.1 g). Below
        this the accelerometer correction step is skipped (gyro-only
        prediction), since gravity is no longer a reliable reference.
    """

    def __init__(self, beta: float = 0.1, dt: float = 0.01, eps: float = 0.1):
        self.beta = beta
        self.dt = dt
        self.eps = eps
        self.q = np.array([1.0, 0.0, 0.0, 0.0])  # q0 initial: body//global

    def reset(self):
        self.q = np.array([1.0, 0.0, 0.0, 0.0])

    def update(self, acc_g: np.ndarray, gyro_dps: np.ndarray) -> np.ndarray:
        """
        One filter step (Algorithm 2, lines 3-19).

        acc_g   : (3,) raw accelerometer reading, units of g (as in KFall).
        gyro_dps: (3,) raw gyroscope reading, units of deg/s (as in KFall).

        Returns the updated unit quaternion q = [q0,q1,q2,q3].
        """
        q0, q1, q2, q3 = self.q
        gyro_rad = np.deg2rad(gyro_dps)

        # 1. Gyroscope-driven prediction, Eq. (2)
        q_dot_omega = 0.5 * quat_multiply(self.q, np.array([0.0, *gyro_rad]))

        a_norm = np.linalg.norm(acc_g)
        if a_norm > self.eps:
            # 2. Accelerometer-based gradient-descent correction, Eqs (3)-(8)
            a_hat = acc_g / a_norm
            f = np.array([
                2 * (q1 * q3 - q0 * q2) - a_hat[0],
                2 * (q0 * q1 + q2 * q3) - a_hat[1],
                2 * (0.5 - q1 ** 2 - q2 ** 2) - a_hat[2],
            ])
            J = np.array([
                [-2 * q2, 2 * q3, -2 * q0, 2 * q1],
                [2 * q1, 2 * q0, 2 * q3, 2 * q2],
                [0.0, -4 * q1, -4 * q2, 0.0],
            ])
            grad = J.T @ f
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-12:
                grad_hat = grad / grad_norm
            else:
                grad_hat = grad
            q_dot = q_dot_omega - self.beta * grad_hat
        else:
            # Free-fall gating: gyroscope-only prediction
            q_dot = q_dot_omega

        # 3. Integration + normalization, Eq. (9)
        q_new = self.q + q_dot * self.dt
        q_new = q_new / np.linalg.norm(q_new)
        self.q = q_new
        return self.q

    def process_sequence(self, acc_seq: np.ndarray, gyro_seq: np.ndarray,
                          warm_start_q=None):
        """
        Run the filter over an (N,3) acc / (N,3) gyro sequence and return
        the global-frame acceleration and angular velocity, Eq. (11), plus
        the quaternion trajectory.

        Returns
        -------
        acc_global  : (N,3)
        gyro_global : (N,3)
        quats       : (N,4)
        """
        if warm_start_q is not None:
            self.q = warm_start_q.copy()
        else:
            self.reset()

        N = acc_seq.shape[0]
        acc_global = np.zeros_like(acc_seq)
        gyro_global = np.zeros_like(gyro_seq)
        quats = np.zeros((N, 4))

        for t in range(N):
            q = self.update(acc_seq[t], gyro_seq[t])
            R = rotation_matrix(q)
            acc_global[t] = R @ acc_seq[t]
            gyro_global[t] = R @ gyro_seq[t]
            quats[t] = q

        return acc_global, gyro_global, quats

    @staticmethod
    def angular_deviation_from_vertical(acc_global_sample: np.ndarray) -> float:
        """
        Angle (deg) between the global-frame acceleration vector and the
        vertical [0,0,1] axis. Used for the warm-up convergence criterion
        in Section 4.4.3 (must stay < 5 deg for 0.5 s).
        """
        v = acc_global_sample / (np.linalg.norm(acc_global_sample) + 1e-12)
        cos_ang = np.clip(np.dot(v, np.array([0.0, 0.0, 1.0])), -1.0, 1.0)
        return np.degrees(np.arccos(cos_ang))

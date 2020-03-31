import numpy as np
from tools.task import *
from tools.jacobian import *
from scipy.stats import chi2


def get_observation_(state, landmark):
    dx = landmark[0] - state[0]
    dy = landmark[1] - state[1]

    distance = np.sqrt(dx ** 2 + dy ** 2)
    bearing = np.arctan2(dy, dx) - state[2]

    return np.array([distance, wrap_angle(bearing)])


class IC:
    def __init__(self, field_map, beta):
        self.lm_map = [list(a) for a in zip(list(field_map.landmarks_poses_x), list(field_map.landmarks_poses_y))]
        beta[1] = np.deg2rad(beta[1])
        self.Q = np.diag([*(beta ** 2)])

    def get_association(self, observation, state, lm_data):
        output_indexes = []
        old_indexes = lm_data

        for z in observation:
            obs_lm_index = None

            for lm in self.lm_map:
                innovation = z[:2] - get_observation_(state, lm)
                H = observation_jacobian(state, lm)[1]
                P = self.get_landmark_covariance_(state, get_observation_(state, lm))
                cov = H @ P @ H.T + self.Q
                mah_dis = innovation @ np.linalg.inv(cov) @ innovation.T

                if mah_dis < chi2.isf(0.05, 2):
                    obs_lm_index = self.lm_map.index(lm)

            output_indexes.append(obs_lm_index)
            if obs_lm_index not in old_indexes:
                for i in range(old_indexes.shape[0]):
                    if old_indexes[i] == -1.:
                        old_indexes[i] = obs_lm_index
                        break

        return output_indexes, old_indexes

    def get_landmark_covariance_(self, state, observation):
        W = inverse_observation_jacobian(state, observation)[1]
        return W @ self.Q @ W.T

    def construct_hypothesis(self, z, state):
        H = []
        for lm in self.lm_map:
            H.append(z - get_observation_(state, lm))
        return H

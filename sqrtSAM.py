import numpy as np
import scipy as scp

from tools.task import get_prediction
from tools.jacobian import state_jacobian, observation_jacobian, inverse_observation_jacobian
from tools.task import get_prediction, wrap_angle, get_motion_noise_covariance
from slam.slamBase import SlamBase
from tools.solvers import solve_least_squares


class SqrtSAM(SlamBase):
    def __init__(self, initial_state_mu, initial_state_Sigma, alphas, Q, solver="numpy"):
        super().__init__("sam", "known", "batch", Q)

        # History of states
        self.states_ = np.array([np.copy(initial_state_mu)])
        self.n_states_ = 1

        # Current set of landmarks
        self.landmarks_ = None
        self.landmark_covariances_ = list()
        self.n_landmarks_ = 0
        # Mapping from real landmark IDs to internal order for simplicity
        self.landmarks_index_map_ = dict()

        # History of actions
        self.actions_ = np.zeros((1, self.state_dim))

        # History of observations
        self.observations_ = None

        self.alphas_ = alphas
        self.Q_ = Q

        self.initial_mu_ = np.copy(initial_state_mu)
        self.initial_Sigma_ = np.copy(initial_state_Sigma)

        self.solver_method_ = solver

        self.chi_squared_errors_ = list()

    """
    Add new state and corresponding action
    """
    def add_state_and_action_(self, x, u):
        self.states_ = np.vstack((self.states_, np.array([x])))
        self.actions_ = np.vstack((self.actions_, np.array([u])))
        self.n_states_ += 1

    """
    Adds observation, augmenting them with
    corresponding state index.
    """
    def add_observations_(self, state_index, z):
        # Augment measurement with the state index
        new_obs = np.hstack((z, np.ones((z.shape[0], 1)) * state_index))
        if self.observations_ is None:
            self.observations_ = new_obs
        else:
            self.observations_ = np.vstack((self.observations_, new_obs))

    """
    Adds new landmark or updates existing.
    """
    def set_landmark_(self, m, landmark_id):
        if landmark_id not in self.landmarks_index_map_:
            if self.landmarks_ is None:
                self.landmarks_ = np.array([m])
            else:
                self.landmarks_ = np.vstack((self.landmarks_, np.array([m])))
            self.landmarks_index_map_[landmark_id] = self.n_landmarks_
            self.n_landmarks_ += 1
        else:
            self.landmarks_[self.landmarks_index_map_[landmark_id]] = m
        return self.landmarks_index_map_[landmark_id]

    """
    Calculates expected observation of landmark
    from specified state.
    """
    def get_observation_(self, state, landmark):
        dx = landmark[0] - state[0]
        dy = landmark[1] - state[1]

        distance = np.sqrt(dx ** 2 + dy ** 2)
        bearing = np.arctan2(dy, dx) - state[2]

        return np.array([distance, wrap_angle(bearing)])

    """
    Calculates landmark coordinates
    from specified observation.
    """
    def get_landmark_position_(self, state, range, bearing):
        angle = wrap_angle(state[2] + bearing)
        x_rel = range * np.cos(angle)
        y_rel = range * np.sin(angle)
        x = x_rel + state[0]
        y = y_rel + state[1]
        return np.array([x, y])

    """
    Constructs adjacency matrix.
    """
    def construct_adjacency_(self):
        n_cols = self.n_states_ * self.state_dim + self.n_landmarks_ * self.lm_dim
        A = np.zeros((self.state_dim, n_cols))

        # Fill priors
        A[0:self.state_dim, 0:self.state_dim] = -self.cholesky_sqrt_transpose_(self.get_state_covariance_(0))

        # Fill odometry factors

        for i in range(1, self.n_states_):
            state_factor_matrix = np.zeros((self.state_dim, n_cols))
            col_G = self.state_dim * (i-1)
            col_I = self.state_dim * i

            # Pre-multiplying matrix
            premult = self.cholesky_sqrt_transpose_(self.get_state_covariance_(i))

            # Fill Jacobians
            state_factor_matrix[0:self.state_dim, col_G:col_G+self.state_dim] = \
                premult @ state_jacobian(self.states_[i-1], self.actions_[i])[0]
            state_factor_matrix[0:self.state_dim, col_I:col_I+self.state_dim] = -premult

            if A is None:
                A = state_factor_matrix
            else:
                A = np.vstack((A, state_factor_matrix))

        # Fill observations factors

        offset = self.state_dim * self.n_states_
        # Observations pre-multiplying matrix
        obs_premult = self.cholesky_sqrt_transpose_(self.Q_)

        for i, observation in enumerate(self.observations_):
            obs_factor_matrix = np.zeros((self.obs_dim, n_cols))

            # Extract state and landmark index from augmented observation
            state = int(observation[self.obs_dim+1])
            landmark = int(observation[self.obs_dim])

            # Fill Jacobians
            H, J = observation_jacobian(self.states_[state], self.landmarks_[landmark])

            obs_factor_matrix[0:self.obs_dim, self.state_dim * state:self.state_dim * (state + 1)] = obs_premult @ H
            obs_factor_matrix[0:self.obs_dim, offset + self.lm_dim * landmark:offset + self.lm_dim * (landmark + 1)] = \
                obs_premult @ J
            A = np.vstack((A, obs_factor_matrix))

        return A


    """
    Constructs residuals vector.
    """
    def construct_residuals_(self):
        # Fill odometry residuals

        state_residuals = None
        for i, state in enumerate(self.states_):
            # Pre-multiplying matrix
            premult = self.cholesky_sqrt_transpose_(self.get_state_covariance_(i))
            if i == 0:
                # For the initial state assume prior mean as previous prediction
                residual = self.states_[i] - self.initial_mu_
                residual[2] = wrap_angle(residual[2])
                state_residuals = premult @ residual
            else:
                residual = self.states_[i] - get_prediction(self.states_[i-1], self.actions_[i])
                residual[2] = wrap_angle(residual[2])
                state_residuals = np.concatenate((state_residuals, premult @ residual))

        # Fill observations residuals

        # Observations pre-multiplying matrix
        obs_premult = self.cholesky_sqrt_transpose_(self.Q_)

        observation_residuals = None
        for observation in self.observations_:
            observation_value = observation[0:self.obs_dim]

            # Extract state and landmark index from augmented observation
            state_index = int(observation[self.obs_dim+1])
            landmark_index = int(observation[self.obs_dim])

            residual = observation_value - self.get_observation_(self.states_[state_index],
                                                                 self.landmarks_[landmark_index])
            residual[1] = wrap_angle(residual[1])
            observation_residuals = obs_premult @ residual if observation_residuals is None else np.concatenate((
                observation_residuals, obs_premult @ residual))

        return np.concatenate((state_residuals, observation_residuals))

    """
    Calculates Sigma^(-T/2) for input matrix Sigma.
    """
    def cholesky_sqrt_transpose_(self, matrix):
        return scp.linalg.cholesky(np.linalg.inv(matrix), lower=True).T

    """
    Calculates transition covariance for the state.
    """
    def get_state_covariance_(self, state_index):
        if state_index == 0:
            return self.initial_Sigma_
        V = state_jacobian(self.states_[state_index-1], self.actions_[state_index])[1]
        M = get_motion_noise_covariance(self.actions_[state_index], self.alphas_)
        return V @ M @ V.T

    """
    Calculates landmark position covariance.
    """
    def get_landmark_covariance_(self, state, observation):
        W = inverse_observation_jacobian(state, observation)[1]
        return W @ self.Q_ @ W.T

    """
    Returns last added state.
    """
    def get_current_state(self):
        return self.states_[-1]

    """
    Returns current landmarks coordinates and their covariances.
    """
    def get_current_map(self):
        return [(self.landmarks_[i], self.landmark_covariances_[i]) for i in range(self.n_landmarks_)]

    def predict(self, u, dt=None):
        x = get_prediction(self.get_current_state(), u)
        self.add_state_and_action_(x, np.copy(u))

    def update(self, z):
        for obs in z:
            if obs[2] not in self.landmarks_index_map_:
                # Initialize new landmark
                m = self.get_landmark_position_(self.get_current_state(), obs[0], obs[1])
                lm_id = self.set_landmark_(m, int(obs[2]))
                # Update landmark ID for simplicity
                obs[2] = lm_id
                self.landmark_covariances_.append(self.get_landmark_covariance_(self.get_current_state(), obs))
            else:
                obs[2] = self.landmarks_index_map_[obs[2]]
                self.landmark_covariances_[int(obs[2])] = self.get_landmark_covariance_(self.get_current_state(), obs)
        self.add_observations_(self.n_states_-1, np.copy(z))

        theta = np.concatenate((np.reshape(self.states_, (self.n_states_*self.state_dim,)),
                                np.reshape(self.landmarks_, (self.n_landmarks_*self.lm_dim,))))
        while True:
            A = self.construct_adjacency_()
            b = self.construct_residuals_()
            delta = solve_least_squares(A, b, self.solver_method_)
            theta = theta + delta

            self.states_ = np.reshape(theta[0:self.n_states_ * self.state_dim],
                                      (self.n_states_, self.state_dim))
            self.landmarks_ = np.reshape(theta[self.n_states_ * self.state_dim:],
                                         (self.n_landmarks_, self.lm_dim))

            for state in self.states_:
                state[2] = wrap_angle(state[2])

            if np.linalg.norm(delta) < 1e-4:
                # Convergence achieved - add new Chi-squared error
                self.chi_squared_errors_.append(b.T @ b)
                break

    @property
    def landmarks_index_map(self):
        return self.landmarks_index_map_



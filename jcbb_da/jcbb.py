import numpy as np
from sqrtSAM import SqrtSAM


class JCBB(SqrtSAM):
    def __init__(self, field_map, beta, initial_state_mu, initial_state_Sigma, alphas, solver="numpy"):
        self.field_map = field_map
        self.Q = np.diag([*(beta ** 2)])
        # self.accord = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 0}
        self.data_ass = []
        self.associated_data = []
        super().__init__(initial_state_mu, initial_state_Sigma, alphas, self.Q, solver)

    def get_jcbb(self, observation, state):
        self.associated_data = []
        N = 4
        M = observation[:, 0].size
        H = np.eye(2*M)

        S = np.zeros((2*M, 2*M))
        for i in range(M):
            S[2*i:2*(i+1), 2*i:2*(i+1)] = self.Q

        # P = np.zeros((3+4*N, 3+4*N))
        P = 1e-4*np.eye(3+4*N)
        P[0:3, 0:3] = self.get_state_covariance_(self.n_states_-1)
        for key, value in self.landmarks_index_map_.items():
            P[(3+2*key):(3+2*(key+1)), (3+2*key):(3+2*(key+1))] = self.landmark_covariances_[value]

        d_sq = np.zeros((2*N)**M)
        Prop = np.zeros(((2*N)**M, 4))
        for i in range(2*N):
            for j in range(2*N):
                Prop[i*2*N+j, :] = [0, i, 1, j]

        y = state

        for i in range(2*N):
            y = np.hstack((y, [self.field_map.landmarks_poses_x[i], self.field_map.landmarks_poses_y[i]]))
        for i in range(Prop[:,0].size):
            f = np.zeros((2*M, 1))
            G = np.zeros((2*M, (3+4*N)))
            for j in range(M):
                lm_id = int(Prop[i,2*j+1])
                landm = [self.field_map.landmarks_poses_x[lm_id], self.field_map.landmarks_poses_y[lm_id]]
                f[2 * j:2 * (j + 1)] = (observation[j, :2] - self.get_observation_(state, landm)).reshape(2, 1)
                dx = self.field_map.landmarks_poses_x[lm_id] - y[0]
                dy = self.field_map.landmarks_poses_y[lm_id] - y[1]
                D = (dx**2 + dy**2)**0.5
                L = dy/dx
                G[2*j:2*(j+1), 0:3] = np.array([[-dx/D,                      -dy/D, 0],
                                                [-L/((1+L**2)*dx), 1/((1+L**2)*dx), 1]])
                G[2*j:2*(j+1), 3+2*lm_id:3+2*(lm_id+1)] = np.array([[dx/D,                        dy/D],
                                                                    [L/((1+L**2)*dx), -1/((1+L**2)*dx)]])

            C = H @ S @ H.T + G @ P @ G.T
            d_sq[i] = f.T @ np.linalg.inv(C) @ f

        ind = np.unravel_index(np.argmin(d_sq), d_sq.shape)
        k = 0
        '''
        for i in range(M):
            if observation[i, 2] == self.accord[int(Prop[ind, 2*i+1])]:
                k += 1
        '''
        for i in range(M):
            self.associated_data.append(int(Prop[ind, 2 * i + 1]))

        return(self.associated_data)

        # self.data_ass = np.append(self.data_ass, k)

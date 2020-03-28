# RL data association
Data Association in SLAM using Reinforcement Learning (["Perception in Robotics"](https://github.com/Kichkun/perception_course) Skoltech course project)

# Custom gym environment
[Gym](https://gym.openai.com/) is a toolkit for developing and comparing reinforcement learning algorithms. For our project we created custom environment.
## How to use it
- Create object of class DataAssociationEnv with necessary parameters
- You can choose two ways to generate observations: 

1 From .npy file or generated data and solving SAM we can obtain robots, landmarks coordinates and observations

2 Random observations generator. You can pick max number of observations and number of landmarks. It will randomly generate observation data (with different number of obserwations at each step) and robot position on the field. 

## Parameters
In file gym_environment.py you can find usage of the environment. List of the parameters which you can pass:
- -i: File with generated data to simulate the filter
- -n: The number of time steps to generate data for the simulation. This option overrides the data file argument
- -a: Diagonal of Standard deviations of the Transition noise in action space (M_t)
- -b: Diagonal of Standard deviations of the Observation noise (Q)
- --dt: Time step (in seconds)
- -s: Show and animation of the simulation, in real-time
- --plot-pause-len: Time (in seconds) to pause the plot animation for between frames
- -m: The full path to movie file to write the simulation animation to
- --movie-fps: The FPS rate of the movie to write
- --solver: Least squares solving method: build-in numpy or Cholesky factorization with back-substitution
- -r: Generate random robots state each step for gym environment

## How it works
For example lets assume that we picked random data generation option (8 landmarks). After first iteration we will get observation parameter:
'observations': 
> array([[ 4.68595243e+02, -6.67027008e-01],
>       [ 3.30830914e+02, -1.04785606e+00],
>       [ 4.04755448e+02,  4.65353626e-03],
>       [ 0.00000000e+00,  0.00000000e+00],
>       [ 0.00000000e+00,  0.00000000e+00],
>       [ 0.00000000e+00,  0.00000000e+00],
>       [ 0.00000000e+00,  0.00000000e+00]])

'robot_coordinates': 
> array([480.49167046, 241.87318988,  -2.40705962])

'LM_data': 
> array([[ 0.,  0., -1.],
>       [ 0.,  0., -1.],
>       [ 0.,  0., -1.],
>       [ 0.,  0., -1.],
>       [ 0.,  0., -1.],
>       [ 0.,  0., -1.],
>       [ 0.,  0., -1.],
>       [ 0.,  0., -1.]])

For LM_data -1 in ID column means that we have no info jet.
On the second step we assume correct DA on the previous step. Thats why our landmark info will look like (this info is available for DA algorithms):
'LM_data': 
> array([[ 21.        , 292.        ,   7.        ],
>        [168.33333333, 292.        ,   6.        ],
>        [168.33333333,   0.        ,   1.        ],
>        [  0.        ,   0.        ,  -1.        ],
>        [  0.        ,   0.        ,  -1.        ],
>        [  0.        ,   0.        ,  -1.        ],
>        [  0.        ,   0.        ,  -1.        ],
>        [  0.        ,   0.        ,  -1.        ]])

Array contains info about global landmark coordinates (perfect for random data generation case) and their IDs. Let's check the first one. For robot coordinates array([480.49167046, 241.87318988,  -2.40705962]) and observation array([ 4.68595243e+02, -6.67027008e-01]) noisy landmark coordinates: array([12.96373119  210.26420754])

# To randomly generate data 
```bash
cd "<project_dir>"
python gym_environment.py -s -r --max-obs-per-time-step 7 -n 10
```

# To record video and used the provided data 
```bash
cd "<project_dir>"
python gym_environment.py -s -i slam-evaluation-input.npy -m da.mp4
```



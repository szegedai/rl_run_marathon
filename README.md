## Learning to Run a Marathon: Avoid Overfitting to Speed(accepted for ICAART 2025)

This repository implements robust deep reinforcement learning techniques, focusing on enhancing stability and utility during policy learning in dynamic environments. Specifically, we utilize the TRPO (Trust Region Policy Optimization) and SPPO (Smooth Policy Optimization) algorithms to train models that can move simulated dummies efficiently while maintaining stability.

## S-PPO (Smoothed - Proximal Policy Optimization)

The repository (https://github.com/Trustworthy-ML-Lab/Robust_HighUtil_Smoothed_DRL) provides the implementation of robust and smoothed deep reinforcement learning (DRL) algorithms designed to improve decision-making in high-stakes scenarios. The methods prioritize robustness to adversarial conditions and ensure smoother policies for safer deployment.

### Setup
```
python=3.7
cd sppo
git clone https://github.com/KaidiXu/auto_LiRPA
cd auto_LiRPA
git checkout 389dc72fcff606944dca0504cc77f52fef024c4e
python setup.py install
cd ..
pip install -r requirements.txt
pip install gym==0.26.2
pip install oslo_config==8.8.1
pip install oslo_utils==4.13.0
pip install matplotlib==3.5.3
```
Then, follow the instructions [here](https://github.com/openai/mujoco-py#install-mujoco) to install mujoco
```
pip install mujoco_py==2.1.2.14
pip install "cython<3"
pip install gym==0.23.0
pip install --force-reinstall torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/
pip install openpyxl
```

### Training
```
cd sppo/src

python run.py --config-path config_hopper_sppo{_sgld}.json --seed=0 --run-type {baseline/static/dynamic}
```
The models are saved in the sppo/src/sppo{_sgld}_hopper/agents folder.

### Evaluate
```
cd sppo/src

python test.py --config-path config_hopper_sppo{_sgld}.json --exp-ids "{model_id1 model_id2 ...}" --deterministic --excel-name {Output excel name} --num-episodes=10
```
The identifier of the models should be provided, even more than one at list level.


## TRPO (Trust Region Policy Optimization with Generalized Advantage Estimation)

The repository, by pat-coady (https://github.com/pat-coady/trpo/tree/aigym_evaluation) contains an implementation of Trust Region Policy Optimization (TRPO), a reinforcement learning algorithm introduced by John Schulman et al. TRPO is a policy optimization method designed to improve stability and efficiency when training agents in environments with continuous action spaces.

### Setup (need extend)
* Python 3.7
* The Usual Suspects: NumPy, matplotlib, scipy
* TensorFlow
* gym - [installation instructions](https://gym.openai.com/docs)
* [MuJoCo](http://www.mujoco.org/)

### Training
```
cd trpo/src

python train_baseline.py --num_episodes 30000 --model_save_frequency 1500 --seed 0 --environment "Hopper-v4"
python train_baseline.py --num_episodes 25200 --model_save_frequency 1260 --seed 0 --environment "Walker2d-v4"
python train_baseline.py --num_episodes 200000 --model_save_frequency 10000 --seed 0 --environment "Humanoid-v4"

python train_static.py --num_episodes 22400 --model_save_frequency 1120 --seed 0 -environment "Hopper-v4" --update_interval_episodes 15000
python train_static.py --num_episodes 18400 --model_save_frequency 920 --seed 0 -environment "Walker2d-v4" --update_interval_episodes 12400
python train_static.py --num_episodes 200000 --model_save_frequency 920 --seed 0 -environment "Humanoid-v4" --update_interval_episodes 100000

python train_dynamic.py --environment "Hopper-v4" --seed 0 --total_training_steps 23000000 --save_steps 1150000
python train_dynamic.py --environment "Walker2d-v4" --seed 0 --total_training_steps 21400000 --save_steps 1070000
python train_dynamic.py --environment "Humanoid-v4" --seed 0 --total_training_steps 150000000 --save_steps 7500000
```
The models are saved in the trpo/model folder.


### Evaluate
```
cd trpo/src/evaluate
python evaluate.py "model1 model2 ..." -env {Hopper-v4/Walker2d-v4/Humanoid-v4} -en {Output Excel Name}
```
The identifier of the models (the name of the folder where the model was saved like 001, 002, etc..) should be provided, even more than one at list level.


## References

1. [Trust Region Policy Optimization](https://arxiv.org/pdf/1502.05477.pdf)
5. [By Patrick Coady](https://github.com/pat-coady/trpo/tree/aigym_evaluation)
6. [SPPO] (https://github.com/Trustworthy-ML-Lab/Robust_HighUtil_Smoothed_DRL)
6. [Breaking the Barrier: Enhanced Utility and Robustness in Smoothed DRL Agents](https://arxiv.org/pdf/2406.18062)

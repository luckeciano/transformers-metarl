# _Transformers are Meta-Reinforcement Learners_

This is the source code for the paper "Transformers are Meta-Reinforcement Learners", presented as part of the ICML 2022. This repository is a fork from the [Garage repository](https://github.com/rlworkgroup/garage).


## Setup
To setup the conda environment, execute the following command on the root directory:

```sh
conda env create -f environment.yml
```

## Reproducibility
To run the experiments from the paper, please execute the following experiments. The experiments scripts are located in the `examples/torch` directory.

#### HalfCheetahVel
```sh
./transformer_ppo_halfcheetah.py --wm_embedding_hidden_size=32 --n_heads=16 --d_model=128 --layers=4 --wm_size=5 --em_size=5 --dim_ff=512 --meta_batch_size=20 --episode_per_task=2 --discount=0.9 --gae_lambda=0.8 --lr_clip_range=0.1 --policy_lr=3e-05 --vf_lr=0.00025 --minibatch_size=256 --max_opt_epochs=10 --policy_ent_coeff=0.0 --entropy_method=regularized --architecture=Encoder  --policy_head_input=latest_memory --attn_type=1 --pre_lnorm --init_params --use_softplus_entropy --gating=residual --learn_std  --init_std=0.2  --tfixup --remove_ln --n_epochs=2500 --policy_lr_schedule=decay --decay_epoch_init=100 --decay_epoch_end=750 --min_lr_factor=0.1 --env_name=HalfCheetahVelEnv
```

#### HalfCheetahDir

```sh
./transformer_ppo_halfcheetah.py --wm_embedding_hidden_size=32 --n_heads=16 --d_model=128 --layers=4 --wm_size=5 --em_size=5 --dim_ff=512 --meta_batch_size=20 --episode_per_task=2 --discount=0.9 --gae_lambda=0.8 --lr_clip_range=0.2 --policy_lr=3e-05 --vf_lr=3e-05 --minibatch_size=256 --max_opt_epochs=10 --policy_ent_coeff=0.0 --entropy_method=regularized --architecture=Encoder --policy_head_input=latest_memory --init_std=0.5 --remove_ln --tfixup --learn_std --pre_lnorm --init_params --use_softplus_entropy --policy_lr_schedule=decay --vf_lr_schedule=decay --share_network --decay_epoch_init=100 --decay_epoch_end=3700 --min_lr_factor=0.0001 --policy_head_type=Default --env_name=HalfCheetahDirEnv
```



#### AntDirEnv

```sh
./transformer_ppo_halfcheetah.py --wm_embedding_hidden_size=32 --n_heads=16 --d_model=128 --layers=4 --wm_size=5 --em_size=5 --dim_ff=512 --meta_batch_size=20 --episode_per_task=2 --discount=0.92 --gae_lambda=0.8 --lr_clip_range=0.2 --policy_lr=3e-05 --vf_lr=3e-05 --minibatch_size=256 --max_opt_epochs=10 --policy_ent_coeff=0.0 --entropy_method=regularized --architecture=Encoder --policy_head_input=latest_memory --init_std=0.5 --remove_ln --tfixup --learn_std --pre_lnorm --init_params --use_softplus_entropy --policy_lr_schedule=decay --vf_lr_schedule=decay --share_network --decay_epoch_init=500 --decay_epoch_end=3500 --min_lr_factor=0.0001 --policy_head_type=Default --output_weights_scale=0.01 --env_name=AntDirEnv
```

#### MetaWorld

For MetaWorld environments, please use with the correct env_name and task_name:

```sh
./transformer_ppo_ml1.py --wm_embedding_hidden_size=32 --n_heads=4 --d_model=32 --layers=4 --wm_size=5 --em_size=5 --dim_ff=128 --meta_batch_size=25 --episode_per_task=10 --discount=0.9 --gae_lambda=0.95 --lr_clip_range=0.2 --policy_lr=5e-05 --vf_lr=5e-05 --minibatch_size=32 --max_opt_epochs=10 --policy_ent_coeff=0.0 --entropy_method=regularized --architecture=Encoder --policy_head_input=latest_memory --init_std=1.0 --remove_ln --tfixup --learn_std --pre_lnorm --init_params --use_softplus_entropy --policy_lr_schedule=no_schedule --vf_lr_schedule=no_schedule --decay_epoch_init=500 --decay_epoch_end=3500 --min_lr_factor=0.0001 --policy_head_type=Default --output_weights_scale=0.01 --env_name=<env_name> --task_name=<task_name> --max_episode_length=500
```


## Reproducibility -- Baselines

For MuJoCo tasks, we used [PEARL repository](https://github.com/katerakelly/oyster) (for PEARL) and [ProMP repository](https://github.com/jonasrothfuss/ProMP) for RL2-PPO and MAML-TRPO.

For MetaWorld tasks, we [reproduced the experiments in the MetaWorld paper using the Garage Repository](https://github.com/rlworkgroup/garage/pull/2287).

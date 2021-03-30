#!/usr/bin/env python3

import subprocess
import time
import psutil
import datetime
import random
import click
import shlex

#wm_embedding_hidden_size = [8, 32, 64, 128, 256] #, 512
heads_dmodel = [(1, 1), (1, 4), (1, 16),(1, 32), (1, 64),
                (2, 2), (2, 4), (2, 8), (2, 64), (2, 32), (2, 16),
                (4, 4), (4, 8), (4, 16), (4, 32), (4, 64), (4, 128), (4, 256), (4, 512),
                (8, 8), (8, 16), (8, 32), (8, 64), (8, 128), (8, 256), (8, 512),
                (16, 16), (16, 32), (16, 64), (16, 128), (16, 256), (16, 512),
                (32, 32), (32, 64), (32, 128), (32, 256), (32, 512),
                (64, 64), (64, 128), (64, 256), (64, 512)]

heads_dmodel = [(16, 128)]

encoder_decoder_layers = [4]

dropout = [0.0, 0.25, 0.5] #0.8]
dropout = [0.0]

wm_size = [5, 25]
em_size = [1, 2, 3, 4]

#meta_batch_size_list = [1, 5, 10, 20]
#episodes_per_task_list = [1, 2, 4]
discount_list = [0.8, 0.9]
gae_lambda_list = [0.95]
lr_clip_range_list =  [0.1, 0.2]
lr_list = [1e-3, 3e-4, 7e-5, 3e-5, 3e-6]
vf_lr_list = [1e-3, 3e-4, 7e-5, 3e-5, 3e-6]
minibatch_size_list = [32, 64, 128, 256, 1024]
max_opt_epochs_list = [1, 5, 10, 20, 50]
center_adv_list = [True, False]
positive_adv_list = [True, False]
entropy_hypers_list = [('no_entropy', 0.0), ('regularized', 0.01), ('regularized', 0.1), ('max', 0.01), ('max', 0.1)]
architecture_list = ['Encoder']
policy_head_input_list = ['full_memory', 'latest_memory', 'mixed_memory']
policy_head_input_list = ['latest_memory']
use_softplus_entropy_list = [True, False]
stop_entropy_gradient_list = [True, False]
policy_head_type_list = ['Default', 'TwoHeaded', 'IndependentStd']
remove_ln = [True, False]
init_params_list = [True, False]
pre_lnorm_list = [True, False]
share_network_list =[True, False]
tfixup_list = [True]
init_std_list = [0.1, 0.2, 0.5, 1.0]
decay_epoch_list = [100, 250, 500, 750]


def trmrl_cmd(env_name, nheads_dmodel, layers, dropout_rate, wm_length, em_length,\
        discount, gae_lambda, lr_clip_range, lr, vf_lr, minibatch_size, max_opt_epochs, center_adv, positive_adv, entropy_hypers, \
        use_softplus_entropy, stop_entropy_gradient, architecture, policy_head_type, init_std, policy_head_input, decay_epoch, \
        init_params, pre_lnorm, share_network, tfixup, gpu_id):
    cmd = "./transformer_ppo_halfcheetah.py --env_name=" + str(env_name) + \
    " --n_heads=" + str(nheads_dmodel[0]) + " --d_model=" + str(nheads_dmodel[1]) + " --layers=" + str(layers) + \
    " --dropout=" + str(dropout_rate) + " --wm_size=" + str(wm_length) + " --em_size=" + str(em_length) + " --dim_ff=" + str(4 * nheads_dmodel[1]) + \
    " --discount=" + str(discount) + \
    " --gae_lambda=" + str(gae_lambda) + " --lr_clip_range=" + str(lr_clip_range) + " --policy_lr=" + str(lr) + " --vf_lr=" + str(vf_lr) + \
    " --minibatch_size=" + str(minibatch_size) + " --max_opt_epochs=" + str(max_opt_epochs)  + \
    " --policy_ent_coeff=" + str(entropy_hypers[1]) + \
    " --entropy_method=" + str(entropy_hypers[0]) + " --gpu_id=" + str(gpu_id) + \
    " --architecture=" + str(architecture) + " --policy_head_type=" + str(policy_head_type) + " --init_std=" + str(init_std) + " --policy_head_input=" + str(policy_head_input) + \
    " --decay_epoch_init=" + str(decay_epoch)

    if center_adv:
        cmd += " --center_adv"
    if positive_adv:
        cmd += " --positive_adv"
    if use_softplus_entropy:
        cmd += " --use_softplus_entropy"
    if stop_entropy_gradient:
        cmd += " --stop_entropy_gradient"
    if remove_ln:
        cmd += " --remove_ln"
    if pre_lnorm:
        cmd += " --pre_lnorm"
    if init_params:
        cmd += " --init_params"
    if share_network:
        cmd += " --share_network"
    if tfixup:
        cmd += " --tfixup"

    print(cmd)

    return cmd

@click.command()
@click.option('--gpu_id', default=0)
@click.option('--env_name', default='HalfCheetahDirEnv')
def run_search(gpu_id, env_name):
    print("GPU ID: " + str(gpu_id))
    process_list = []
    MAX_RUNNING_PROCESSES = 8
    while True:
        print("New iteration. Current number of processes in list: " + str(len(process_list)))
        rm_p = []
        for p in process_list:
            if p.poll() is not None:
                print("Process " + str(p.pid) + " finished.")
                rm_p.append(p)
            else:
                try:
                    x = psutil.Process(p.pid)
                    start_time = datetime.datetime.fromtimestamp(x.create_time())
                    allowed_time = start_time + datetime.timedelta(seconds=1200 * 3600)
                    now = datetime.datetime.now()
                    if now > allowed_time:
                        print("Killing process " + str(p.pid) + " because it reached max execution time.")
                        x.kill()
                        rm_p.append(p)
                except:
                    rm_p.append(p)
        for p in rm_p:
            process_list.remove(p)
        
        for _ in range(MAX_RUNNING_PROCESSES - len(process_list)):
            print("Starting new process...")
            cmd = trmrl_cmd(env_name, random.choice(heads_dmodel), random.choice(encoder_decoder_layers), \
                random.choice(dropout), random.choice(wm_size), random.choice(em_size), \
                random.choice(discount_list), random.choice(gae_lambda_list), random.choice(lr_clip_range_list), random.choice(lr_list), random.choice(vf_lr_list), \
                random.choice(minibatch_size_list), random.choice(max_opt_epochs_list), random.choice(center_adv_list), random.choice(positive_adv_list),\
                random.choice(entropy_hypers_list), random.choice(use_softplus_entropy_list), random.choice(stop_entropy_gradient_list), \
                random.choice(architecture_list), random.choice(policy_head_type_list), random.choice(init_std_list), random.choice(policy_head_input_list), random.choice(decay_epoch_list), \
                random.choice(init_params_list), random.choice(pre_lnorm_list), random.choice(share_network_list), random.choice(tfixup_list), gpu_id)
            p = subprocess.Popen(shlex.split(cmd))
            time.sleep(10)
            process_list.append(p)

        time.sleep(30)

run_search()
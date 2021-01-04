#!/usr/bin/env python3

import subprocess
import time
import psutil
import datetime
import random
import click
import shlex

wm_embedding_hidden_size = [64, 128, 256] #, 512]
heads_dmodel = [(2, 1024), (2, 512), (2, 256), (2, 64), (2, 32), (2, 16),
    (4, 128), (4, 64), (4, 32), (4, 16),
    (8, 128), (8, 64), (8, 32), (8, 16), (8, 8),
    (16, 64), (16, 32),
    (32, 64), (32, 32)]

encoder_decoder_layers = [1, 2, 3, 4]
dropout = [0.0, 0.1, 0.25, 0.5, 0.8]
wm_size = [1, 5, 25, 50] #, 75, 100]
em_size = [1, 2, 3, 4]

meta_batch_size_list = [1, 5, 10, 20]
episodes_per_task_list = [1, 2, 4, 10]
discount_list = [0.95, 0.99, 0.999, 0.995]
gae_lambda_list = [0.8, 0.9, 0.95, 0.99]
lr_clip_range_list =  [0.1, 0.2]
lr_list = [1e-3, 2.5e-4, 7e-5]
vf_lr_list = [1e-2, 1e-3, 2.5e-4, 7e-5	]
minibatch_size_list = [16, 32, 64, 128, 256, 1024]
max_opt_epochs_list = [1, 5, 10, 20, 50]
center_adv_list = [True, False]
positive_adv_list = [True, False]
policy_ent_coeff_list = [0.0, 0.01, 0.1]
use_softplus_entropy_list = [True, False]
stop_entropy_gradient_list = [True, False]
entropy_method_list = ['max', 'regularized', 'no_entropy']

def trmrl_cmd(wm_emb_hidden_size, nheads_dmodel, layers, dropout_rate, wm_length, em_length, meta_batch_size, episodes_per_task, \
        discount, gae_lambda, lr_clip_range, lr, vf_lr, minibatch_size, max_opt_epochs, center_adv, positive_adv, policy_ent_coeff, \
        use_softplus_entropy, stop_entropy_gradient, entropy_method):
    cmd = "./transformer_ppo_halfcheetah.py --wm_embedding_hidden_size=" + str(wm_emb_hidden_size) + \
    " --n_heads=" + str(nheads_dmodel[0]) + " --d_model=" + str(nheads_dmodel[1]) + " --layers=" + str(layers) + \
    " --dropout=" + str(dropout_rate) + " --wm_size=" + str(wm_length) + " --em_size=" + str(em_length) + " --dim_ff=" + str(4 * nheads_dmodel[1]) + \
    " --meta_batch_size=" + str(meta_batch_size) + " --episode_per_task=" + str(episodes_per_task) + " --discount=" + str(discount) + \
    " --gae_lambda=" + str(gae_lambda) + " --lr_clip_range=" + str(lr_clip_range) + " --policy_lr=" + str(lr) + " --vf_lr=" + str(vf_lr) + \
    " --minibatch_size=" + str(minibatch_size) + " --max_opt_epochs=" + str(max_opt_epochs) + " --center_adv=" + str(center_adv) + \
    " --positive_adv=" + str(positive_adv) + " --policy_ent_coeff=" + str(policy_ent_coeff) + " --use_softplus_entropy=" + str(use_softplus_entropy) + \
    " --stop_entropy_gradient=" + str(stop_entropy_gradient) + " --entropy_method=" + str(entropy_method)

    print(cmd)

    return cmd

@click.command()
@click.option('--gpu_id', default=0)
def run_search(gpu_id):
    print("GPU ID: " + str(gpu_id))
    process_list = []
    MAX_RUNNING_PROCESSES = 2
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
                    allowed_time = start_time + datetime.timedelta(seconds=9000)
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
            cmd = trmrl_cmd(random.choice(wm_embedding_hidden_size), random.choice(heads_dmodel), random.choice(encoder_decoder_layers), \
                random.choice(dropout), random.choice(wm_size), random.choice(em_size), random.choice(meta_batch_size_list), random.choice(episodes_per_task_list), \
                random.choice(discount_list), random.choice(gae_lambda_list), random.choice(lr_clip_range_list), random.choice(lr_list), random.choice(vf_lr_list), \
                random.choice(minibatch_size_list), random.choice(max_opt_epochs_list), random.choice(center_adv_list), random.choice(positive_adv_list),\
                random.choice(policy_ent_coeff_list), random.choice(use_softplus_entropy_list), random.choice(stop_entropy_gradient_list), random.choice(entropy_method_list))
            p = subprocess.Popen(shlex.split(cmd))
            time.sleep(10)
            process_list.append(p)

        time.sleep(30)

run_search()
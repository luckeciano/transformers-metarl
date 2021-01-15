#!/usr/bin/env python3

import subprocess
import time
import psutil
import datetime
import random
import click
import shlex

wm_embedding_hidden_size = [8, 32, 64, 128, 256] #, 512]
heads_dmodel = [(1, 1), (1, 4), (1, 16),(1, 32), (1, 64),
    (2, 2), (2, 4), (2, 8), (2, 64), (2, 32), (2, 16),
    (4, 64), (4, 32), (4, 16), (4, 8), (4, 4),
    (8, 64), (8, 32), (8, 16), (8, 8),
    (16, 64), (16, 32), (16, 16),
    (32, 64), (32, 32)]

encoder_decoder_layers = [1, 2, 3, 4]
dropout = [0.0, 0.1, 0.25, 0.5] #0.8]
wm_size = [5, 25, 50, 75, 100]
em_size = [1, 2, 3, 4]


def trmrl_cmd(wm_emb_hidden_size, nheads_dmodel, layers, dropout_rate, wm_length, em_length, gpu_id):
    cmd = "./transformer_ppo_halfcheetah.py --wm_embedding_hidden_size=" + str(wm_emb_hidden_size) + \
    " --n_heads=" + str(nheads_dmodel[0]) + " --d_model=" + str(nheads_dmodel[1]) + " --layers=" + str(layers) + \
    " --dropout=" + str(dropout_rate) + " --wm_size=" + str(wm_length) + " --em_size=" + str(em_length) + " --dim_ff=" + str(4 * nheads_dmodel[1]) + \
    " --meta_batch_size=" + str(10) + " --episode_per_task=" + str(4) + " --discount=" + str(0.99) + \
    " --gae_lambda=" + str(0.95) + " --lr_clip_range=" + str(0.2) + " --policy_lr=" + str(0.00025) + " --vf_lr=" + str(0.00025) + \
    " --minibatch_size=" + str(32) + " --max_opt_epochs=" + str(10) + " --center_adv=" + str(False) + \
    " --positive_adv=" + str(False) + " --policy_ent_coeff=" + str(0.02) + " --use_softplus_entropy=" + str(False) + \
    " --stop_entropy_gradient=" + str(True) + " --entropy_method=" + "max" + " --gpu_id=" + str(gpu_id)

    print(cmd)

    return cmd

@click.command()
@click.option('--gpu_id', default=0)
def run_search(gpu_id):
    print("GPU ID: " + str(gpu_id))
    process_list = []
    MAX_RUNNING_PROCESSES = 10
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
                    allowed_time = start_time + datetime.timedelta(seconds=12 * 3600)
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
                random.choice(dropout), random.choice(wm_size), random.choice(em_size), gpu_id)
            p = subprocess.Popen(shlex.split(cmd))
            time.sleep(10)
            process_list.append(p)

        time.sleep(30)

run_search()
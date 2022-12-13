from utils import * # ./utils.py
import sys
sys.path.insert(0, '..')
import gutils
from module import FlowModule

import os
import argparse
import time
import jax
import numpy as np

def build_state(ckpt_dir, model_type, sample_data):
    if model_type == 'simple':
        model = create_simple_flow(False)
    elif model_type == 'vardeq':
        model = create_simple_flow(True)
    elif model_type == 'multi-simple':
        model = create_multiscale_flow(False)
    elif model_type == 'multi-vardeq':
        model = create_multiscale_flow(True)
    else:
        raise NotImplementedError

    dumb_lr_scheduler = FlowModule.build_lr_scheduler(0, 1)
    dumb_optimizer = FlowModule.build_optimizer(dumb_lr_scheduler)
    params = FlowModule.build_model(jax.random.PRNGKey(0), model, sample_data)
    state = FlowModule.build_state(model, params, dumb_optimizer)
    state = FlowModule.load_model(ckpt_dir, state, model)

    return model, state

def inference_time(model, state, rng, dataloader):
    st = time.time()
    avg_loss = FlowModule.pred_epoch(state, rng, dataloader,
            FlowModule.build_pred_step_fn())
    ed = time.time()

    return avg_loss, (ed - st) / len(dataloader) / model.import_samples

def num_of_params(state):
    res = jax.tree_util.tree_map(lambda x: np.prod(x.shape), state.params)
    return sum(jax.tree_util.tree_flatten(res)[0])

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-l', '--log_dir', required=True)
    parser.add_argument('-c', '--ckpt_dir', required=True)
    parser.add_argument('-d', '--dataset', default='MNIST')
    parser.add_argument('-b', '--batch_size', default=256)

    return parser.parse_args()

def main():
    args = parse_args()
    (_, _, _), (train_loader, val_loader, test_loader) = \
            gutils.load_data(args.dataset, img2np, [50000, 10000], args.batch_size)
    rng = jax.random.PRNGKey(0)

    # statistic
    from collections import defaultdict
    info = defaultdict(dict)
    for model_type in ['simple', 'vardeq', 'multi-simple', 'multi-vardeq']:
        model, state = build_state(os.path.join(args.ckpt_dir, model_type), \
                model_type, next(iter(test_loader))[0])
        num_params = num_of_params(state)
        train_avg_loss, _ = inference_time(model, state, rng, train_loader)
        val_avg_loss, _ = inference_time(model, state, rng, val_loader)
        test_avg_loss, time_spent = inference_time(model, state, rng, test_loader)
        
        info[model_type]['model'] = model_type
        info[model_type]['Train Bpd'] = train_avg_loss
        info[model_type]['Val Bpd'] = val_avg_loss
        info[model_type]['Test Bpd'] = test_avg_loss
        info[model_type]['Inference Time (ms)'] = time_spent * 1000
        info[model_type]['Num Params'] = num_params

    import pandas as pd
    df = pd.DataFrame(info)
    print(df)
    
    # print out examples
    #TODO  

if __name__ == '__main__':
    main()

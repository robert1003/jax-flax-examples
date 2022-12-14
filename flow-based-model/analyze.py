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

def sample_images(model, state, rng, input_size):
    samples, _ = FlowModule.sample(state, model.sample, rng, input_size)

    return samples

def interpolate(model, state, rng, img1, img2, num_steps=8):
    imgs = np.stack([img1, img2], axis=0)
    z, _, rng = FlowModule.encode(state, model.encode, rng, imgs)
    alpha = jnp.linspace(0, 1, num=num_steps).reshape(-1, 1, 1, 1)
    z_middle = z[0:1] * alpha + z[1:2] * (1 - alpha)
    imgs_middle, _ = FlowModule.sample(state, model.sample, rng,
            (num_steps,)+imgs.shape[1:], z_init=z_middle)

    return imgs_middle

def visualize_dequent_dist(model, state, rng, imgs, fig_name, title):
    ldj = np.zeros(imgs.shape[0], dtype=jnp.float32)
    dequant_vals = []

    from tqdm import trange
    for _ in trange(8, leave=False):
        if 'flows_0' in state.params: # vardeq
            deq_val, _, rng = model.flows[0].apply(
                    {'params': state.params['flows_0']},
                    imgs, ldj, rng, reverse=False)
        else: # simple
            deq_val, _, rng = model.flows[0](imgs, ldj, rng, reverse=False)
        dequant_vals.append(deq_val)
    dequant_vals = jnp.concatenate(dequant_vals, axis=0)
    dequant_vals = jax.device_get(dequant_vals.reshape(-1))

    import seaborn as sns
    from matplotlib.colors import to_rgb
    import matplotlib.pyplot as plt
    sns.set()
    plt.figure(figsize=(10, 3))
    plt.hist(dequant_vals, bins=256, color=to_rgb("C0")+(0.5,),
                edgecolor="C0", density=True)
    plt.title(title)
    plt.savefig(fig_name, bbox_inches='tight')
    plt.clf()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-l', '--log_dir', required=True)
    parser.add_argument('-c', '--ckpt_dir', required=True)
    parser.add_argument('-f', '--figure_dir', required=True)
    parser.add_argument('-d', '--dataset', default='MNIST')
    parser.add_argument('-b', '--batch_size', default=256)

    return parser.parse_args()

def main():
    args = parse_args()
    (_, _, test_set), (train_loader, val_loader, test_loader) = \
            gutils.load_data(args.dataset, img2np, [50000, 10000], args.batch_size)
    rng = jax.random.PRNGKey(0)

    # statistic
    from collections import defaultdict
    info = defaultdict(dict)
    for model_type in ['simple', 'vardeq', 'multi-simple', 'multi-vardeq']:
        model, state = build_state(os.path.join(args.ckpt_dir, model_type), \
                model_type, next(iter(test_loader))[0])

        # check num of params
        num_params = num_of_params(state)

        # check model performance
        train_avg_loss, _ = inference_time(model, state, rng, train_loader)
        val_avg_loss, _ = inference_time(model, state, rng, val_loader)
        test_avg_loss, time_spent = inference_time(model, state, rng, test_loader)
        info[model_type]['model'] = model_type
        info[model_type]['Train Bpd'] = train_avg_loss
        info[model_type]['Val Bpd'] = val_avg_loss
        info[model_type]['Test Bpd'] = test_avg_loss
        info[model_type]['Inference Time (ms)'] = time_spent * 1000
        info[model_type]['Num Params'] = num_params

        # sample some graphs for qualitative results
        if 'multi' not in model_type:
            samples = sample_images(model, state, rng, [16, 28, 28, 1])
        else:
            samples = sample_images(model, state, rng, [16, 7, 7, 8])
        show_imgs(samples, os.path.join(args.figure_dir,
            model_type + '_img.png'))

        # sample interpolation in latent space
        show_imgs(interpolate(model, state, rng,
            test_set[0][0], test_set[1][0], num_steps=8),
            os.path.join(args.figure_dir, model_type+'_interpolate_1.png'),
            title=model_type,
            row_size=8)
        show_imgs(interpolate(model, state, rng,
            test_set[2][0], test_set[3][0], num_steps=8),
            os.path.join(args.figure_dir, model_type+'_interpolate_2.png'),
            title=model_type,
            row_size=8)

        # show dequantization distribution
        visualize_dequent_dist(
                model, state, rng, next(iter(test_loader))[0],
                os.path.join(args.figure_dir, model_type+'_dequant.png'),
                model_type)

    import pandas as pd
    df = pd.DataFrame(info)
    print(df)
    
    # print out examples

    #TODO  

if __name__ == '__main__':
    main()

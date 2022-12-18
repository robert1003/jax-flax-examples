from utils import * # ./utils.py
import sys
sys.path.insert(0, '..')
import gutils
from module import FlowModule

import argparse
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model_type', required=True)
    parser.add_argument('-l', '--log_dir', required=True)
    parser.add_argument('-c', '--ckpt_dir', required=True)
    parser.add_argument('-d', '--dataset', default='MNIST')
    parser.add_argument('-b', '--batch_size', default=128)
    parser.add_argument('-e', '--num_epochs', default=200)
    parser.add_argument('--lr', default=1e-3)

    return parser.parse_args()

def main():
    # parse argument
    args = parse_args()

    # rng
    rng = jax.random.PRNGKey(0)
    def next_rng():
        nonlocal rng
        new_rng, rng = jax.random.split(rng, 2)
        return new_rng

    # load data & dataloader
    (train_set, val_set, test_set), (train_loader, val_loader, test_loader) = \
            gutils.load_img_data(args.dataset, img2np, [50000, 10000], args.batch_size)

    # setup logger
    logger = SummaryWriter(log_dir=args.log_dir)

    # build model
    if args.model_type == 'simple':
        model = create_simple_flow(False)
    elif args.model_type == 'vardeq':
        model = create_simple_flow(True)
    elif args.model_type == 'multi-simple':
        model = create_multiscale_flow(False)
    elif args.model_type == 'multi-vardeq':
        model = create_multiscale_flow(True)
    else:
        raise NotImplementedError


    # build optimizer, lr_scheduler, model param
    lr_scheduler = FlowModule.build_lr_scheduler(args.lr, len(train_loader))
    optimizer = FlowModule.build_optimizer(lr_scheduler)
    params = FlowModule.build_model(next_rng(), model,
            next(iter(train_loader))[0])
    state = FlowModule.build_state(model, params, optimizer)

    # build train, val, pred step
    train_step = FlowModule.build_train_step_fn()
    val_step = FlowModule.build_val_step_fn()
    pred_step = FlowModule.build_pred_step_fn()

    # start training for args.num_epochs
    best_val_bpd = 1e6
    for epoch in range(1, args.num_epochs+1):
        state, loss = FlowModule.train_epoch(state, next_rng(), train_loader, train_step, epoch)
        logger.add_scalar('train/bpd', loss, global_step=epoch)
        print('Epoch', epoch, 'train/bpd', loss)
        if epoch % 5 == 0:
            loss = FlowModule.val_epoch(state, next_rng(), val_loader, val_step)
            logger.add_scalar('val/bpd', loss, global_step=epoch)
            print('Epoch', epoch, 'val/bpd', loss)
            if loss < best_val_bpd:
                print('saving better model: {:.3f} -> {:.3f}'.format(
                    best_val_bpd, loss))
                FlowModule.save_model(args.ckpt_dir, state, epoch)
                best_val_bpd = loss

if __name__ == '__main__':
    main()

import sacred

from utils.misc import nested_dict_to_namespace
from gnn_trainer import GnnTrainer

ex = sacred.Experiment('gnn_train')
ex.add_named_config('train', 'cfgs/train.yaml')
ex.add_named_config('track', 'cfgs/track.yaml')
ex.add_named_config('gnn', 'cfgs/gnn_train.yaml')


@ex.main
def load_config(_config, _run):
    sacred.commands.print_config(_run)


if __name__ == '__main__':
    train_config = ex.named_configs['train']._conf
    track_config = ex.named_configs['track']._conf
    gnn_config = ex.named_configs['gnn']._conf

    train_args = nested_dict_to_namespace(train_config)
    track_args = nested_dict_to_namespace(track_config)
    gnn_args = nested_dict_to_namespace(gnn_config)

    main = GnnTrainer(train_args=train_args, track_args=track_args, gnn_args=gnn_args)
    if gnn_args.train:
        main.train()
    elif gnn_args.test:
        main.test()
    else:
        raise ValueError("Please assign a state in (train, test)")

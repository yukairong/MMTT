from argparse import Namespace
from pathlib import Path

import sacred
import torch
import yaml

from src.utils.misc import nested_dict_to_namespace

# from src.datasets import build_dataset

# 创建一条实验记录
ex = sacred.Experiment('train')
# 添加运行需要的配置文件 yaml格式
ex.add_config('cfgs/demo.yaml')


# 打印当前运行的参数和对应的值
@ex.main
def load_config(_config, _run):
    sacred.commands.print_config(_run)


def train(args: Namespace) -> None:
    print(args)
    # 保存运行配置参数的文件夹路径
    output_dir = Path(args.output_dir)
    # 如果记录本次运行的配置参数，就将其保存到output文件夹下的test.yaml文件
    if args.output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        yaml.dump(vars(args), open(output_dir / 'test.yaml', 'w'), allow_unicode=True)

    device = torch.device(args.device)
    # dataset_train = build_dataset(split='train')


if __name__ == '__main__':
    config = ex.run_commandline().config
    # 将args的字典参数转换成namespace
    args = nested_dict_to_namespace(config)
    train(args)

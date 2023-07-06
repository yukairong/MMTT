import os
from argparse import Namespace
from pathlib import Path

import sacred
import torch
import yaml
from torch.utils.data import DataLoader

from src.datasets import build_dataset
from src.utils import misc
from src.utils.misc import nested_dict_to_namespace

# 创建一条实验记录
ex = sacred.Experiment('train')
# 添加运行需要的配置文件 yaml格式
ex.add_config('../cfgs/train.yaml')


# 打印当前运行的参数和对应的值
@ex.main
def load_config(_config, _run):
    sacred.commands.print_config(_run)


def train(args: Namespace) -> None:
    # ********************************  参数初始化 **********************************************************************
    print('\n')
    # 保存运行配置参数的文件夹路径
    output_dir = Path(args.output_dir)
    # 如果记录本次运行的配置参数，就将其保存到output文件夹下的test.yaml文件
    if args.output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        yaml.dump(vars(args), open(output_dir / 'train.yaml', 'w'), allow_unicode=True)

    device = torch.device(args.device)
    seed = args.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # TODO 构建模型 优化器 ....
    # ******************************* 构建模型 **************************************************************************
    # build_model

    # ****************************** 构建数据集 **************************************************************************
    dataset_train = build_dataset(split='train', args=args)
    img, target = dataset_train.__getitem__(0)
    print('\n.......... for test .........\n', f'img.shape = {img.shape}, 这张图片的目标数 = ')

    # dataset_val = build_dataset(split='val', args=args)

    if args.distributed:
        pass
        # sampler_train = utils.DistributedWeightedSampler(dataset_train)
        # # sampler_train = DistributedSampler(dataset_train)
        # sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        # 训练集使用随机采样器
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        # 验证集使用顺序采样器
        # sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # 训练集的批采样
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset=dataset_train,
                                   batch_sampler=batch_sampler_train,
                                   collate_fn=misc.collate_fn,
                                   num_workers=args.num_workers)

    # data_loader_val = DataLoader(
    #     dataset=dataset_val,
    #     batch_size=args.batch_size,
    #     sampler=sampler_val,
    #     drop_last=False,
    #     collate_fn=misc.collate_fn,
    #     num_workers=args.num_workers)

    # ****************************** 开始训练 **************************************************************************
    print('\n........start training.....\n')

    for i, (samples, targets) in enumerate(data_loader_train):
        samples = samples.to(device)
        targets = [misc.nested_dict_to_device(t, device) for t in targets]
        break


if __name__ == '__main__':
    config = ex.run_commandline().config
    # 将args的字典参数转换成namespace
    args = nested_dict_to_namespace(config)
    train(args)

import os
import pprint
import numpy as np
import sys
import argparse
import time
import warnings
from functools import partial
import random
import datetime
from math import e
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import resnet
from resnet import MultiBatchNorm2d
from utils import *

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')


def qac(qims, flag=1):
    if flag == 0:
        return 1.
    return np.sum(1. / np.array(qims)) / 128.


def to_status(m, status):
    if hasattr(m, 'batch_type'):
        m.batch_type = status


def time_str(fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d_%H_%M_%S'
    return datetime.datetime.today().strftime(fmt)


def str2bool(str):
    return True if str.lower() == 'true' else False


class Config(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
        parser.add_argument('--bn_num',
                            type=int,
                            default=None)
        parser.add_argument('--network',
                            type=str,
                            default="resnet")
        parser.add_argument('--base_lr',
                            type=float,
                            default=0.1)
        parser.add_argument('--momentum',
                            type=float,
                            default=0.9)
        parser.add_argument('--weight_decay',
                            type=float,
                            default=5e-4)
        parser.add_argument('--base_epochs',
                            type=int,
                            default=200)
        parser.add_argument('--base_per_node_batch_size',
                            type=int,
                            default=128)
        parser.add_argument('--meta_per_node_batch_size',
                            type=int,
                            default=128)
        parser.add_argument('--dataset',
                            type=str,
                            default='cifar100')
        parser.add_argument('--milestones',
                            type=list,
                            default=[60, 120, 160])
        parser.add_argument('--num_workers',
                            type=int,
                            default=16)
        parser.add_argument('--seed', type=int, default=0, help='seed')
        parser.add_argument('--print_interval',
                            type=bool,
                            default=30)
        parser.add_argument('--meta_epochs',
                            type=int,
                            default=150)
        parser.add_argument('--meta_step_size',
                            type=float,
                            default=0.001)
        parser.add_argument('--meta_lr',
                            type=float,
                            default=0.004)
        parser.add_argument('--exp_decay_at_epoch', type=int, default=21)
        parser.add_argument('--qac',
                            type=int,
                            default=1, )

        args = parser.parse_args()

        self.base_qfs = [90, 60, 30, 10]
        self.meta_qfs = [80, 70, 50, 40, 20]
        self.anchor_qfs = [95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10]
        self.dataset = args.dataset
        self.base_per_node_batch_size = args.base_per_node_batch_size
        self.meta_per_node_batch_size = args.meta_per_node_batch_size
        self.base_epochs = args.base_epochs
        self.milestones = args.milestones
        self.base_lr = args.base_lr
        self.weight_decay = args.weight_decay
        self.num_workers = args.num_workers
        self.print_interval = args.print_interval
        self.momentum = args.momentum
        self.qac = args.qac
        self.meta_epochs = args.meta_epochs
        self.exp_decay_at_epoch = args.exp_decay_at_epoch
        self.meta_lr = args.meta_lr
        self.mt_weight = 1.
        self.lb_weight = 1.
        self.meta_step_size = args.meta_step_size

        self.bn_num = args.bn_num if args.bn_num else len(self.base_qfs)
        self.network = args.network

        if self.dataset == 'cifar10':
            self.num_classes = 10
        elif self.dataset == 'cifar100':
            self.num_classes = 100
        else:
            raise NotImplementedError

        self.seed = args.seed

        working_dir = os.path.dirname(os.path.abspath(__file__))
        self.exp_dir = os.path.join(working_dir, 'outputs/{}'.format(self.network),
                                    '{}'.format(self.dataset),
                                    'base_qfs{}_meta_qfs{}'.format(str(self.base_qfs).replace(',', ''),
                                                                   str(self.meta_qfs).replace(',', '')),
                                    'meta_qpoch{}_l1_mss{}_qac{}_mt{}_lb{}'.format(self.meta_epochs,
                                                                                   self.meta_step_size, self.qac,
                                                                                   self.mt_weight, self.lb_weight),
                                    '{}'.format(time_str()))
        self.data_root = os.path.join(working_dir, 'cifar-{}/data'.format(self.num_classes))

        self.log = self.exp_dir + 'log'
        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)
        self.checkpoints = os.path.join(self.exp_dir, 'checkpoints')
        self.resume = os.path.join(self.checkpoints, 'latest.pth')

        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.2023, 0.1994, 0.2010]

        self.base_train_transform = transforms.Compose([
            AddMultiJpgNoise(self.base_qfs),
            MultiPad(4, padding_mode='reflect'),
            MultiRandomHorizontalFlip(),
            MultiRandomCrop(32),
            MultiToTensor(),
            MultiNormalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                           np.array([63.0, 62.1, 66.7]) / 255.0), ])
        self.base_train_dataset_init = {
            "root": self.data_root,
            "train": True,
            "download": True,
            "transform": self.base_train_transform
        }

        self.base_val_transform = [transforms.Compose([
            AddJpgNoise(jpg_quality),
            transforms.ToTensor(),
            transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                                 np.array([63.0, 62.1, 66.7]) / 255.0), ]) for jpg_quality in self.base_qfs]
        self.base_val_dataset_init = [{
            "root": self.data_root,
            "train": False,
            "download": True,
            "transform": transform
        } for transform in self.base_val_transform]

        self.meta_train_transform = [transforms.Compose([
            AddJpgNoise(jpg_quality),
            GetQim(),
            FormerToRGB(),
            FormerPad(4, padding_mode='reflect'),
            FormerRandomHorizontalFlip(),
            FormerRandomCrop(32),
            AllToTensor(),
            NormalizeWithoutQims(np.array([125.3, 123.0, 113.9]) / 255.0,
                                 np.array([63.0, 62.1, 66.7]) / 255.0), ]) for jpg_quality in self.base_qfs]

        self.meta_test_transform = [transforms.Compose([
            AddJpgNoise(jpg_quality),
            GetQim(),
            FormerToRGB(),
            FormerPad(4, padding_mode='reflect'),
            FormerRandomHorizontalFlip(),
            FormerRandomCrop(32),
            AllToTensor(),
            NormalizeWithoutQims(np.array([125.3, 123.0, 113.9]) / 255.0,
                                 np.array([63.0, 62.1, 66.7]) / 255.0), ]) for jpg_quality in self.meta_qfs]

        self.meta_train_dataset_init = [{
            "root": self.data_root,
            "train": True,
            "download": True,
            "transform": transform
        } for transform in self.meta_train_transform]
        self.meta_test_dataset_init = [{
            "root": self.data_root,
            "train": True,
            "download": True,
            "transform": transform
        } for transform in self.meta_test_transform]

        self.meta_val_transform = [transforms.Compose([
            AddJpgNoise(jpg_quality),
            GetQim(),
            FormerToRGB(),
            AllToTensor(),
            NormalizeWithoutQims(np.array([125.3, 123.0, 113.9]) / 255.0,
                                 np.array([63.0, 62.1, 66.7]) / 255.0), ]) for jpg_quality in self.anchor_qfs]
        self.meta_val_dataset_init = [{
            "root": self.data_root,
            "train": False,
            "download": True,
            "transform": transform
        } for transform in self.meta_val_transform]


def base_train(train_loader, model, criterion, optimizer, scheduler, epoch, args, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()

    losses_ce_s = [AverageMeter() for i in range(args.bn_num)]
    top1_s = [AverageMeter() for i in range(args.bn_num)]
    end = time.time()

    model.train()
    iters = len(train_loader.dataset) // (args.base_per_node_batch_size * gpus_num)
    prefetcher = DataPrefetcher(train_loader)
    inputs, labels = prefetcher.next()
    iter_index = 1
    while inputs is not None:
        data_time.update(time.time() - end)

        labels = labels.cuda()
        b, = labels.size()
        l = len(inputs)

        inputs = [i.cuda() for i in inputs]

        input_mix = torch.cat(inputs, dim=0)
        labels_mix = torch.cat([labels for i in range(l)], dim=0)
        outputs_mix = model(input_mix)

        losses_ce_tmp = []
        for i in range(l):
            losses_ce_tmp.append(qac(qims=get_qim(args.base_qfs[i]), flag=args.qac) * criterion(
                outputs_mix[i * b: (i + 1) * b], labels.data))

        loss_ce = sum(losses_ce_tmp)
        loss = loss_ce

        for i in range(l):
            losses_ce_s[i].update(losses_ce_tmp[i].item(), b)
            with torch.no_grad():
                top1_s[i].update(accuracy(outputs_mix[i * b:(i + 1) * b].data, labels.data, topk=(1,))[0], b)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        acc1, acc5 = accuracy(outputs_mix, labels_mix, topk=(1, 5))
        top1.update(acc1.item(), b)
        top5.update(acc5.item(), b)
        losses.update(loss.item(), b)

        batch_time.update(time.time() - end)
        end = time.time()

        inputs, labels = prefetcher.next()

        if iter_index % args.print_interval == 0:
            loss_str = "{:.2f}".format(losses.avg)
            top1_str = "{:.2f}".format(top1.avg)
            logger.info(
                'train: epoch {epoch:0>3d}|({batch}/{size}) lr: {lr:.6f} Data: {data:.2f}s | Batch: {bt:.2f}s | Loss: {loss:s} | top1: {top1:s} | top5: {top5: .1f}'.format(
                    epoch=epoch,
                    batch=iter_index,
                    size=iters,
                    lr=scheduler.get_lr()[0],
                    data=data_time.val,
                    bt=batch_time.val,
                    loss=loss_str,
                    top1=top1_str,
                    top5=top5.avg,
                ))

        iter_index += 1

    scheduler.step()

    return losses.avg, top1.avg, [a.avg for a in losses_ce_s], [c.avg for c in top1_s]


def validate(val_loader, model, args, stat, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.set_multibn(stat)
    model.eval()

    with torch.no_grad():
        end = time.time()
        for inputs, labels in val_loader:
            data_time.update(time.time() - end)
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            acc1, acc5 = accuracy(outputs.data, labels.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

    throughput = 1.0 / (batch_time.avg / inputs.size(0))

    return top1.avg, top5.avg, losses.avg, throughput


def meta_train(base_loaders, meta_loaders, model, mlp, criterion, optimizer, epoch, args, writer):
    meta_train_top1s = AverageMeter()
    meta_train_top5s = AverageMeter()
    meta_test_top1s = AverageMeter()
    meta_test_top5s = AverageMeter()
    losses = AverageMeter()
    meta_test_losses = AverageMeter()

    end = time.time()

    mlp.train()
    model.eval()

    iters = len(base_loaders[0].dataset) // (args.meta_per_node_batch_size * gpus_num)

    base_prefetchers = [DataPrefetcher(base_loader) for base_loader in base_loaders]
    meta_prefetchers = [DataPrefetcher(meta_loader) for meta_loader in meta_loaders]

    for iter_index in range(1, iters + 1):
        batch_time = AverageMeter()
        data_time = AverageMeter()

        next = []
        for base_prefetcher in base_prefetchers:
            next.append(base_prefetcher.next())
        i = random.choice(list(range(len(args.base_qfs))))
        inputs, labels = next[i]

        data_time.update(time.time() - end)

        labels = labels.cuda()
        inputs = [i.cuda() for i in inputs]

        b, = labels.size()
        l = len(inputs)

        bn_weight = mlp(inputs[1])
        model.set_multibn(bn_weight)
        outputs = model(inputs[0])
        loss_base = None
        if i == 0:
            loss_base = torch.nn.functional.l1_loss(bn_weight, torch.stack(
                [torch.tensor([1., 0., 0., 0.]) for i in range(b)]).cuda())
        elif i == 1:
            loss_base = torch.nn.functional.l1_loss(bn_weight, torch.stack(
                [torch.tensor([0., 1., 0., 0.]) for i in range(b)]).cuda())
        elif i == 2:
            loss_base = torch.nn.functional.l1_loss(bn_weight,
                                                    torch.stack(
                                                        [torch.tensor([0., 0., 1., 0.]) for i in range(b)]).cuda())
        elif i == 3:
            loss_base = torch.nn.functional.l1_loss(bn_weight,
                                                    torch.stack(
                                                        [torch.tensor([0., 0., 0., 1.]) for i in range(b)]).cuda())

        meta_train_loss = qac(inputs[1].cpu().numpy(), flag=args.qac) * criterion(
            outputs,
            labels.data)

        next_meta = []
        for meta_prefetcher in meta_prefetchers:
            next_meta.append(meta_prefetcher.next())
        j = random.choice(list(range(len(args.meta_qfs))))
        meta_inputs, meta_labels = next_meta[j]

        meta_labels = meta_labels.cuda()
        meta_inputs = [i.cuda() for i in meta_inputs]
        bn_weight = mlp(meta_inputs[1],
                        meta_loss=meta_train_loss,
                        meta_step_size=args.meta_step_size,
                        stop_gradient=False)
        model.set_multibn(bn_weight)
        meta_outputs = model(meta_inputs[0])

        meta_test_loss = qac(meta_inputs[1].cpu().numpy(), flag=args.qac) * criterion(
            meta_outputs,
            meta_labels.data)

        loss = meta_train_loss + args.mt_weight * meta_test_loss
        if loss_base is not None:
            loss += args.lb_weight * loss_base

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
        test_acc1, test_acc5 = accuracy(meta_outputs, meta_labels, topk=(1, 5))
        meta_train_top1s.update(acc1.item(), b)
        meta_train_top5s.update(acc5.item(), b)
        losses.update(loss.item(), b)
        meta_test_top1s.update(test_acc1.item(), b)
        meta_test_top5s.update(test_acc5.item(), b)
        meta_test_losses.update(meta_test_loss.item(), b)

        batch_time.update(time.time() - end)
        end = time.time()

        if iter_index % args.print_interval == 0:
            loss_str = "{:.6f}-{:.6f}".format(losses.avg, meta_test_losses.avg)
            top1_str = "{:.2f}-{:.2f}".format(meta_train_top1s.avg, meta_test_top1s.avg)
            top5_str = "{:.2f}-{:.2f}".format(meta_train_top5s.avg, meta_test_top5s.avg)
            logger.info(
                'train: epoch {epoch:0>3d}|({batch}/{size}) Data: {data:.2f}s | Batch: {bt:.2f}s | Loss: {loss:s} | top1: {top1:s} | top5: {top5:s}'.format(
                    epoch=epoch,
                    batch=iter_index,
                    size=iters,
                    data=data_time.val,
                    bt=batch_time.val,
                    loss=loss_str,
                    top1=top1_str,
                    top5=top5_str,
                ))
            iter_index += 1

    return losses.avg, meta_test_losses.avg, meta_train_top1s.avg, meta_test_top1s.avg


def meta_validate(val_loader, model, mlp, args, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    mlp.eval()

    with torch.no_grad():
        end = time.time()
        for inputs, labels in val_loader:
            data_time.update(time.time() - end)
            labels = labels.cuda()
            b, = labels.size()
            inputs = [i.cuda() for i in inputs]
            bn_weight = mlp(inputs[1])

            model.set_multibn(bn_weight)
            outputs = model(inputs[0])
            loss = criterion(outputs, labels)
            acc1, acc5 = accuracy(outputs.data, labels.data, topk=(1, 5))
            losses.update(loss.item(), b)
            top1.update(acc1.item(), b)
            top5.update(acc5.item(), b)
            batch_time.update(time.time() - end)
            end = time.time()

    throughput = 1.0 / (batch_time.avg / b)

    return top1.avg, top5.avg, losses.avg, throughput


def main():
    args = Config()
    pprint.pprint(args.__dict__)

    global logger
    logger = get_logger(__name__, args.log)
    logger.info(args.__dict__)
    writer = SummaryWriter(log_dir=os.path.join(args.checkpoints, 'tensorboard'))

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True

    global gpus_num
    gpus_num = torch.cuda.device_count()
    logger.info(f'use {gpus_num} gpus')
    logger.info(f"args: {args}")

    cudnn.benchmark = True
    cudnn.enabled = True
    start_time = time.time()

    logger.info('1. Start training batch normalization basis')
    logger.info('start loading data')
    if args.dataset == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(**args.base_train_dataset_init)
        val_datasets = [torchvision.datasets.CIFAR10(**val_dataset_init) for val_dataset_init in
                        args.base_val_dataset_init]
    elif args.dataset == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100(**args.base_train_dataset_init)
        val_datasets = [torchvision.datasets.CIFAR100(**val_dataset_init) for val_dataset_init in
                        args.base_val_dataset_init]
    else:
        raise NotImplementedError

    train_loader = DataLoader(train_dataset,
                              batch_size=args.base_per_node_batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=args.num_workers)

    val_loaders = [DataLoader(val_dataset,
                              batch_size=args.base_per_node_batch_size,
                              shuffle=False,
                              pin_memory=True,
                              num_workers=args.num_workers) for val_dataset in val_datasets]
    logger.info('finish loading data')

    logger.info(f"creating model '{args.network}'")

    if args.network == 'resnet':
        model = resnet.resnet18(num_classes=args.num_classes, norm_layer=MultiBatchNorm2d, bn_num=args.bn_num)
    else:
        raise NotImplementedError

    model.set_multibn(list(range(args.bn_num)))
    model = model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.base_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=0.2)

    start_epoch = 1

    acc1s = np.zeros(shape=len(args.anchor_qfs))
    acc5s = np.zeros(shape=len(args.anchor_qfs))
    losses = np.zeros(shape=len(args.anchor_qfs))

    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)

    logger.info('start base training')

    for i, val_loader in enumerate(val_loaders):
        stat = i
        acc1s[i], acc5s[i], losses[i], throughput = validate(val_loader, model, args, stat, criterion)
        logger.info(
            f"QF: {args.base_qfs[i]}, top1 acc: {acc1s[i]: .2f} %, top5 acc: {acc5s[i]: .2f} %, throughput: {throughput: .2f}sample / s")

    for epoch in range(start_epoch, args.base_epochs + 1):

        train_func = partial(base_train, train_loader, model, criterion, optimizer,
                             scheduler, epoch, args, writer)
        model.set_multibn(list(range(args.bn_num)))

        train_loss, train_acc, loss_ce_list, top1_list = train_func()
        loss_str = "{:.4f}".format(train_loss)
        top1_str = "{:.4f}".format(train_acc)
        logger.info('Epoch {epoch:0>3d}: | Loss: {loss:s} | top1: {top1:s}'.format(
            epoch=epoch,
            loss=loss_str,
            top1=top1_str,
        ))
        writer.add_scalar('Train/loss', train_loss, epoch)
        writer.add_scalar('Train/acc', train_acc, epoch)

        for i in range(args.bn_num):
            writer.add_scalar('Train/loss_ce_{}'.format(i), loss_ce_list[i], epoch)
            writer.add_scalar('Train/top-1_ce_{}'.format(i), top1_list[i], epoch)

        if epoch > 180:
            for i, val_loader in enumerate(val_loaders):
                stat = i
                acc1s[i], acc5s[i], losses[i], throughput = validate(val_loader, model, args, stat, criterion)
                writer.add_scalar('Test/loss_{}'.format(i), losses[i], epoch)
                writer.add_scalar('Test/acc_{}'.format(i), acc1s[i], epoch)
                logger.info(
                    f"QF: {args.base_qfs[i]}, top1 acc: {acc1s[i]: .2f} %, top5 acc: {acc5s[i]: .2f} %, throughput: {throughput: .2f}sample / s")

    training_time = (time.time() - start_time) / 3600
    logger.info("finish base training, total training time: {} hours".format(training_time))
    torch.save(model.state_dict(),
               os.path.join(
                   args.checkpoints, "{}-base.pth".format(
                       args.network)))

    logger.info("2. Starting meta training")
    logger.info('start loading data')
    if args.dataset == 'cifar10':
        meta_train_datasets = [torchvision.datasets.CIFAR10(**base_dataset_init) for base_dataset_init in
                               args.meta_train_dataset_init]
        meta_test_datasets = [torchvision.datasets.CIFAR10(**meta_dataset_init) for meta_dataset_init in
                              args.meta_test_dataset_init]
        meta_val_datasets = [torchvision.datasets.CIFAR10(**val_dataset_init) for val_dataset_init in
                             args.meta_val_dataset_init]
    elif args.dataset == 'cifar100':
        meta_train_datasets = [torchvision.datasets.CIFAR100(**base_dataset_init) for base_dataset_init in
                               args.meta_train_dataset_init]
        meta_test_datasets = [torchvision.datasets.CIFAR100(**meta_dataset_init) for meta_dataset_init in
                              args.meta_test_dataset_init]
        meta_val_datasets = [torchvision.datasets.CIFAR100(**val_dataset_init) for val_dataset_init in
                             args.meta_val_dataset_init]
    else:
        raise NotImplementedError

    meta_train_loaders = [DataLoader(base_dataset,
                                     batch_size=args.meta_per_node_batch_size,
                                     shuffle=True,
                                     pin_memory=True,
                                     num_workers=args.num_workers) for base_dataset in meta_train_datasets]
    meta_test_loaders = [DataLoader(meta_dataset,
                                    batch_size=args.meta_per_node_batch_size,
                                    shuffle=True,
                                    pin_memory=True,
                                    num_workers=args.num_workers) for meta_dataset in meta_test_datasets]
    meta_val_loaders = [DataLoader(val_dataset,
                                   batch_size=args.meta_per_node_batch_size,
                                   shuffle=False,
                                   pin_memory=True,
                                   num_workers=args.num_workers) for val_dataset in meta_val_datasets]
    logger.info('finish loading data')

    mlp = resnet.MetaMLP(out_channel=args.bn_num, channel=2)
    mlp = mlp.cuda()

    parameters_group = [{'params': mlp.parameters(), 'lr': args.meta_lr}, ]
    optimizer = torch.optim.Adam(parameters_group,
                                 weight_decay=args.weight_decay, )
    start_epoch = 1

    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)

    logger.info('start meta training')

    best_acc_avg = 0.
    acc1s = np.zeros(shape=len(args.anchor_qfs))
    acc5s = np.zeros(shape=len(args.anchor_qfs))
    losses = np.zeros(shape=len(args.anchor_qfs))

    for epoch in range(start_epoch, args.meta_epochs + 1):

        adjust_lr_exp(
            optimizer,
            args.meta_lr,
            epoch,
            args.meta_epochs,
            args.exp_decay_at_epoch,
            logger)

        train_func = partial(meta_train, meta_train_loaders, meta_test_loaders, model, mlp, criterion, optimizer, epoch,
                             args, writer)
        meta_train_loss, meta_test_loss, meta_train_top1, meta_test_top1 = train_func()

        writer.add_scalar('MetaTrain/loss', meta_train_loss, epoch)
        writer.add_scalar('MetaTest/loss', meta_test_loss, epoch)

        writer.add_scalar('MetaTrain/top-1', meta_train_top1, epoch)
        writer.add_scalar('MetaTest/top-1', meta_test_top1, epoch)

        for i, val_loader in enumerate(meta_val_loaders):
            acc1s[i], acc5s[i], losses[i], throughput = meta_validate(val_loader, model, mlp, args, criterion)
            writer.add_scalar('Test/loss_{}'.format(i), losses[i], epoch)
            writer.add_scalar('Test/acc_{}'.format(i), acc1s[i], epoch)
            logger.info(
                f"QF: {args.anchor_qfs[i]}, top1 acc: {acc1s[i]: .2f} %, top5 acc: {acc5s[i]: .2f} %, throughput: {throughput: .2f}sample / s")

        if best_acc_avg < acc1s.mean():
            best_acc_avg = acc1s.mean()
            best_acc = acc1s
            torch.save(model.state_dict(),
                       os.path.join(
                           args.checkpoints, "{}-best.pth".format(
                               args.network)))
            torch.save(mlp.state_dict(),
                       os.path.join(
                           args.checkpoints, "mlp-best.pth"))
            logger.info("best acc1 avg: {}, best acc1s: {}".format(best_acc_avg, best_acc))
        if epoch == args.meta_epochs:
            torch.save(
                model.state_dict(),
                os.path.join(
                    args.checkpoints, "{}-epoch{}-acc{}.pth".format(
                        args.network, epoch, best_acc_avg)))

    training_time = (time.time() - start_time) / 3600
    logger.info("finish training, total training time: {} hours".format(training_time))


if __name__ == '__main__':
    main()

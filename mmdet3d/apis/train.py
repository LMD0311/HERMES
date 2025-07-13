# Copyright (c) OpenMMLab. All rights reserved.
import torch
import warnings
import random
import warnings

import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer,
                         build_runner, GradientCumulativeFp16OptimizerHook, GradientCumulativeOptimizerHook)
from mmcv.utils import build_from_cfg

from mmdet.core import DistEvalHook, EvalHook
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_root_logger

from mmdet.apis import train_detector
from mmseg.apis import train_segmentor

import copy
from collections import defaultdict
from itertools import chain

from torch.nn.utils import clip_grad

from mmcv.utils import TORCH_VERSION, _BatchNorm, digit_version
from mmcv.runner.dist_utils import allreduce_grads
from mmcv.runner.fp16_utils import LossScaler, wrap_fp16_model
from mmcv.runner.hooks.hook import HOOKS, Hook
from mmcv.runner.hooks.optimizer import OptimizerHook

try:
    # If PyTorch version >= 1.6.0, torch.cuda.amp.GradScaler would be imported
    # and used; otherwise, auto fp16 will adopt mmcv's implementation.
    from torch.cuda.amp import GradScaler
except ImportError:
    pass


@HOOKS.register_module()
class Fp16GanOptimizerHook(OptimizerHook):
    """FP16 optimizer hook (using PyTorch's implementation).

    If you are using PyTorch >= 1.6, torch.cuda.amp is used as the backend,
    to take care of the optimization procedure.

    Args:
        loss_scale (float | str | dict): Scale factor configuration.
            If loss_scale is a float, static loss scaling will be used with
            the specified scale. If loss_scale is a string, it must be
            'dynamic', then dynamic loss scaling will be used.
            It can also be a dict containing arguments of GradScalar.
            Defaults to 512. For Pytorch >= 1.6, mmcv uses official
            implementation of GradScaler. If you use a dict version of
            loss_scale to create GradScaler, please refer to:
            https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler
            for the parameters.

    Examples:
        >>> loss_scale = dict(
        ...     init_scale=65536.0,
        ...     growth_factor=2.0,
        ...     backoff_factor=0.5,
        ...     growth_interval=2000
        ... )
        >>> optimizer_hook = Fp16OptimizerHook(loss_scale=loss_scale)
    """

    def __init__(self,
                 grad_clip=None,
                 coalesce=True,
                 bucket_size_mb=-1,
                 loss_scale=512.,
                 distributed=True):
        self.grad_clip = grad_clip
        self.coalesce = coalesce
        self.bucket_size_mb = bucket_size_mb
        self.distributed = distributed
        self._scale_update_param = None
        if loss_scale == 'dynamic':
            self.loss_scaler = GradScaler()
        elif isinstance(loss_scale, float):
            self._scale_update_param = loss_scale
            self.loss_scaler = GradScaler(init_scale=loss_scale)
        elif isinstance(loss_scale, dict):
            self.loss_scaler = GradScaler(**loss_scale)
        else:
            raise ValueError('loss_scale must be of type float, dict, or '
                             f'"dynamic", got {loss_scale}')

    def before_run(self, runner):
        """Preparing steps before Mixed Precision Training."""
        # wrap model mode to fp16
        wrap_fp16_model(runner.model)
        # resume from state dict
        if 'fp16' in runner.meta and 'loss_scaler' in runner.meta['fp16']:
            scaler_state_dict = runner.meta['fp16']['loss_scaler']
            self.loss_scaler.load_state_dict(scaler_state_dict)

    def copy_grads_to_fp32(self, fp16_net, fp32_weights):
        """Copy gradients from fp16 model to fp32 weight copy."""
        for fp32_param, fp16_param in zip(fp32_weights,
                                          fp16_net.parameters()):
            if fp16_param.grad is not None:
                if fp32_param.grad is None:
                    fp32_param.grad = fp32_param.data.new(
                        fp32_param.size())
                fp32_param.grad.copy_(fp16_param.grad)

    def copy_params_to_fp16(self, fp16_net, fp32_weights):
        """Copy updated params from fp32 weight copy to fp16 model."""
        for fp16_param, fp32_param in zip(fp16_net.parameters(),
                                          fp32_weights):
            fp16_param.data.copy_(fp32_param.data)

    def after_train_iter(self, runner):
        """Backward optimization steps for Mixed Precision Training. For
        dynamic loss scaling, please refer to
        https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler.

        1. Scale the loss by a scale factor.
        2. Backward the loss to obtain the gradients.
        3. Unscale the optimizer’s gradient tensors.
        4. Call optimizer.step() and update scale factor.
        5. Save loss_scaler state_dict for resume purpose.
        """
        # clear grads of last iteration
        runner.model.zero_grad()
        # if runner._iter % 2 == 0 or runner._iter < 30000:
        if runner._iter % 2 == 0:
            runner.optimizer["g"].zero_grad()

            self.loss_scaler.scale(runner.outputs['loss']).backward()
            self.loss_scaler.unscale_(runner.optimizer["g"])
            # grad clip
            if self.grad_clip is not None:
                grad_norm = self.clip_grads(runner.model.parameters())
                if grad_norm is not None:
                    # Add grad norm to the logger
                    runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                             runner.outputs['num_samples'])
            # backward and update scaler
            self.loss_scaler.step(runner.optimizer["g"])
            self.loss_scaler.update(self._scale_update_param)
        elif runner._iter % 2 == 1:
            runner.optimizer["d"].zero_grad()

            self.loss_scaler.scale(runner.outputs['loss']).backward()
            self.loss_scaler.unscale_(runner.optimizer["d"])
            # grad clip
            if self.grad_clip is not None:
                grad_norm = self.clip_grads(runner.model.parameters())
                if grad_norm is not None:
                    # Add grad norm to the logger
                    runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                             runner.outputs['num_samples'])
            # backward and update scaler
            self.loss_scaler.step(runner.optimizer["d"])
            self.loss_scaler.update(self._scale_update_param)

        # save state_dict of loss_scaler
        runner.meta.setdefault(
            'fp16', {})['loss_scaler'] = self.loss_scaler.state_dict()


def train_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                meta=None):
    """A function wrapper for launching model training according to cfg.

    Because we need different eval_hook in runner. Should be deprecated in the
    future.
    """
    if cfg.model.type in ['EncoderDecoder3D']:
        train_segmentor(
            model,
            dataset,
            cfg,
            distributed=distributed,
            validate=validate,
            timestamp=timestamp,
            meta=meta)
    elif hasattr(cfg.model, "pts_bbox_head") and hasattr(cfg.model.pts_bbox_head,
                                                         "render_ssl_cfg") and cfg.model.pts_bbox_head.render_ssl_cfg.type in [
        "NeuSModelV4"]:
        train_gan_detector(
            model,
            dataset,
            cfg,
            distributed=distributed,
            validate=validate,
            timestamp=timestamp,
            meta=meta
        )
    elif 'deepspeed' in cfg.keys():
        train_deepspeed_detector(
            model,
            dataset,
            cfg,
            distributed=distributed,
            validate=validate,
            timestamp=timestamp,
            meta=meta)
    else:
        train_detector(
            model,
            dataset,
            cfg,
            distributed=distributed,
            validate=validate,
            timestamp=timestamp,
            meta=meta)


def train_deepspeed_detector(model,
                             dataset,
                             cfg,
                             distributed=False,
                             validate=False,
                             timestamp=None,
                             meta=None):
    from mmdet.utils import (build_ddp, build_dp, build_ZeROddp,
                             find_latest_checkpoint, get_root_logger)
    import os
    logger = get_root_logger(cfg.log_level)
    logger.info("Using deepspeed for training")

    # # prepare data loaders
    # dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    # # The default loader config
    # # loader_cfg = dict(
    # #     # cfg.gpus will be ignored if distributed
    # #     num_gpus=len(cfg.gpu_ids),
    # #     dist=distributed,
    # #     seed=cfg.seed,
    # #     drop_last=True)
    # # # The overall dataloader settings
    # # loader_cfg.update({
    # #     k: v
    # #     for k, v in cfg.data.items() if k not in [
    # #         'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
    # #         'test_dataloader'
    # #     ]
    # # })
    # runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner[
    #     'type']
    # train_dataloader_default_args = dict(
    #     samples_per_gpu=2,
    #     workers_per_gpu=2,
    #     # `num_gpus` will be ignored if distributed
    #     num_gpus=len(cfg.gpu_ids),
    #     dist=distributed,
    #     seed=cfg.seed,
    #     runner_type=runner_type,
    #     persistent_workers=False
    # )
    #
    # train_loader_cfg = {
    #     **train_dataloader_default_args,
    #     **cfg.data.get('train_dataloader', {})
    # }
    #
    # # The specific dataloader settings
    # data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed) for ds in dataset
    ]

    logger.info("Data loaders built")

    # put model on gpus
    from mmdet.core import build_optimizer
    from mmcv.runner.checkpoint import load_checkpoint
    optimizer = build_optimizer(model, cfg.optimizer)
    deepspeed_enabled = 'deepspeed' in cfg.keys() and cfg.get('deepspeed') is True
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        if deepspeed_enabled:
            if cfg.load_from:
                logger.info("Loading checkpoint from {}".format(cfg.load_from))
                load_checkpoint(model, cfg.load_from, map_location="cpu", strict=False)

                import time
                time.sleep(5)

            cfg.optimizer = optimizer
            cfg.model = model
            model, optimizer, _, _ = build_ZeROddp(
                model=model,
                optimizer=optimizer,
                model_parameters=model.parameters(),
                args=cfg,
            )
            logger.info("Build ZeROddp model and optimizer successfully")
            # if cfg.load_from:
            #     message = model.load_state_dict({"module." + k: v for k, v in new_ckpt.items()}, strict=False)
            #     # print(2, torch.distributed.get_rank(), message)
            #     # logger.info(message)
            #     import time
            #     time.sleep(5)
            model.device_ids = [int(os.environ['LOCAL_RANK'])]
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    if 'runner' not in cfg:
        cfg.runner = {
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.total_epochs
        }
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)
    else:
        if 'total_epochs' in cfg:
            assert cfg.total_epochs == cfg.runner.max_epochs

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting

    # optimizer_config = GradientCumulativeFp16OptimizerHook(
    #     cumulative_iters=4, loss_scale='dynamic',
    #     grad_clip=dict(max_norm=35, norm_type=2), distributed=distributed)
    optimizer_config = OptimizerHook(**cfg.optimizer_config)

    if deepspeed_enabled:
        assert isinstance(optimizer_config, OptimizerHook), "deepspeed must use OptimizerHook"


    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,  # <class 'mmcv.runner.hooks.optimizer.OptimizerHook'>
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config=cfg.get('custom_hooks', None))

    if distributed:
        # when distributed training by epoch, using`DistSamplerSeedHook` to set
        # the different seed to distributed sampler for each epoch, it will
        # shuffle dataset at each epoch and avoid overfitting.
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # register eval hooks
    if validate:
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
        if val_samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    # user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from and not deepspeed_enabled:
        # DeepSpeed load_from has been handled before build_ZeROddp
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)


def train_gan_detector(model,
                       dataset,
                       cfg,
                       distributed=False,
                       validate=False,
                       timestamp=None,
                       meta=None):
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed) for ds in dataset
    ]

    # put model on gpus

    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer_g = build_optimizer(model, cfg.optimizer)
    discriminator = model.pts_bbox_head.render_model.discriminator if not hasattr(model,
                                                                                  "module") else model.module.pts_bbox_head.render_model.discriminator
    optimizer_d = build_optimizer(discriminator, cfg.optimizer_d)

    if 'runner' not in cfg:
        cfg.runner = {
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.total_epochs
        }
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)
    else:
        if 'total_epochs' in cfg:
            assert cfg.total_epochs == cfg.runner.max_epochs

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=dict(g=optimizer_g, d=optimizer_d),
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16GanOptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
        if val_samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    # user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)

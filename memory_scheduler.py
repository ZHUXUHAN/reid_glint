from default import config
import mxnet as mx


def get_scheduler():
    if config.scheduler_type == 'poly':
        backbone_lr_scheduler = mx.lr_scheduler.PolyScheduler(
            max_update=config.max_update,
            base_lr=config.backbone_lr,
            final_lr=config.backbone_final_lr,
            warmup_steps=config.warmup_steps,
        )
        memory_bank_lr_scheduler = mx.lr_scheduler.PolyScheduler(
            max_update=config.max_update,
            base_lr=config.memory_bank_lr,
            final_lr=config.memory_bank_final_lr,
            warmup_steps=config.warmup_steps,
        )
    elif config.scheduler_type == 'sgd':
        step = [int(x) for x in config.lr_steps.split(',')]
        backbone_lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(
            step=step, factor=0.1,  base_lr=config.backbone_lr
        )

        memory_bank_lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(
            step=step, factor=0.1, base_lr=config.memory_bank_lr / len(config.head_name_list))

    return backbone_lr_scheduler, memory_bank_lr_scheduler

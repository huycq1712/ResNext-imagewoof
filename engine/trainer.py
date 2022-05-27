import os
import time
import logging
import datetime

import torch
import torch.distributed as dist

from data.datasets.evaluation.ImagewoofDataset import accuracy, auc, f1_score


def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        meters,
        checkpoints,
        arguments
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("ResNeXt.train")
    logger.info("Start training")
    max_epoch = arguments["max_epoch"]
    epoch = arguments["epoch"]
    max_iter = max_epoch*len(train_loader)
    iteration = arguments["iteration"]
    distributed = arguments["distributed"]

    start_training_time = time.time()
    end = time.time()

    while epoch < max_epoch:

        epoch = epoch + 1
        arguments["epoch"] = epoch
        scheduler.step()
        for step, (images, labels, labels2, labels5) in enumerate(train_loader):
            data_time = time.time() - end
            inner_iter = step
            iteration = iteration + 1
            arguments["iteration"] = iteration

            model.train()

            images = images.to(device)
            predict = model(images)

            losses = loss_fn(predict, labels)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            batch_time = time.time() - end
            end = time.time()

            meters.update(time=batch_time, data=data_time)
            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if inner_iter % 1 == 0:
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "epoch [{epoch}][{inner_iter}/{num_iter}]",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        epoch=epoch,
                        inner_iter=inner_iter,
                        num_iter=len(train_loader),
                        meters=str(meters),
                        lr=optimizer.param_groups[-1]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )






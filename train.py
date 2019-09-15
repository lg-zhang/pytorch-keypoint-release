import logging
import argparse

import torch
from torch.utils.data import DataLoader
from dataset import PatchDataset
from modules import KeypointNet, LossFunction
from evaluation import (
    KeypointDetector,
    RepeatabilityEvalDataset,
    evaluate_repeatability_on_dataset,
)

logging.basicConfig(
    format="%(asctime)s %(levelname)-4s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%m-%d:%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def eval_model(model, eval_dset):
    model.eval()
    key_det = KeypointDetector(model)
    _, _, rep_max = evaluate_repeatability_on_dataset(
        eval_dset, key_det, top_n=200, use_maxima=True
    )
    _, _, rep_min = evaluate_repeatability_on_dataset(
        eval_dset, key_det, top_n=200, use_maxima=False
    )
    logger.info(f"maxima repeatability = {rep_max:.2f}%")
    logger.info(f"minima repeatability = {rep_min:.2f}%")

    model.train()
    return rep_max, rep_min


def main(args):
    # set up eval dataset
    eval_dset = RepeatabilityEvalDataset(args.eval_data)

    # set up network
    model = KeypointNet()
    if args.pretrained is not None:
        model.load_state_dict(torch.load(model))
    model = torch.nn.DataParallel(model).cuda()

    finetuning = args.peakedness_weight > 0

    if finetuning:
        rep_max, rep_min = eval_model(model, eval_dset)
        criterion = LossFunction(
            optimize_maxima=rep_max > rep_min, peakedness_weight=args.peakedness_weight
        )
        model_p = "tuned.pth"
    else:
        criterion = LossFunction()
        model_p = "pretrained.pth"

    criterion = criterion.cuda()

    optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # training dataset
    train_dset = PatchDataset(args.train_data, 71 if finetuning else 65)
    train_loader = DataLoader(
        train_dset,
        batch_size=args.bs * 2,  # double to form quadruple
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True,
    )

    for epoch_idx in range(args.num_epochs):
        logger.info(f"starting epoch {epoch_idx}...")
        model.train()
        for batch_idx, batch_data in enumerate(train_loader):
            step = epoch_idx * len(train_loader) + batch_idx

            pa, pb = batch_data
            pa, pb = pa.cuda(), pb.cuda()

            n = pa.size(0) // 2

            # split batch into two halves
            pa1, pa2 = pa[:n], pa[n:]
            pb1, pb2 = pb[:n], pb[n:]

            loss = criterion(model(pa1), model(pa2), model(pb1), model(pb2))

            if step % args.log_interval == 0:
                logger.info(
                    f"Epoch {epoch_idx} [{batch_idx}/{len(train_loader)}]: loss = {loss.item():.4f}"
                )

            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)

            optimizer.step()

        torch.save(model.module.state_dict(), model_p)

        logger.info("evaluating repeatability...")
        rep_max, rep_min = eval_model(model, eval_dset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    def add_arg(*args, **kwargs):
        kwargs["help"] = "(default: %(default)s)"
        if not kwargs.get("type", bool) == bool:
            kwargs["metavar"] = ""
        parser.add_argument(*args, **kwargs)

    # training params
    add_arg("--num_epochs", type=int, default=10)
    add_arg("--log_interval", type=int, default=10)
    add_arg("--lr", type=float, default=0.01)
    add_arg("--bs", type=int, default=256)
    add_arg("--pretrained", type=str, default=None)
    add_arg("--peakedness_weight", type=float, default=0)

    add_arg("--train_data", type=str, default=None)
    add_arg("--eval_data", type=str, default=None)

    args = parser.parse_args()
    main(args)

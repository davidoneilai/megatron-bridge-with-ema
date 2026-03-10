import argparse

from megatron.bridge.recipes.llama import llama32_1b_pretrain_config
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain

from EMA import EMACallback


def parse_args():
    parser = argparse.ArgumentParser()

    # treino
    parser.add_argument("--train-iters", type=int, default=30)
    parser.add_argument("--seq-length", type=int, default=1024)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--lr-decay-iters", type=int, default=10000)

    # EMA
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.95)
    parser.add_argument("--ema-start-step", type=int, default=0)
    parser.add_argument("--ema-store-on-cpu", action="store_true")
    parser.add_argument("--ema-log-interval", type=int, default=5)
    parser.add_argument(
        "--save",
        type=str,
        default="/opt/Megatron-Bridge/nemo_experiments/default/checkpoints",
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
    )
    return parser.parse_args()


def build_config(args):
    cfg = llama32_1b_pretrain_config()

    cfg.train.train_iters = args.train_iters
    cfg.scheduler.lr_decay_iters = args.lr_decay_iters

    cfg.model.seq_length = args.seq_length
    cfg.model.vocab_size = args.vocab_size
    cfg.tokenizer.vocab_size = args.vocab_size

    if hasattr(cfg.dataset, "seq_length"):
        cfg.dataset.seq_length = args.seq_length
    if hasattr(cfg.dataset, "sequence_length"):
        cfg.dataset.sequence_length = args.seq_length

    cfg.train.micro_batch_size = 1
    cfg.train.global_batch_size = 1

    cfg.checkpoint.save = args.save
    cfg.checkpoint.async_save = False

    if args.load is not None:
        cfg.checkpoint.load = args.load

    return cfg

def main():
    args = parse_args()
    cfg = build_config(args)

    callbacks = []

    if args.use_ema:
        callbacks.append(
            EMACallback(
                decay=args.ema_decay,
                start_step=args.ema_start_step,
                store_on_cpu=args.ema_store_on_cpu,
                log_interval=args.ema_log_interval,
            )
        )

    pretrain(cfg, forward_step, callbacks=callbacks)


if __name__ == "__main__":
    main()
"""
Sample from a trained model
"""
import argparse
import os
import pickle
from contextlib import nullcontext
from distutils.util import strtobool

import numpy as np
import tiktoken
import torch
from torch.utils.data import random_split

import misc_utils as mu
from model import GPTConfig, GPT_interchannel


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--num_samples", type=int, default=9)
    parser.add_argument("--nrows", type=int, default=3)
    parser.add_argument("--ncols", type=int, default=3)
    parser.add_argument(
        "--sample_prompt",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
    )
    parser.add_argument("--prompt_size", type=int, default=150)
    parser.add_argument("--token_len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--top_k",
        type=int,
        default=200,
        help="retain only the top_k most likely tokens, clamp others to have 0 probability",
    )
    parser.add_argument(
        "--duplicate",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="ignore num of samples and generate data for the same prompt 9 times",
    )
    parser.add_argument("--median_filter_size", type=int, default=9)
    parser.add_argument("--independent", type=bool, default=False)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    # some other parameters
    # -----------------------------------------------------------------------------
    init_from = "resume"
    start = "\n"  # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
    nrows = 3
    ncols = 3
    temperature = (
        0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    )
    seed = args.seed
    device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
    dtype = "bfloat16"  # 'float32' or 'bfloat16' or 'float16'
    compile = True  # use PyTorch 2.0 to compile the model to be faster
    # -----------------------------------------------------------------------------

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # do we need this in training as well?
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = (
        "cuda" if "cuda" in device else "cpu"
    )  # for later use in torch.autocast
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )

    if init_from == "resume":
        # init from a model saved in a specific directory
        ckpt_path = os.path.join(args.ckpt_path, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=device)
        filter_class = checkpoint["config"]["filter_class"]
        gptconf = GPTConfig(**checkpoint["model_args"])
        model = GPT_interchannel(gptconf)
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)

    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model)  # requires PyTorch 2.0 (optional)

    from chatemg_dataset import ChatEMGDataset

    test_dataset = ChatEMGDataset(
        csv_files=["../data/p7_131.csv"],
        filter_class=filter_class,
        block_size=args.token_len,
        median_filter_size=args.median_filter_size
        if args.median_filter_size is not None
        else checkpoint["config"]["median_filter_size"],
    )
    if not args.duplicate:
        real_x = mu.sample_from_dataset(test_dataset, args.num_samples)[0]
    else:
        real_x = mu.sample_from_dataset(test_dataset, 1)[0]
        real_x = np.tile(real_x, (9, 1, 1))

    if args.sample_prompt:
        if args.independent:
            x = torch.tensor(real_x, device=device)
        else:
            x = torch.tensor(real_x[:, : args.prompt_size, :], device=device)
    else:
        x = torch.tensor(
            [[[0, 0, 0, 0, 0, 0, 0, 0]]] * args.num_samples,
            dtype=torch.long,
            device=device,
        )

    with torch.no_grad():
        with ctx:
            num_new_tokens = args.token_len - args.prompt_size
            Y = model.generate(
                x,
                num_new_tokens,
                temperature=temperature,
                top_k=args.top_k,
                prompt_size=args.prompt_size,
                independent=args.independent,
            )
            Y = Y.cpu().numpy()

    # use real_x and Y to compute the mse error over the whole predicted trajectory. Both are numpy array
    mse = mu.compute_mse(real_x, Y, starting_pos=150)
    rmse = np.sqrt(mse)

    mu.plot_emg_chunks_parallel(
        real_x,
        Y,
        rmse=rmse,
        nrows=nrows,
        ncols=ncols,
        vertical_location=None,
        save_fnm="real_vs_synthetic.png",
    )
    print("done")

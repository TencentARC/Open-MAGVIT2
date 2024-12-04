"""
Reference to
https://github.com/CompVis/taming-transformers
"""

import argparse, os, sys, glob
import torch
import time
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import repeat
import importlib
from src.Open_MAGVIT2.modules.transformer.gpt import sample_Open_MAGVIT2
from src.IBQ.modules.transformer.llama import sample_IBQ
import time
try:
    import torch_npu
except:
    pass

if hasattr(torch, "npu"):
    DEVICE = torch.device("npu:0" if torch_npu.npu.is_available() else "cpu")
else:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

rescale = lambda x: (x + 1.) / 2.

SAMPLE = {
    "Open-MAGVIT2": sample_Open_MAGVIT2,
    "IBQ": sample_IBQ
}

def get_obj_from_str(string, reload=False):
    print(string)
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "class_path" in config:
        raise KeyError("Expected key `class_path` to instantiate.")
    return get_obj_from_str(config["class_path"])(**config.get("init_args", dict()))

def chw_to_pillow(x):
    return Image.fromarray((255*rescale(x.detach().cpu().numpy().transpose(1,2,0))).clip(0,255).astype(np.uint8))


@torch.no_grad()
def sample_classconditional(model, batch_size, class_label, model_type, steps=256, temperature=None, top_k=None, callback=None,
                            dim_z=18, h=16, w=16, verbose_time=False, top_p=None, token_factorization=False, 
                            cfg_scale=1.0):
    log = dict()
    assert type(class_label) == int, f'expecting type int but type is {type(class_label)}'
    assert not model.be_unconditional, 'Expecting a class-conditional Net2NetTransformer.'
    c_indices = repeat(torch.tensor([class_label]), '1 -> b 1', b=batch_size).to(model.device)  # class token
    if token_factorization:
        if cfg_scale[0] > 1.0:
            cond_null = torch.ones_like(c_indices) * model.transformer.config.class_num
            cond_combined = torch.concat([c_indices, cond_null], dim=0) #(2B 1)
        else:
            cond_combined = c_indices # B 1
    else:
        if cfg_scale > 1.0:
            cond_null = torch.ones_like(c_indices) * model.transformer.config.class_num
            cond_combined = torch.concat([c_indices, cond_null], dim=0) #(2B 1)
        else:
            cond_combined = c_indices # B 1

    qzshape = [batch_size, dim_z, h, w]
    
    t1 = time.time()
    index_sample = SAMPLE[model_type](cond_combined, model.transformer, steps=steps,
                            sample_logits=True, top_k=top_k, callback=callback,
                            temperature=temperature, top_p=top_p, token_factorization=token_factorization,
                            cfg_scale=cfg_scale)
    if verbose_time:
        sampling_time = time.time() - t1
        print(f"Full sampling takes about {sampling_time:.2f} seconds.")
    x_sample = model.decode_to_img(index_sample, qzshape)
    log["samples"] = x_sample
    log["class_label"] = c_indices
    return log

@torch.no_grad()
def run_for_evaluation(logdir, model, batch_size, temperature, top_k, model_type, dim_z, unconditional=True, num_samples=50000,
        given_classes=None, top_p=None, token_factorization=False, cfg_scale=1.0, chunk_id=0):
    batches = [batch_size for _ in range(num_samples//batch_size)] + [num_samples % batch_size]
    assert given_classes is not None
    print("Running in pure class-conditional sampling mode. I will produce "
            f"{num_samples} samples for each of the {len(given_classes)} classes, "
            f"i.e. {num_samples*len(given_classes)} in total.")
    images_npz = []
    for class_label in tqdm(given_classes, desc="Classes"):
        for n, bs in tqdm(enumerate(batches), desc="Sampling Class"):
            if bs == 0: break
            logs = sample_classconditional(model, batch_size=bs, class_label=class_label,
                                            temperature=temperature, top_k=top_k, top_p=top_p, token_factorization=token_factorization
                                            ,cfg_scale=cfg_scale, dim_z=dim_z, model_type=model_type)
            batch_images = save_npz_from_logs(logs, logdir, base_count=n * batch_size)
            images_npz.append(batch_images)

    images_npz = np.vstack(images_npz)
    np.savez(os.path.join(logdir, 'samples_{}.npz'.format(chunk_id)), images_npz)

def save_npz_from_logs(logs, logdir, base_count, key="samples", cond_key=None):
    xx = logs[key]
    xs = []
    for i, x in enumerate(xx):
        x = chw_to_pillow(x)
        xs.append(x)
    return np.stack(xs)

def get_parser():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        nargs="?",
        help="path where the samples will be logged to.",
        default=""
    )
    parser.add_argument(
        "--config",
        nargs="*",
        metavar="config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-n",
        "--num_samples",
        type=int,
        nargs="?",
        help="num_samples to draw",
        default=50000
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="?",
        help="the batch size",
        default=25
    )
    parser.add_argument(
        "-k",
        "--top_k",
        type=str,
        nargs="?",
        help="top-k value to sample with",
        default=250,
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=str,
        nargs="?",
        help="temperature value to sample with",
        default=1.0
    )
    parser.add_argument(
        "-p",
        "--top_p",
        type=str,
        nargs="?",
        help="top-p value to sample with",
        default=1.0
    )
    parser.add_argument(
        "--classes",
        type=str,
        nargs="?",
        help="specify comma-separated classes to sample from. Uses 1000 classes per default.",
        default="imagenet"
    )
    parser.add_argument(
        "--token_factorization",
        action="store_true",
        help="whether to use token factorization"
    )
    parser.add_argument(
        "--cfg_scale",
        type=str,
        default=1.0
    )
    parser.add_argument(
        "--global_seed",
        type=int,
    )
    parser.add_argument(
        "--chunk_idx",
        type=int,
        default=0
    )
    parser.add_argument(
        "--num_chunks",
        type=int,
        default=4
    )
    parser.add_argument(
        "--model",
        choices=["Open-MAGVIT2", "IBQ"]
    )
    return parser

def load_model_from_config(config, sd, gpu=True, eval_mode=True):
    model = instantiate_from_config(config)
    if sd is not None:
        model.load_state_dict(sd, strict=False)
    if gpu:
        model = model.to(DEVICE)
    if eval_mode:
        model.eval()
    return {"model": model}


def load_model(config, ckpt, gpu, eval_mode):
    # load the specified checkpoint
    if ckpt:
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd.get("global_step", None)
        if global_step:
            print(f"loaded model from global step {global_step}.")
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model, pl_sd["state_dict"], gpu=gpu, eval_mode=eval_mode)["model"]
    return model, global_step


if __name__ == "__main__":
    sys.path.append(os.getcwd())
    parser = get_parser()

    opt, unknown = parser.parse_known_args()
    seed = opt.global_seed * opt.num_chunks + opt.chunk_idx
    torch.manual_seed(seed)
    logdir = opt.outdir
    ckpt = opt.ckpt
    ckpt_name = ckpt.split("/")[-1]
    config = OmegaConf.load(opt.config[0]) #since only one config

    model, global_step = load_model(config, ckpt, gpu=True, eval_mode=True)

    ## handle topk topp tempeature and cfg_scale
    if opt.token_factorization:
        opt.top_k = [int(topk) for topk in opt.top_k.split(",")]
        opt.top_p = [float(topp) for topp in opt.top_p.split(",")]
        opt.temperature = [float(temp) for temp in opt.temperature.split(",")]
        opt.cfg_scale = [float(cfg_scal) for cfg_scal in opt.cfg_scale.split(",")]
    else:
        opt.top_k = [int(topk) for topk in opt.top_k.split(",")][0]
        opt.top_p = [float(topp) for topp in opt.top_p.split(",")][0]
        opt.temperature = [float(temp) for temp in opt.temperature.split(",")][0]
        opt.cfg_scale = [float(cfg_scal) for cfg_scal in opt.cfg_scale.split(",")][0]

    dim_z = config.model.init_args.first_stage_config.params.embed_dim

    chunk_id = opt.chunk_idx
    if opt.classes == "imagenet":
        given_classes = [i for i in range(1000)]
        chunk_classes = len(given_classes) // opt.num_chunks ## should be divided
        class_chunks = given_classes[ chunk_id * chunk_classes : (chunk_id+1) * chunk_classes ]
        given_classes = class_chunks
    else:
        cls_str = opt.classes
        assert not cls_str.endswith(","), 'class string should not end with a ","'
        given_classes = [int(c) for c in cls_str.split(",")]

    ### The ckpt should be only a name and the logdir is the version dir
    if opt.token_factorization:
        logdir = os.path.join(logdir, "samples", f"top_k_{opt.top_k[0]}_{opt.top_k[1]}_temp_{opt.temperature[0]:.2f}_{opt.temperature[1]:.2f}_top_p_{opt.top_p[0]}_{opt.top_p[1]}_cfg_{opt.cfg_scale[0]}_{opt.cfg_scale[1]}",
                            f"{ckpt_name}")
    else:
        logdir = os.path.join(logdir, "samples",
                        f"top_k_{opt.top_k}_temp_{opt.temperature:.2f}_top_p_{opt.top_p}_cfg_{opt.cfg_scale}",
                        f"{ckpt_name}")

    print(f"Logging to {logdir}")
    os.makedirs(logdir, exist_ok=True)

    start_time = time.time()
    run_for_evaluation(logdir, model, opt.batch_size, opt.temperature, opt.top_k, unconditional=model.be_unconditional,
        given_classes=given_classes, num_samples=opt.num_samples, top_p=opt.top_p, token_factorization=opt.token_factorization,
        cfg_scale=opt.cfg_scale, chunk_id=chunk_id, model_type=opt.model, dim_z=dim_z)
    end_time = time.time()
    print(end_time - start_time, 's')

    print("done.")
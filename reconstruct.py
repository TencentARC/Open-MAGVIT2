"""
Image Reconstruction code
"""
import os
import sys
sys.path.append(os.getcwd())
import torch
from omegaconf import OmegaConf
import importlib
import numpy as np
from PIL import Image
from tqdm import tqdm
from src.Open_MAGVIT2.models.lfqgan import VQModel
from src.IBQ.models.ibqgan import IBQ
import argparse
try:
	import torch_npu
except:
    pass

if hasattr(torch, "npu"):
    DEVICE = torch.device("npu:0" if torch_npu.npu.is_available() else "cpu")
else:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## for different model configuration
MODEL_TYPE = {
    "Open-MAGVIT2": VQModel,
    "IBQ": IBQ
}

def load_vqgan_new(config, model_type, ckpt_path=None, is_gumbel=False):
	model = MODEL_TYPE[model_type](**config.model.init_args)
	if ckpt_path is not None:
		sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
		missing, unexpected = model.load_state_dict(sd, strict=False)
	return model.eval()

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

def custom_to_pil(x):
	x = x.detach().cpu()
	x = torch.clamp(x, -1., 1.)
	x = (x + 1.)/2.
	x = x.permute(1,2,0).numpy()
	x = (255*x).astype(np.uint8)
	x = Image.fromarray(x)
	if not x.mode == "RGB":
		x = x.convert("RGB")
	return x

def main(args):
    config_file = args.config_file
    configs = OmegaConf.load(config_file)
    configs.data.init_args.batch_size = args.batch_size # change the batch size
    configs.data.init_args.test.params.config.size = args.image_size #using test to inference
    configs.data.init_args.test.params.config.subset = args.subset #using the specific data for comparsion

    model = load_vqgan_new(configs, args.model, args.ckpt_path).to(DEVICE)

    visualize_dir = args.save_dir
    visualize_version = args.version
    visualize_original = os.path.join(visualize_dir, visualize_version, "original_{}".format(args.image_size))
    visualize_rec = os.path.join(visualize_dir, visualize_version, "rec_{}".format(args.image_size))
    if not os.path.exists(visualize_original):
       os.makedirs(visualize_original, exist_ok=True)
    
    if not os.path.exists(visualize_rec):
       os.makedirs(visualize_rec, exist_ok=True)
    
    dataset = instantiate_from_config(configs.data)
    dataset.prepare_data()
    dataset.setup()

    count = 0
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(dataset._val_dataloader())):
            if count > args.image_num:
               break
            images = batch["image"].permute(0, 3, 1, 2).to(DEVICE)

            count += images.shape[0]
            if model.use_ema:
                with model.ema_scope():
                    if args.model == "Open-MAGVIT2":
                        quant, diff, indices, _ = model.encode(images)
                    elif args.model == "IBQ":
                        quant, qloss, (_, _, indices) = model.encode(images)
                    reconstructed_images = model.decode(quant)
            else:
                if args.model == "Open-MAGVIT2":
                    quant, diff, indices, _ = model.encode(images)
                elif args.model == "IBQ":
                    quant, qloss, (_, _, indices) = model.encode(images)
                reconstructed_images = model.decode(quant)
            
            image = images[0]
            reconstructed_image = reconstructed_images[0]

            image = custom_to_pil(image)
            reconstructed_image = custom_to_pil(reconstructed_image)

            image.save(os.path.join(visualize_original, "{}.png".format(idx)))
            reconstructed_image.save(os.path.join(visualize_rec, "{}.png".format(idx)))

    
def get_args():
   parser = argparse.ArgumentParser(description="inference parameters")
   parser.add_argument("--config_file", required=True, type=str)
   parser.add_argument("--ckpt_path", required=True, type=str)
   parser.add_argument("--image_size", default=256, type=int)
   parser.add_argument("--batch_size", default=1, type=int) ## inference only using 1 batch size
   parser.add_argument("--image_num", default=50, type=int)
   parser.add_argument("--subset", default=None)
   parser.add_argument("--version", type=str, required=True)
   parser.add_argument("--save_dir", type=str, required=True)
   parser.add_argument("--model", choices=["Open-MAGVIT2", "IBQ"])

   return parser.parse_args()
  
if __name__ == "__main__":
  args = get_args()
  main(args)
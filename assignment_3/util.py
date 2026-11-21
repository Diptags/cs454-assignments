from PIL import Image

import torch
from torchvision.models import vgg11, VGG11_Weights

MODEL = vgg11(weights=VGG11_Weights.DEFAULT)
MODEL.eval()
IMG_SIZE = ((224, 224))
with open('category_names.txt') as f:
    IDX2NAME = eval(f.read())

SOFTMAX = torch.nn.Softmax(dim=1)
TRANSFORM = VGG11_Weights.DEFAULT.transforms(antialias=True)
IMAGES_ORG_PREDS = {
    "images/knot.jpg": 616,
    "images/mink.jpg": 357,
    "images/orca.jpg": 148,
    "images/sock.jpg": 806,
    "images/tank.jpg": 847,
}

def set_random_seed(seed:int):
    torch.manual_seed(seed)

def get_random_noise() -> torch.Tensor:
    return torch.randn(1, 3, *IMG_SIZE)

def get_image_tensor_from(image_path:str) -> torch.Tensor:
    with Image.open(image_path) as image:
        img_tensor = TRANSFORM(image.resize(IMG_SIZE)).unsqueeze(0)
    return img_tensor

def get_model_predictions_for(image_tensor:torch.Tensor) -> torch.Tensor:
    output = SOFTMAX(MODEL(image_tensor.clip(-3, 3)))
    return output

def get_model_most_likely_for(image_tensor:torch.Tensor):
    output = get_model_predictions_for(image_tensor)
    prob, idx = torch.max(output, dim=1)
    prob, idx = prob.item(), idx.item()
    return prob, idx

def get_name_for(idx:int):
    return IDX2NAME[idx]

def get_Linf_norm_for(noise_tensor:torch.Tensor) -> float:
    return max(abs(torch.max(noise_tensor).item()), abs(torch.min(noise_tensor).item()))

def get_L2_norm_for(noise_tensor:torch.Tensor) -> float:
    return torch.sum(noise_tensor.pow(2)).item()

def get_L0_norm_for(noise_tensor:torch.Tensor) -> int:
    return torch.sum(noise_tensor != 0).item()

def get_pixel_norm_for(noise_tensor:torch.Tensor) -> int:
    return torch.sum(torch.sum(noise_tensor, dim=1) != 0).item()

def _unnormalize(image_tensor:torch.Tensor):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    for t, m, s in zip(image_tensor, means, stds):
        t.mul_(s).add_(m)
    return image_tensor

def get_viz_for(image_tensor:torch.Tensor, save_to:str='visualization.png', show_k=10, max_len=20):
    image_tensor = image_tensor.clip(-3, 3)
    assert image_tensor.size(0) == 1
    raw_filename = "raw_{}".format(save_to)
    torch.save(image_tensor, raw_filename)
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    
    probabilities = get_model_predictions_for(image_tensor)
    topk_results = torch.topk(probabilities[0], k=show_k)
    p_values = [e.item() for e in topk_results.values][::-1]
    idx_values = [e.item() for e in topk_results.indices][::-1]
    names = [IDX2NAME[i] for i in idx_values]
    short_names = [n if len(n) < max_len else n[:max_len-3]+'...' for n in names]
    ax1.barh(short_names, p_values, align='center', zorder=5)
    ax1.set_xscale('log')
    ax1.grid(True)
    ax1.set_xlabel('Probability (log)')

    unnorm_image = _unnormalize(image_tensor[0])
    ax2.imshow(unnorm_image.numpy().transpose((1, 2, 0)))
    ax2.set_xticks([])
    ax2.set_yticks([])

    plt.savefig(save_to, bbox_inches='tight')
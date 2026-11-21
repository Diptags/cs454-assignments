import argparse
import random
from datetime import datetime

import torch
from perlin_noise import PerlinNoise

from util import *

def perlin_image_generator(octaves):
    size = min(octaves*5, 224)
    noise = PerlinNoise(octaves=octaves, seed=1)
    pic = [[noise([i/size, j/size]) for j in range(size)] for i in range(size)]
    pic_tensor = TRANSFORM(torch.Tensor(pic).unsqueeze(0).repeat(3, 1, 1)).unsqueeze(0)
    pic_tensor = (pic_tensor - pic_tensor.min())/(pic_tensor.max()-pic_tensor.min())
    return pic_tensor

def genotype2phenotype(genes):
    noise = get_random_noise()*0
    for octave, magnitude in genes:
        noise += magnitude * perlin_image_generator(octave)
    return noise

def add_noise_to_image(noise, image):
    noise = noise.clip(0, 1)
    actual_white = torch.Tensor([[[2.2489]], [[2.428]], [[2.64]]]).repeat(1, 224, 224).unsqueeze(0)
    return (1-noise)*image + noise*actual_white

def adv_func(image_path):
    raise NotImplementedError

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Perlin Adversarial Evolver')
    parser.add_argument('-f', '--filename', required=True, default='images/knot.jpg')
    args = parser.parse_args()
    
    noise_genotype = adv_func(args.filename)
    noise_img = genotype2phenotype(noise_genotype)
    org_img = get_image_tensor_from(args.filename)
    adv_image = add_noise_to_image(noise_img, org_img)
    get_viz_for(adv_image)

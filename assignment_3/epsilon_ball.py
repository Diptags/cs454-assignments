from util import *
import argparse

def make_Linf_adversarial_from(image_path):
    raise NotImplementedError

def make_L2_adversarial_from(image_path):
    raise NotImplementedError

def make_L0_adversarial_from(image_path):
    raise NotImplementedError

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='L-norm Adversarial Evolver')
    parser.add_argument('-f', '--filename', required=True, default='images/knot.jpg')
    parser.add_argument('-n', '--norm', type=str, required=True, default='inf')
    args = parser.parse_args()

    if args.norm.lower() == 'inf':
        adv_func = make_Linf_adversarial_from
    elif args.norm == '2':
        adv_func = make_L2_adversarial_from
    elif args.norm == '0':
        adv_func = make_L0_adversarial_from
    else:
        raise ValueError(f'Unknown norm type {args.norm}')
    
    adv_image = adv_func(args.filename)
    get_viz_for(adv_image)

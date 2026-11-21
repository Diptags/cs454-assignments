import argparse
import os
import time

from epsilon_ball import make_Linf_adversarial_from, make_L2_adversarial_from, make_L0_adversarial_from
from perlin_fog import adv_func, genotype2phenotype, add_noise_to_image
from util import *

LARGE_NUMBER = 999999999

def grade_function(image_path:str, label:str):
    '''Returns a CSV-readable line containing distance metric and actual distance.'''
    org_tensor = get_image_tensor_from(image_path)
    img_name = os.path.basename(image_path)
    try:
        if label == 'Linf':
            new_tensor = make_Linf_adversarial_from(image_path)
            loss = get_Linf_norm_for(new_tensor - org_tensor)
        elif label == 'L2':
            new_tensor = make_L2_adversarial_from(image_path)
            loss = get_L2_norm_for(new_tensor - org_tensor)
        elif label == 'L0':
            new_tensor = make_L0_adversarial_from(image_path)
            loss = get_L0_norm_for(new_tensor - org_tensor)
        elif label == 'Perlin':
            noise_genotype = adv_func(image_path)
            noise_magnitude = [e[1] for e in noise_genotype]
            assert all(e > 0 for e in noise_magnitude), 'All Perlin noise magnitudes should be positive.'
            noise_tensor = genotype2phenotype(noise_genotype)
            new_tensor = add_noise_to_image(noise_tensor, org_tensor)
            loss = sum(noise_magnitude)
        else:
            raise ValueError(f'Unknown loss type {label}')
        
        prob, idx = get_model_most_likely_for(new_tensor)
        assert idx != IMAGES_ORG_PREDS[image_path], 'The class of the new image has not changed.'
        error_str = ''
    except Exception as e:
        error_str = f'"{type(e)}: {str(e)}"'
        loss = LARGE_NUMBER
    return f'{img_name},{label},{loss},{error_str}'

def main(args):
    if args.loss_type == 'all':
        labels = ['Linf', 'L2', 'L0', 'Perlin']
    else:
        labels = [args.loss_type]
    
    if args.image_path == 'all':
        image_paths = IMAGES_ORG_PREDS.keys()
    else:
        image_paths = [args.image_path]

    result_str = ''
    for image_path in image_paths:
        for label in labels:
            s = time.time()
            grade_result = grade_function(image_path, label)
            result_str += f'{args.name},{time.time()-s:.2f},{grade_result}\n'
    
    print('====REPORT====')
    print('id,time,image,loss,score,error')
    print(result_str)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Coursework Grading Simulator'
    )
    parser.add_argument('-n', '--name', default='default')
    parser.add_argument('-l', '--loss_type', default='all')
    parser.add_argument('-i', '--image_path', default='all')
    args = parser.parse_args()

    main(args)
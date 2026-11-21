# CS454 Assignment: Genetic Algorithm for Adversarial Examples

As described in the class, adversarial examples are a manifestation of metamorphic testing on machine learning modules. This coursework aims to help you generate adversarial examples of your own, both of the $\epsilon$-ball mathematical adversarial example variety and the 'realistic' adversarial examples perhaps more important to practical applications (like adding fog).

## Aim

In the first part of this coursework, you will be loosely recreating results from the earlier papers that made adversarial examples, but instead of using their gradient-based optimization schema, you will use a genetic algorithm to find vectors that cause misclassification.

 * [Goodfellow et al., _Explaining and Harnessing Adversarial Examples_](https://arxiv.org/abs/1412.6572)
 * [Carlini and Wagner, _Towards Evaluating the Robustness of Neural Networks_](https://arxiv.org/abs/1608.04644)
 * [Su et al., _One pixel attack for fooling deep neural networks_](https://arxiv.org/abs/1710.08864)

In the second part, you will implement "realistic" perturbations that are nonetheless scalable. The effects are similar to the following paper:

 * [Tian et al., _DeepTest: Automated Testing of Deep-Neural-Network-driven Autonomous Cars_](https://arxiv.org/abs/1708.08559)

Specifically, in this coursework, you will be generating adversarial examples based on the 'seed' images provided in the `images/` directory. In this assignment, an "adversarial example" is defined as an image which is modified from the original image such that the original (correct) label is no longer the most likely prediction according to the model. See the example results in `images/examples/`.

## Setup

 * Use Python 3.11 or later.
 * Install PyTorch (2.15.0) and torchvision (0.29.2), then the remaining dependencies:
   ```bash
   pip install -r torch_cpu.txt
   pip install -r requirements.txt
   ```
   * ⚠️ For maximum accessibility, `torch_cpu.txt` installs the CPU-only build of PyTorch. If you have an NVIDIA GPU on Linux or Windows, you may use `torch_gpu.txt` instead, which may help your code run faster. The instructions for installing other variants of PyTorch can be found [here](https://pytorch.org/get-started/locally/).
   * ⚠️ PyTorch no longer publishes builds for Intel (x86_64) Macs. If you are on an Intel Mac, please use a different machine (e.g., a lab machine, a Linux VM, or Google Colab).
   * The first run downloads the pretrained VGG-11 weights (about 500MB).

## Genetic Algorithm

### $\epsilon$-ball Adversarial Examples

In the first task, you will implement genetic algorithms that find adversarial examples (more specifically, adversarial noise) that differ from the original image by a maximum of $\epsilon$. The specific targets are:

 * The $L_\infty$ norm, which measures the _maximum_ difference between the original pixels and the adversarial example. A good $L_\infty$ adversarial example will change almost all the pixels by a small value.
 * The $L_2$ norm, which measures the Euclidean distance between the original image and the adversarial image.
 * The $L_0$ norm (actually mathematically not a norm) measures the number of pixels which changed between the original and adversarial image. A good $L_0$ adversarial example would change a few pixels by a large amount.

Implement a GA for each norm, including a fitness function, in `epsilon_ball.py`. Your task is to generate noise that can be added to the image tensor, so that the neural network (VGG-11) will mispredict the image. The norms themselves are implemented in `util.py`, so you may make use of them. (The observant of you will notice that e.g. $L_0$ norm does not actually measure the number of pixels changed, but on the number of tensor elements changed. This is intended; use the `get_pixel_norm_for` function if you want to play with that, although it will not be used for evaluation.) Using the `get_viz_for` function, you may generate an image which visualizes the top ten likely classes and their probabilities according to the VGG-11 model. Look to the `images/init_viz/` directory for visualization examples for the original images.

For the first task, you will be evaluated based on the smallest $\epsilon$ you could achieve for each of the five images for each of the three norms (15 numbers in total). Use a table to report your results.

### Bonus: "Foggy" Adversarial Examples

In the second task, instead of generating meaningless noise, you will use [Perlin noise](https://en.wikipedia.org/wiki/Perlin_noise) to implement "fog" that in turn induces DNN misprediction. In this scenario, instead of directly manipulating the noise vectors as you did in the first task, you are to evolve the parameters of Perlin noise to induce misclassification. An example of combining Perlin noise is provided [on the Perlin noise package webpage](https://pypi.org/project/perlin-noise/), also presented here for completeness:

```python
import matplotlib.pyplot as plt
from perlin_noise import PerlinNoise

noise1 = PerlinNoise(octaves=3)
noise2 = PerlinNoise(octaves=6)
noise3 = PerlinNoise(octaves=12)
noise4 = PerlinNoise(octaves=24)

xpix, ypix = 100, 100
pic = []
for i in range(xpix):
    row = []
    for j in range(ypix):
        noise_val = noise1([i/xpix, j/ypix])
        noise_val += 0.5 * noise2([i/xpix, j/ypix])
        noise_val += 0.25 * noise3([i/xpix, j/ypix])
        noise_val += 0.125 * noise4([i/xpix, j/ypix])

        row.append(noise_val)
    pic.append(row)
```

For this task, evolve a list of [(octave, magnitude)] parameters that will be added as above; for example, the representation for the noise in the code above would be `[(3, 1), (6, 0.5), (12, 0.25), (24, 0.125)]`. 

As we still need some sort of metric to judge how much obscuring the "fog" is doing, the sum of the magnitude parameters is used to evaluate the degree of obscuring. Again, having a "weaker" fog according to this metric and still causing misclassification is the goal.

### Deliverables and Report

Based on the contents of your `epsilon_ball.py` / `perlin_fog.py`, we will evaluate the $\epsilon$ that your solution achieves on a set of held-out images. Make sure not to change the function interface, as your solution will be automatically graded. Your solution is expected to take **three** minutes at maximum to generate an adversarial example for `epsilon_ball.py` on CPU, and **five** minutes for `perlin_fog.py`. See `grade.py` for an example of a grading script. (The actual grading will use the same logic, but will use additional safeguards to prevent tampering of key functions such as the loss function when grading, as well as to deal with timeouts.)

Finally, include a PDF report of your efforts when you are done. In the report, include the following: 

 1. A description of the strategies that you used for each of the norms, and whether you felt compelled to use different strategies in each case
 2. The 15 images that you evolved for each seed image and norm, their $\epsilon$ distance from the original image, the 15 raw tensor files generated by `get_viz_for` (their filenames start with `raw_`), and the classification results as visualized by the `get_viz_for` function. For each seed image and norm, include the $\epsilon$ distance from the original image in the report.
 3. (Bonus) The evolved Perlin noise expression, and a visualization of the best foggy images generated.

#### Naming the raw tensor files

The raw tensor files are graded automatically, so name each one `raw_<image>_<norm>.png`, where `<image>` is one of `knot`, `mink`, `orca`, `sock` or `tank`, and `<norm>` is one of `linf`, `l2` or `l0`. For example, `raw_knot_linf.png` or `raw_tank_l0.png`. The easiest way is to pass the name without the `raw_` prefix to `get_viz_for`:

```python
get_viz_for(adv_image, save_to='knot_linf.png')   # writes knot_linf.png and raw_knot_linf.png
```

Use a plain filename for `save_to`, not a path with a folder, because `raw_` is added to the front of whatever you pass. Submit exactly one raw file for each of the 15 seed image and norm combinations. A file whose name does not follow this pattern may not be graded.

If you attempt the bonus, also submit, for each seed image:

 * `raw_<image>_perlin.png`, the raw tensor of your best foggy image, saved with `get_viz_for(adv_image, save_to='<image>_perlin.png')`.
 * `<image>_perlin.json`, the evolved genotype as a JSON list of `[octave, magnitude]` pairs. The fog is graded from this file: it is regenerated with `genotype2phenotype`, its $\epsilon$ is the sum of the magnitudes, and the result must still cause misclassification. Octaves must be positive integers, magnitudes must be positive numbers, and at most 50 pairs are allowed. For example:

   ```python
   import json
   noise_genotype = adv_func('images/knot.jpg')   # e.g. [(3, 0.4), (12, 0.1)]
   with open('knot_perlin.json', 'w') as f:
       json.dump([[int(o), float(m)] for o, m in noise_genotype], f)
   ```
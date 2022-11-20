# Inverting Gradients Attack
A privacy-related attack that attempts to restore input data from model's gradient information.

## Data Poisoning Attack with Inverting Gradients Methodology
### Code organization:
We will use Jupyter notebook to wrap up the attack (i.e., build attack interface) just like all the other examples under `breaching/examples`.

Here is the notebook: [Data_Poisoning_Attack_via_IG.ipynb](https://github.com/Kexin6/let_the_neurons_die/tree/main/inverting_gradients/breaching/Data_Poisoning_Attack_via_IG.ipynb)

All the changes to be made to the breaching framework will be under [breaching/breaching](https://github.com/Kexin6/let_the_neurons_die/tree/main/inverting_gradients/breaching/breaching). 


## Breaching library
This is a PyTorch framwork for privacy attacks in federated learning. The original repository can be found: https://github.com/JonasGeiping/breaching.

It is noteworthy that the `breaching` framework itself is an implementation of `inverting gradients` attack which tries to infringe privacy by recovering input data from the gradients. The code repository can be found: https://github.com/JonasGeiping/invertinggradients. 

There are mainly two components in the folder: `breaching` and `examples`. `breaching` contains the source code of the library and `examples` are the successful attacks that were implemented using this framework. 

For more high-level information on the PyTorch `breaching` framework, please check [HERE](https://github.com/JonasGeiping/breaching/blob/main/README.md).

More implementation details are as follows:

### Attacks
This is the most important folder to look at for our extension on `inverting gradients attack`. This module allows attackers to extend upon existing attacks. For high-level information, please check [here](https://github.com/JonasGeiping/breaching/blob/main/breaching/attacks/README.md).

Some important highlights:

#### [optimization _based_attack.py](https://github.com/JonasGeiping/breaching/blob/main/breaching/attacks/optimization_based_attack.py):
It is recommended that any new attacks should inherit from this attack.
* reconstruct function:
  * self.prepare_data (defined in `base_attack.py`): gradient information can be stored in shared_data variable
  * Come up with reconstruction candidates
  * pick the optimal reconstruction

#### [auxiliaries](https://github.com/JonasGeiping/breaching/tree/main/breaching/attacks/auxiliaries):
Helper functions for the attacks.

It is noteworthy to look at `objectives.py`:
This file includes various objective functions (i.e., distance heuristics) such as Euclidean and L1 distance that measures distance of two gradient vectors.

### Cases
This folder contains the use cases for each attack under `breaching/examples/`. The main difference between `cases` and `attacks` is that `attacks` are the actual implementation that reconstructs user data, and `cases` is like a wrapper that sets up the attack scenario (i.e., threat model).

It is noteworthy to look at [users.py](https://github.com/JonasGeiping/breaching/blob/main/breaching/cases/users.py):
This file contains a function `compute_local_updates` that updates the `shared_data` variable which is a list containing `shared_gradient` information we concern.

#### Models:
This subfolder under `breaching/cases` contains neural networks that we craft the attack on, such as ResNet, VGG, etc.

It is noteworthy to look at [model_preparation.py](https://github.com/JonasGeiping/breaching/blob/main/breaching/cases/models/model_preparation.py):
This is the file that allows us to construct the model and initialize weights.

### Analysis
This folder implements metrics that examine the quality of reconstructed data, e.g., SSIM to measure similarity. For our use case, we may not need this far, since all we really need is the reconstructed data.

## Remark
To understand the workflow. This example is the closest to our objective: [Inverting Gradients - Optimization-based Attack](https://github.com/JonasGeiping/breaching/blob/main/examples/Inverting%20Gradients%20-%20Optimization-based%20Attack%20-%20ResNet18%20on%20ImageNet.ipynb)

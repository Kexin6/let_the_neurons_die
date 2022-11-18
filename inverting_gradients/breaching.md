# Inverting Gradients Attack
A privacy-related attack that attempts to restore input data from model's gradient information.

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

### Models

### Cases

### Analysis

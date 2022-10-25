# let_the_neurons_die

## Objective
Generate an adversarial/surrogate network that contains dead neurons 

## Step 1
Get a set to target weights Wt based on weight-based Soft Knockout/Shift Attack, from the initial weights that prevents model convergence and leads to poor utility.

## Step 2
Goal: minimize the distance between target weights and current weights by:

### Option 1: Data Ordering Attack
### Option 2: Inverting Gradients Attack

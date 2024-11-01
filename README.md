# Adaptive Node Positioning in Transport Networks
<p align="center">
<img width="400" height="320" src="https://github.com/user-attachments/assets/613b931f-ab3e-43c0-9cc0-ebc211f7048f">
</p>

This repository contains the code used in the paper "Adaptive Node Positioning in Biological Transport Networks".

## Setup
```
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements
python3 -m pip install jax[cuda12] # In case of GPU available
```

## Run
To run the code, just type
```
python optimize.py
```
This will create a folder with the current time inside `runs/,` e.g., `runs/2024-05-06_112929`, which it will make if it doesn't exist.
There, it will save the arrays of the network and some results on `npz` files inside the `arrays/` directory.

Some options (`python optimize.py --help`)
```
  -h, --help            show this help message and exit
  --gamma GAMMA         The exponent of the power dissipation.
  --n_nodes N_NODES     The number of nodes in the network.
  --learning_rate LEARNING_RATE, -lr LEARNING_RATE
                        The learning rate of the optimizer.
  --init_noise INIT_NOISE, -in INIT_NOISE
                        Initial noise to the positions
  --num_iters NUM_ITERS
                        The number of iterations.
  --rtol RTOL           Relative tolerance for convergence.
  --atol ATOL           Absolute tolerance for convergence.
  --beta BETA           The β parameter of the leaf.
  --theta THETA         The rotation of the leaf.
  --save_interval SAVE_INTERVAL
                        Interval to save the network.
  --out OUT             Output folder to save the results.
  --name NAME           Name of the run.
  --seed SEED           Seed for the random number generator.
```

To visualize the network one can use (PS: You need to have ffmpeg installed)
```
python visual.py
```

## LICENSE

MIT License

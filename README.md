# FLCSP
Code for paper "Fusion of Latent Categorical Prediction and Sequential Prediction for Session-based Recommendation (2021, Information Sciences)"

## How to Use

You need to run the file `run_randomwalk.py` first to generate random walk distribution.

At last, you can run the file `main.py` to train the model.

Take Yoochoose 1/64 dataset as example:
```
python run_randomwalk.py --dataset yoochoose1_64 --data_path data --anchor_num 1500 --alpha 0.5 --rw w
python main.py --dataset yoochoose1_64 --anchor_num 1500 --alpha 0.5 --ksize 3 --rw r --batch_size 512
```

## Requirements

- Python3
- pytorch==1.0.1

There is a version issue of torch, the version of torch must be 1.0.1. If it's a higher version of the torch, loss will be 'nan'. We have not solved this problem.

## Citation
Please cite our paper if you use the code!


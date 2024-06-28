# Introduction

The input of the model includes the features used by the tree model, along with the user historical sequences. We adopt the officially provided pre-trained vectors, reduce them to 64 dimensions, initialize the article embeddings and freeze them during training and inferring. The sequences are enhanced by the history file and the behavior file for the validation and test data. Due to the trade-off between training cost and performance, we set sequence length to 200, and adopt the target-aware mechanism similar to DIN to capture the user long-term preferences. The item of historical sequences is represented by various auxiliary information, the bucketing number of read time, and the bucketing number of scroll percentage. Then, the sparse and dense feature representation addition to the target-aware sequence representation of behavior are learned by the feature importance component similar to MaskNet to learn the feature importance weights at the instance level. Finally, multiple layers of MLP are used to estimate the click-through rate.

We conducted comprehensive ablation experiments and demonstrated the effectiveness of the behavior sequence, pre-trained vectors, feature importance components, and fine-grained listwise feature design. In the end, our single nn model achieved a score of 0.873 on the leaderboard, which ranked second among all teams' scores. After integrating with the tree model, the final score reached 0.8808. You can refer to our report for more details.

| Model          | Offline Score | Online Score |
|----------------|---------------|--------------|
| NN            | 0.856         | 0.873        |



# Dependencies

- python 3.9
- pytorch 1.10
- FuxiCTR 2.2.3


# Install Guidence

- conda create --name py39 python=3.9
- conda activate py39
- pip install -r requirements.txt
- conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch



# Running
please change the executing path to this nn folder and run the following scripts.


generate training/validation/testing samples: 
- `sh gen_sample.sh`

running training/infering process, we support multi-gpu running, you can pass the gpus parameter
- `sh run.sh`

You can also run the process separately, train first, then pass the pretrained path to get submission result and cv result. e.g.,
- `python run.py --gpus 0 --mode train`
- `python run.py --gpus 0 --mode submit --model_dir checkpoints/V1_6e00181c_1719465783`
- `python run.py --gpus 0 --mode cv --model_dir checkpoints/V1_6e00181c_1719465783`

# References

https://recsys.eb.dk/

# RecSys'24 Competition - BlackPearl Team Solution
### Introduction
The Ekstra Bladet RecSys Challenge aims to predict which article a user will click on from a list of articles that were seen during a specific impression. Utilizing the user's click history, session details (like time and device used), and personal metadata (including gender and age), along with a list of candidate news articles listed in an impression log, the challenge's objective is to rank the candidate articles based on the user's personal preferences. This involves developing models that encapsulate both the users and the articles through their content and the users' interests. The models are to estimate the likelihood of a user clicking on each article by evaluating the compatibility between the article's content and the user's preferences. The articles are ranked based on these likelihood scores, and the precision of these rankings is measured against the actual selections made by users.

### Brief Solution
Through data analysis, we discovered that the length of the news list displayed in the exposure sequence and the hourly granularity data of news exposure within the exposure interval on the same day are very strong features. We then meticulously mined and derived a series of related features, combined with features related to users' historical behaviors, to construct tree models and neural network models for training and fitting the target. Our solution is a simple weighted combination of two tree models (CatBoost's [pair rank loss](https://catboost.ai/en/docs/concepts/loss-functions-ranking#PairLogitPairwise) and CatBoost's [query loss](https://catboost.ai/en/docs/references/querycrossentropy)) and a DIN model, achieving a score of 0.8808 on the online test set. In terms of single model performance, the best single tree model was CatBoost's query loss, with an offline GAUC score of 0.866 and an online score of 0.879. The DIN model had an offline GAUC score of 0.856 and an online score of 0.873.

| Model          | Offline Score | Online Score |
|----------------|---------------|--------------|
| DIN            | 0.856         | 0.873        |
| Cat Pair Rank  | 0.860         | 0.873        |
| Cat Query Loss | 0.866         | 0.879        |
| Blend          | 0.8678        | **0.8808**       |



### Requirements
* Hardware Requirements: 8 x A100 GPUs + 1TB of memory
* Software Requirements:

    ```sh
    pip install -r requirements.txt
    ```

### Data Preparation
Create an "inputs" directory in the working directory, and then create "large" and "vectors" subdirectories within it. First, unzip the five vector zip files and place them in the "vectors" subdirectory. Unzip the "ebnerd_large.zip" file and place it in the "large" directory. Also, unzip the "ebnerd_testset.zip" file and place it in the "large" directory as well.

### Feature Engineer
Switch to the 'code/Feature_Engineer' subdirectory and execute the following command:
```sh
sh FE.sh
```

### Train Tree model 
Switch to the 'code' subdirectory and execute the following command:
```sh
sh train.sh
```
### NN Model
switch to the 'code/nn' subdirectory and execute the following command:
```sh
sh gen_sample.sh
sh run.sh
```

### Blend and Submit
```sh
sh submit.sh
```
If you have any questions, Welcome to contact us.



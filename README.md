# Fake news detection 
==============================

### Description of the project

Following the intention of the [Kaggle](https://www.kaggle.com/competitions/fake-news/overview) competition, we aim to build a system capable of classifying unreliable news articles.

### Framework 

We plan on using the [Huggingface Transformer](https://github.com/huggingface/transformers) framework. We will be using a pre-trained model included in the framework.

### Data description

The [data](https://www.kaggle.com/competitions/fake-news/data) is made available through the competition. This dataset contains the following features for each news article: A unique ID, the title, the author, the textual content, and finally a binary label of 0 or 1 corresponding to “reliable” and unreliable respectively. The model will be trained using DTU HPC.

### Models
We expect to use a transformer model made for Natural Language Processing - specifically a pre-trained BERT. The model used is the base version with 12 layers and 12 self-attention heads. The hidden size is 784.

### Usage
To use this project, you will need to install the required packages listed in requirements.txt. You can then run the scripts in the src directory to process the data, build features, train models, and make predictions. The Makefile includes commands for running these scripts, such as make data, make train, and make predict.

### License
This project is licensed under the terms of the MIT License.


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

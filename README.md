# NeuralLog
Repository for the paper: [Log-based Anomaly Detection Without Log Parsing](https://ieeexplore.ieee.org/document/9678773).

**Abstract**: Software systems often record important runtime information in system logs for troubleshooting purposes. There have been many studies that use log data  to construct machine learning models for detecting system anomalies. Through our empirical study, we find that existing log-based anomaly detection approaches are significantly affected by log parsing errors that are introduced by 1) OOV (out-of-vocabulary) words, and 2) semantic misunderstandings. The log parsing errors could cause the loss of important information for anomaly detection. To address the limitations of existing methods, we propose NeuralLog, a novel log-based anomaly detection approach that does not require log parsing. NeuralLog extracts the semantic meaning of raw log messages and represents them as semantic vectors. These representation vectors are then used to detect anomalies through a Transformer-based classification model, which can capture the contextual information from log sequences. Our experimental results show that the proposed approach can effectively understand the semantic meaning of log messages and achieve accurate anomaly detection results. Overall, NeuralLog achieves F1-scores greater than 0.95 on four public datasets, outperforming the existing approaches.

## Framework
<p align="center"><img src="docs/images/framework.jpg" width="502"><br>An overview of NeuralLog</p>

NeuralLog consists of the following components:
1. **Preprocessing**: Special characters and numbers are removed from log messages.
2. **Neural Representation**: Semantic vectors are extracted from log messages using BERT.
3. **Transformer-based Classification**: A transformer-based classification model containing Positional Encoding and Transformer Encoder is applied to detect anomalies.

[//]: # ([PyTorch version]&#40;https://github.com/LogIntelligence/LogADEmpirical&#41;)
## Requirements
1. Python 3.5 - 3.8
2. tensorflow 2.4
3. transformers
4. tf-models-official 2.4.0
5. scikit-learn
6. pandas
7. numpy
8. gensim
## Demo
- Extract Semantic Vectors

```python
from neurallog import data_loader

log_file = "../data/raw/BGL.log"
emb_dir = "../data/embeddings/BGL"

(x_tr, y_tr), (x_te, y_te) = data_loader.load_Supercomputers(
     log_file, train_ratio=0.8, windows_size=20,
     step_size=5, e_type='bert')
```
- Train/Test Transformer Model

See [notebook](demo/Transformer_based_Classification.ipynb)

- Full demo on the BGL dataset
```shell
$ pip install -r requirements.txt
$ wget https://zenodo.org/record/3227177/files/BGL.tar.gz && tar -xvzf BGL.tar.gz
$ mkdir logs && mv BGL.log logs/.
$ cd demo
$ python NeuralLog.py
```
## Data and Models
Datasets and pre-trained models can be found here: [Data](https://figshare.com/s/6d3c6a83f4828d17be79)
## Results
| Dataset | Metrics | LR | SVM | IM | LogRobust | Log2Vec | NeuralLog |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
|  | Precision | 0.99 | 0.99 | **1.00** | 0.98 | 0.94 | 0.96 |
| HDFS | Recall | 0.92 | 0.94 | 0.88 | **1.00** | 0.94 | **1.00** |
|  | F1-score | 0.96 | 0.96 | 0.94 | **0.99** | 0.94 | 0.98 |
|  | Precision | 0.13 | 0.97 | 0.13 | 0.62 | 0.80 | **0.98** |
| BGL | Recall | 0.93 | 0.30 | 0.30 | 0.96 | **0.98** | **0.98** |
|  | F1-score | 0.23 | 0.46 | 0.18 | 0.75 | 0.88 | **0.98** |
|  | Precision | 0.46 | 0.34 | - | 0.61 | 0.74 | **0.93** |
| Thunderbird | Recall | 0.91 | 0.91 | - | 0.78 | 0.94 | **1.00** |
|  | F1-score | 0.61 | 0.50 | - | 0.68 | 0.84 | **0.96** |
|  | Precision | 0.89 | 0.88 | - | 0.97 | 0.91 | **0.98** |
| Spirit | Recall | 0.96 | **1.00** | - | 0.94 | 0.96 | 0.96 |
|  | F1-score | 0.92 | 0.93 | - | 0.95 | 0.95 | **0.97** |


## Citation
If you find the code and models useful for your research, please cite the following paper:
```
@inproceedings{le2021log,
  title={Log-based anomaly detection without log parsing},
  author={Le, Van-Hoang and Zhang, Hongyu},
  booktitle={2021 36th IEEE/ACM International Conference on Automated Software Engineering (ASE)},
  pages={492--504},
  year={2021},
  organization={IEEE}
}
```

# H3GNN
This is the code for our paper [《Heterogeneous Hyperbolic Hypergraph Neural Network for Friend Recommendation in Location-based Social Networks》](https://dl.acm.org/doi/full/10.1145/3708999), which has been published in TKDD.


#  Data Preparation

We provide the processed datasets for six cities in the `data` folder.

The **Gowalla** dataset comes from [Yong Liu](https://stephenliu0423.github.io/datasets.html). 
We noticed that this page is no longer accessible due to the expiration of his homepage domain sometimes. 
Therefore, we have made the downloaded dataset available in [our Google Drive](https://drive.google.com/drive/folders/1stJucklk9FLeFUFMVATXeHv6aXzc2hL9?usp=sharing).


!!! Importantly, in this work, we have open-sourced the data preprocessing code, using the **Gowalla** dataset as an example.
This includes instructions on how to construct the heterogeneous hypergraph and the heterogeneous multigraph in HMGCL. Please refer to the `data_preprocess` folder.



# Citation
If you find this work helpful, please consider citing our paper:
```bibtex
@article{li2025heterogeneous,
author = {Li, Yongkang and Fan, Zipei and Song, Xuan},
title = {Heterogeneous Hyperbolic Hypergraph Neural Network for Friend Recommendation in Location-based Social Networks},
year = {2025},
issue_date = {April 2025},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {19},
number = {3},
issn = {1556-4681},
url = {https://doi.org/10.1145/3708999},
doi = {10.1145/3708999},
journal={ACM Transactions on Knowledge Discovery from Data},
month = feb,
articleno = {57},
numpages = {29},
}
```

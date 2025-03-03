# H3GNN
This is the code for our paper [《Heterogeneous Hyperbolic Hypergraph Neural Network for Friend Recommendation in Location-based Social Networks》](https://dl.acm.org/doi/full/10.1145/3708999), which has been published in TKDD.


#  Data Preparation

We provide the processed datasets for six cities in the `data` folder.

The **Gowalla** dataset comes from [Yong Liu](https://stephenliu0423.github.io/datasets.html). 
We noticed that this page is no longer accessible due to the expiration of his homepage domain sometimes. 
Therefore, we have made the downloaded dataset available in [our Google Drive](https://drive.google.com/drive/folders/1stJucklk9FLeFUFMVATXeHv6aXzc2hL9?usp=sharing).


!!! Importantly, in this work, we have open-sourced the data preprocessing code, using the **Gowalla** dataset as an example.
This includes instructions on how to construct the heterogeneous hypergraph and the heterogeneous multigraph in HMGCL. Please refer to the `data_preprocess` folder.

### Data Processing Steps:

1. **Download Raw Data**  
   Download the raw data into the `data/yongliu_gowalla_data` folder.

2. **Split Data by City**  
   Run `data_preprocess/split_data_into_city.ipynb` to divide the global dataset into individual cities.

3. **Build Hypergraph Data Format**  
   Run `data_preprocess/build_hypergraph.ipynb` to generate the hypergraph data format for each city.

4. **Build Multigraph Data Format**  
   Run `data_preprocess/build_multigraph.ipynb` to generate the multigraph data format for each city.

**Note:**  
The hypergraph and multigraph data formats obtained here can be directly used in my previous [**HMGCL**](https://github.com/liyongkang123/HMGCL) and [**HHGNN**](https://github.com/liyongkang123/HHGNN) work, serving as baselines.


# Requirements
- Python 3.10, Pytorch, DGL, PyG etc.
- GPU is recommended with 48GB memory or more.

# Usage
To run the code, you can use the following command:



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

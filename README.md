# Wrongdoing-Monitor
This is an example implementation (on Darknet dataset) of our work in Wrongdoing-Monitor, and we run our code on Windows 10. More experimental details can be refered in our paper.

## Requirements

* python 3.6+

* pytorch 1.4+

* tensorflow 2.4+

* numpy

* tqdm

* scikit-learn

* pandas

* xgboost

## Descriptions of  The Selected Properties

The detail of the selected properties are shown in Properties_Detail.pdf.

## Using The Code

At present, our method consists of several python script files. You need to execute the corresponding Python files in turn. 
In the first step, we  provide a preprocessing process for Darknet dataset。
In the second step, we provide two representative network representation learning algorithms (GTN and MARINE) to obtain the property representations.
Note that some network representation learning algorithms need to input the graph whoes nodes are numbered from 0. It may be necessary to simply change the node_id before running the second step.
In one execution, you only need to select the version corresponding to one algorithm.


## Datasets

The datasets used in our paper can be found by following this link: 

* Kyoto: [http://www.takakura.com/Kyoto_data](http://www.takakura.com/Kyoto_data)

* Darknet: [https://www.unb.ca/cic/datasets/darknet2020.html](https://www.unb.ca/cic/datasets/darknet2020.html)

* Gowalla: [http://snap.stanford.edu/data/loc-Gowalla.html](http://snap.stanford.edu/data/loc-Gowalla.html)

* CICDDoS: [https://www.unb.ca/cic/datasets/ddos-2019.html](https://www.unb.ca/cic/datasets/ddos-2019.html)

## Citation
Waiting for update.


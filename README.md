## MaskPlace: Fast Chip Placement via Reinforced  Visual Representation Learning

A new chip placement method based on visual representation learning. 

###  Publication
Lai, Yao, Yao Mu, and Ping Luo. "Maskplace: Fast chip placement via reinforced visual representation learning." Advances in Neural Information Processing Systems 35 (2022): 24019-24030. (NeurIPS 2022, **spotlight**)

[paper](https://arxiv.org/pdf/2211.13382.pdf)


### Usage

You can start easily by using the following script.

```
cd maskplace
python PPO2.py
```

### Parameter

- **gamma** Decay factor.
- **seed** Random seed.
- **disable_tqdm** Whether to disable the progress bar.
- **lr** Learning rate.
- **log-interval** Interval between training status logs.
- **pnm** Number of place modules for each placement trajectory. 
- **benchmark** Circuit benchmark.
- **soft_coefficient** Whether to constriant the actions based on the wiremask.
- **batch_size** Batch size.
- **is_test** Testing mode based on the trained agent.
- **save_fig** Whether to save placement figures.


### Benchmark
The repo has provided the benchmark *adaptec1* and *ariane*. For other benchmarks, you can download them by the following the link:

http://www.cerc.utexas.edu/~zixuan/ispd2005dp.tar.xz


### Dependency
- [Python](https://www.python.org/) >= 3.9

- [Pytorch](https://pytorch.org/) >= 1.10

  - Other versions may also work, but not tested

- [gym](https://www.gymlibrary.dev/index.html) >= 0.21.0
- [matplotlib](https://matplotlib.org/) >= 3.7.1
- [tqdm](https://tqdm.github.io/)
- [protobuf](https://pypi.org/project/protobuf/) (for benchmark *ariane*)

### Citation
If you find our paper/code useful in your research, please cite

```
@article{lai2022maskplace,
  title={Maskplace: Fast chip placement via reinforced visual representation learning},
  author={Lai, Yao and Mu, Yao and Luo, Ping},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={24019--24030},
  year={2022}
}
```

 
### The placement process animation

Benchmark: Bigblue3


|Placement| Pos Mask t | Wire Mask t |
|---|---|---|
|<img src="imgs/place.gif" width="250">|<img src="imgs/pos_img.gif" width="250">|<img src="imgs/net_img.gif" width="250">|
|<center><b> View Mask </b></center>| <center><b> Pos Mask t+1 </b></center>| <center><b> Wire Mask t+1 </b></center>|
|<img src="imgs/view_img.gif" width=250>|<img src="imgs/pos_img_next.gif" width=250> | <img src="imgs/net_img_next.gif" width=250>|


### Standard Cell Placement

Fix macros and use DREAMPlace (classic optimization-based method) to place standard cells.

|<center>adaptec2</center>| <center>adaptec4 </center>| <center> bigblue3 </center>|
|---|---|---|
|<img src="imgs/stdcell_a2.gif" width="250">|<img src="imgs/stdcell_a4.gif" width="250">|<img src="imgs/stdcell_b3.gif" width="250">|

### Full Benchmark demonstration


|Benchmark| DREAMPlace | Graph | DeepPR | MaskPlace |
|---|---|---|---|---|
|adaptec1|<img src="imgs/dreamplace/adaptec1.png" width="160">|<img src="imgs/graph/adaptec1.png" width="160">|<img src="imgs/deeppr/adaptec1.png" width="160">|<img src="imgs/maskplace/adaptec1.png" width="160">|
|HPWL (10<sup>5</sup>)|17.94|26.05|21.36|<strong>6.57</strong>|
|Wirel (10<sup>5</sup>)|19.24|28.54|25.64|<strong>7.36</strong>|
|Overlap|0.34%|1.89%|32.03%|<strong>0</strong>|
|adaptec2|<img src="imgs/dreamplace/adaptec2.png" width="160">|<img src="imgs/graph/adaptec2.png" width="160"> | <img src="imgs/deeppr/adaptec2.png" width="160">|<img src="imgs/maskplace/adaptec2.png" width="160">|
|HPWL (10<sup>5</sup>)|135.32|359.35|197.13|<strong>79.98</strong>|
|Wirel (10<sup>5</sup>)|140.91|381.64|205.78|<strong>83.59</strong>|
|Overlap|0.16%|1.54%|49.10%|<strong>0</strong>|
|adaptec3|<img src="imgs/dreamplace/adaptec3.png" width="160">|<img src="imgs/graph/adaptec3.png" width="160"> | <img src="imgs/deeppr/adaptec3.png" width="160">|<img src="imgs/maskplace/adaptec3.png" width="160">|
|HPWL (10<sup>5</sup>)|112.28|392.66|340.29|<strong>79.33</strong>|
|Wirel (10<sup>5</sup>)|119.23|409.37|372.02|<strong>85.28</strong>|
|Overlap|<strong>0</strong>|1.26%|29.10%|<strong>0</strong>|
|adaptec4|<img src="imgs/dreamplace/adaptec4.png" width="160">|<img src="imgs/graph/adaptec4.png" width="160"> | <img src="imgs/deeppr/adaptec4.png" width="160">|<img src="imgs/maskplace/adaptec4.png" width="160">|
|HPWL (10<sup>5</sup>)|<strong>37.77</strong>|152.89|243.12|75.75|
|Wirel (10<sup>5</sup>)|<strong>47.90</strong>|179.43|290.14|88.87|
|Overlap|<strong>0</strong>|7.43%|19.29%|<strong>0</strong>|
|bigblue1|<img src="imgs/dreamplace/bigblue1.png" width="160">|<img src="imgs/graph/bigblue1.png" width="160"> | <img src="imgs/deeppr/bigblue1.png" width="160">|<img src="imgs/maskplace/bigblue1.png" width="160">|
|HPWL (10<sup>5</sup>)|2.50|8.32|20.49|<strong>2.42</strong>|
|Wirel (10<sup>5</sup>)|3.41|10.00|25.68|<strong>3.14</strong>|
|Overlap|<strong>0</strong>|2.48%|9.33%|<strong>0</strong>|
|bigblue3|<img src="imgs/dreamplace/bigblue3.png" width="160">|<img src="imgs/graph/bigblue3.png" width="160"> | <img src="imgs/deeppr/bigblue3.png" width="160">|<img src="imgs/maskplace/bigblue3.png" width="160">|
|HPWL (10<sup>5</sup>)|104.05|345.49|439.09|<strong>82.61</strong>|
|Wirel (10<sup>5</sup>)|107.58|373.33|517.86|<strong>88.51</strong>|
|Overlap|8.06%|0.80%|85.23%|<strong>0</strong>|
|ariane|<img src="imgs/dreamplace/ariane.png" width="160">|<img src="imgs/graph/ariane.png" width="160"> | <img src="imgs/deeppr/ariane.png" width="160">|<img src="imgs/maskplace/ariane.png" width="160">|
|HPWL (10<sup>5</sup>)|20.30|16.83|51.43|<strong>14.86</strong>|
|Wirel (10<sup>5</sup>)|21.72|18.48|55.85|<strong>15.80</strong>|
|Overlap|<strong>0.78%</strong>|3.72%|38.91%|1.94%|




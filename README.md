# OFPNet

![workflow](https://github.com/YoushaaMurhij/OFPNet/actions/workflows/main.yml/badge.svg) [![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause) <br>
**OFPNet** (Occupancy and flow predictive network) is developed for end-to-end prediction of occupancy map and flow
using reccurent blocks, additional convolutional heads, etc. <br>

**OFPNet** is a baseline solution for Waymo Occupancy and Flow Prediction 

<img src="./assets/complete_scene.gif" alt="complete_scene" align="left" width="350" /> <img src="./assets/observed_occupancy_rgb.gif" alt="observed_occupancy_rgb" align="middle" width="350"/>
<img src="./assets/occluded_occupancy_rgb.gif" alt="occluded_occupancy_rgb" align="left" width="350"/> <img src="./assets/flow_rgb.gif" alt="flow_rgb" align="middle" width="350"/>

## Main Metrics

|     Metrics    | Observed Occupancy | Occluded Occupancy  | 	Flow	 | Flow-Grounded Occupancy | 
| :-----------:  | :-----------:      | :-----------:       |:----------:| :-----------:           |
     
| Model          | AUC           | 	Soft IoU     | AUC           |	Soft IoU     |	EPE          | AUC           | 	Soft IoU    |
| :------------: |:------------: | :-----------: | :-----------: |:------------: | :-----------: | :-----------: |:-----------: |
| UNet_LSTM	     | 0.6559        | 0.4007        | 0.1227	     | 0.0261	     | 20.5876       | 0.5768	     | 0.4280       |
| UNet_LSTM_Head | 0.6517	     | 0.3859	     | 0.1199	     | 0.0225	     | 20.1838	     | 0.5840	     | 0.4119       |
| unext	         | 0.6485	     | 0.3580	     | 0.0376	     | 0.0084	     | 21.6873	     | 0.5598	     | 0.4098       |
| unext_head	 | 0.7119	     | 0.4257	     | 0.1451	     | 0.0309	     | 21.6873	     | 0.5691	     | 0.4243       |


## Basic Installation

### Docker container:
Using nvidia-docker with cuda-11.3, Pytorch  
```bash
cd path/to/workspace
git clone https://github.com/YoushaaMurhij/Occ_Flow_Pred.git
cd Occ_Flow_Pred/docker
./build.sh
cd ..
./docker/start.sh
./docker/into.sh

```
### Conda environment (not recommended):
```bash
conda create --name occ_flow 
conda activate occ_flow
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
git clone https://github.com/YoushaaMurhij/Occ_Flow_Pred.git
cd Occ_Flow_Pred
pip install -r requirements.txt

# add Occ_Flow_Pred to PYTHONPATH by adding the following line to ~/.bashrc (change the path accordingly)
export PYTHONPATH="${PYTHONPATH}:/path/to/Occ_Flow_Pred/"
```
## TODOs:
- change data input
- add more aux losses


## Contribution:
Questions, suggestions and pull-requests are welcome! <br>
Feel free to open an issue or a pull-request :relaxed: <br>

## Contacts:

Youshaa Murhij  :mailbox_with_mail: yosha[dot]morheg[at]phystech[dot]edu <br>
Dmitry Yudin    :mailbox_with_mail: yudin[dot]da[at]mipt[dot]ru          <br>
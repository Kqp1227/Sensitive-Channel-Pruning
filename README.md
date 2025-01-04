# Sensitive-Channel-Pruning

This is the official repository of the following paper: ["**Achieving Fairness Through Channel Pruning for Dermatological Disease Diagnosis**"](https://arxiv.org/abs/2405.08681) in Proc. of Medical Image Computing and Computer Assisted Interventions (MICCAI), 2024 (early accept, acceptance rate 11%)

<img width="1714" alt="miccai24" src="https://github.com/Kqp1227/Sensitive-Channel-Pruning/assets/104506575/47c86367-b0e2-4d0c-aa51-f1d77e46b5de">

### Environment setup
```
pip install -r requirements.txt
```

### 1. Calculate sensitive channels
To get sensitive channels, run: 
```
python SNNL.py
```
remember to input your own pre-trained CNN model at ```line #379```.

### 2. Prune and fine-tune the model
Using ```train_cnn.py``` to train and fine-tune your own model to achieve better fairness.
```
python train_cnn.py
```
remember to input your pre-trained CNN model at ```line #208``` and the indexes of sensitive channels at ```line #212```, which can be obtained from the last step where sensitive channels were calculated.

### 3. Retrain
You can repeat ```step 1-2``` for several times (until the stopping criteria are met) to achieve better results.


### Citation
If it is helpful to you, please cite our work:
```
@inproceedings{kong2024achieving,
  title={Achieving Fairness Through Channel Pruning for Dermatological Disease Diagnosis},
  author={Kong, Qingpeng and Chiu, Ching-Hao and Zeng, Dewen and Chen, Yu-Jen and Ho, Tsung-Yi and Hu, Jingtong and Shi, Yiyu},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={24--34},
  year={2024},
  organization={Springer}
}
```

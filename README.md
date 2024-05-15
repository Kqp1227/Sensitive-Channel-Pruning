# Sensitive-Channel-Pruning

This is the official repository of the following paper: "**Achieving Fairness Through Channel Pruning for Dermatological Disease Diagnosis**" in Proc. of Medical Image Computing and Computer Assisted Interventions (MICCAI), 2024 (early accept, acceptance rate 11%)

### Environment setup
```
pip install -r requirements.txt
```

### Calculate sensitive channels
To get sensitive channels, run: 
```
python SNNL.py
```
remember to input your own pre-trained CNN model at ```line #379```.

### Prune and fine-tune the model
Using ```train_cnn.py``` to train and fine-tune your own model to achieve better fairness.
```
python train_cnn.py
```
remember to input your pre-trained CNN model at ```line #208``` and the indexes of sensitive channels at ```line #212```, which can be obtained from the last step where sensitive channels were calculated.

### Citation
If it is helpful to you, please cite our work:
```
@misc{kong2024achieving,
      title={Achieving Fairness Through Channel Pruning for Dermatological Disease Diagnosis}, 
      author={Qingpeng Kong and Ching-Hao Chiu and Dewen Zeng and Yu-Jen Chen and Tsung-Yi Ho and Jingtong hu and Yiyu Shi},
      year={2024},
      eprint={2405.08681},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

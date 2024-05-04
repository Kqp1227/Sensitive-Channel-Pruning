# Sensitive-Channel-Pruning

This is the official repository of the following paper: Achieving Fairness Through Channel Pruning for Dermatological Disease Diagnosis

### Environment setup
```
pip install -r requirements.txt
```

### Calculate Sensitive Channels
To get sensitive channels, run: 
```
python SNNL.py
```
remember to input your own pre-trained CNN model in ```#L:379```.

### Prune and Fine-Tune the Model
Using ```train_cnn.py``` to train and fine-tune your own model to achieve better fairness.
```
python train_cnn.py
```

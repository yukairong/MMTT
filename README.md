# MMTT
Multi perspective and multi target tracking

## Tips
[1] You should set the **src** as **Source Root**, then run the train.py.   
[2] The configuration file should be written in **English** and cannot appear in Chinese.    
[3] If you want to **INSTALL MultiScaleDeformableAttention**, you should run 
``` python
    python src/models/ops/setup.py build --build-base=src/models/ops/ install
```
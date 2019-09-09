# Description for parameters that can be changed 

```
name - training session name 

n_gpu - number of GPUs to use. 0 means no GPU; use CPU instead.

gpu_id - GPU's id(s)

optimizer
    type - choice of optimizer : SGD, 

    args 
        lr - learning rate (default: 1.0)
        weight_decay - optional
        amsgrad 

weight_init_range - values to initialize the weight and other 
                    parameters

dropout_rate - dropout rate to be applied in hidden layer 
                (default: 0.3)

num_layer - number of layers (default:2)

emb_dim - emvedding size (default: 300)

hidden_dim - hiddent state size (default: 300)

batch_size - batch size (default: 32)

epoch - (default: 32)

grad_clipping - gradient clipping value (default: 5.0)
```
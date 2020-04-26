# CaffeBatchnormFuse
Python script to fuse the conv-batchnorm-scale and batchnorm-scale group layers in caffe model to speed up the inference

# Conv-Batchnorm-Scale layer group
Fuse the conv-batchnorm-scale layer into a singel conv layer:

Conv:
```
convout(N_i,C_out)=bias(C_out )+∑_(k=0)^(C_in-1)〖weight(C_out,k)*input(N_i,k)〗
```
BN:
```
bnout(N_i,C_out )=((convout(N_i,C_out)-μ))⁄√(〖ϵ+ σ〗^2 )
```
SC:
```
scout(N_i,C_out )=γ*bn_out(N_i,C_out)+β
```
Overall:
```
out(N_i,C_out )=γ*((bias(C_out )+∑_(k=0)^(C_in-1)〖weight(C_out,k)*input(N_i,k)〗-μ))⁄√(〖ϵ+ σ〗^2 )+β
```
To fold BN and SC into Conv we can convert the formula to get new bias and weights:
```
newbias=γ*(bias-μ)/√(〖ϵ+ σ〗^2 )+β
newweights=γ*weights/√(〖ϵ+ σ〗^2 )
```

# Batchnorm-Scale layer group
Fuse the batchnorm-scale layer into a singel scale layer:

For an input, `x`, the batch normalisation and scale layers at test time, perform

```
\gamma * (x - \mu) / \sigma + \beta
```

This can be converted to a single scale layer

```
(\gamma / \sigma) * x + (\beta - \gamma * \mu / \sigma)
```

Here, `\mu` is the mean, `\sigma` the standard deviation, `\gamma` the learned scale, and `\beta` the learned bias.

# Usage
The caffe python environment should be setup before run this python script.
The fused caffe prototxt and caffemoel file will be generated with the "_m" suffix.

```
python CaffeBatchnormFuse.py 
--proto <path to original input caffe .prototxt file>
--model <path to original input caffe .caffemodel file> 
--test <1 to inference with the fused model and original model>
```

# Reference:
https://github.com/zhang-xin/CNN-Conv-BatchNorm-fusion
https://github.com/hmph/caffe-fold-batchnorm
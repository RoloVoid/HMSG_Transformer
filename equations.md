### Temporal Attention Layer

$$
e_l^k=\mathbf{v}_e^Ttanh(\mathbf{W_e z_i}+\mathbf{U_e x^k})
$$

* 其中除z,x外都是需要学习的参数,x为特征序列

$$
\alpha_l^k=\frac {exp(e_t^k)}{\Sigma_{i=1}^nexp(e_l^i)}
$$

* 使用SoftMax计算出对应的权重，用于最后加权输出

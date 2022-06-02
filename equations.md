### Temporal Attention Layer

* 其中除z,x外都是需要学习的参数,x为特征序列
* 使用SoftMax计算出对应的权重，用于最后加权输出

$$
e_l^k=\mathbf{v}_e^Ttanh(\mathbf{W_e[d_{t-1};s^{'}_{t-1}]}+\mathbf{U_e x^k})
$$


$$
\alpha_l^k=\frac {exp(e_t^k)}{\Sigma_{i=1}^nexp(e_l^i)}
$$


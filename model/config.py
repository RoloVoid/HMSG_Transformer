'''
declare global constants and basic import
'''

device = "cuda"
f_size = 5 # feature series
d_ff = 2048 # dim of hidden state from posfeedforward 
n_heads = 4 # heads number from multihead_attn
n_layers = 6 # nums of encoder_layers
d_k = 9
d_v = 9
epochs = 100
batch_size = 5
seq_len = 10
# simga_hs = [5,10,20,40]
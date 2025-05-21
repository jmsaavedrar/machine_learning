import torch

x1 = torch.Tensor([[4,2,2,4]])
x2 = torch.Tensor([[1,3,1,4]])
x3 = torch.Tensor([[4,2,4,6]])

batch = torch.cat((x1,x2,x3), dim = 0)

#batch norm
bn = torch.nn.BatchNorm1d(num_features = 4, affine = False)
batch_bn = bn(batch)
mm = batch.mean(dim = 0, keepdim = True)
std = batch.std(dim = 0, unbiased = False, keepdim = True)
print((batch - mm)/std)
print(batch_bn)

#layer norm
ln = torch.nn.LayerNorm(normalized_shape = (4), elementwise_affine = False)
batch_ln = ln(batch)
mm = batch.mean(dim = 1, keepdim = True)
std = batch.std(dim = 1, unbiased = False, keepdim = True)
print((batch - mm)/(std))
print(batch_ln)


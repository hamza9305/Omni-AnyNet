import torch


a = torch.zeros([2, 4], dtype=torch.int32)
print(a)
print(a.dtype)


b = a.float()
print(b)
print(b.dtype)

b[0,0] = 4.5
print(b)

c = b.int()

print(c)

d = torch.ceil(b)
print(d)
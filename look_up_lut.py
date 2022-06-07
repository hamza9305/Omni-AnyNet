import torch

# pseudo init out cost volume and index guess


w, h = 128, 128
depth = 25

lut = torch.ones(size=(h, w, depth), dtype=float)
tmp_array = torch.arange(0, depth)
tmp_array = tmp_array[None, None, :]
lut *= tmp_array

# first slice: only 0's
# second slice: only 1's...

# best_index_guess = torch.randint(low=0, high=12, size=(64, 64, 1))
#    | 
# upsample
#    |
#    V
best_index_guess = torch.randint(low=0, high=12, size=(128, 128, 1))
best_index_guess *= 2 # upsample

print(f"{best_index_guess.shape=}")
print(best_index_guess)
best_index_guess = best_index_guess.repeat([1, 1, 5])
print(f"{best_index_guess.shape=}")


lower = -2
upper = +2
summand = torch.arange(lower, upper+1)

print(f"{summand.shape=}")
summand = summand[None, None, :]
print(f"{summand.shape=}")
# print(summand)
# exit()


# best_index_guess has shape h x w x d
# summand          has shape 1 x 1 x d

index_guesses = best_index_guess + summand
invalid_index_guesses = index_guesses < 0
invalid_index_guesses = torch.logical_or(invalid_index_guesses, index_guesses >= depth)

index_guesses[invalid_index_guesses] = 0
print(index_guesses)
print(index_guesses.size())



lut_small = torch.gather(lut, 2, index_guesses)
print(lut_small)
print(lut_small.size())
assert lut_small.shape == (h, w, upper+1 - lower)




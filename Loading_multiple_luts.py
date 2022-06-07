import glob
import os
import re
import torch

variable = 'mask_stage1.pt'
regex = r'mask'


# if re.match(regex, variable):
#     print("wmwejniweb")
#
# exit()
def assig_mask_and_lut(list):

    for item in list:
        basname = os.path.basename(item)

        if re.match(regex, basname):
            mask = torch.load(item,map_location="cuda:0")
        else:
            lut = torch.load(item,map_location="cuda:0")

    return lut, mask




lookup_path = '/home/haahm/PycharmProjects/Master_Thesis/FisheyeNet_MultipleLuts/AnyNet/Lookup_tables/'

dict = {"stage0":[],"stage1":[],"stage2":[]};


for path, subdirs, files in os.walk(lookup_path):
    for name in files:
        file = os.path.join(path, name)
        print(file)

        stage = file[-4]
        stage = 'stage' + str(stage)
        print(stage)
        dict[stage].append(file)


lut, mask = assig_mask_and_lut(dict['stage1'])

print(lut.size())
print(mask.size())



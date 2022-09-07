import torch

for i in range(6):
    confidence_orig = torch.load("../tensor_comp/orig/confidence{}.pt".format(i))
    confidence_padd = torch.load("../tensor_comp/padd/confidence{}.pt".format(i))
    location_orig = torch.load("../tensor_comp/orig/location{}.pt".format(i))
    location_padd = torch.load("../tensor_comp/padd/location{}.pt".format(i))
    if (i == 0):
        confidence_padd = confidence_padd[:, :, 0:19, 0:19]
        location_padd = location_padd[:, :, 0:19, 0:19]
    print(i)
    print("confidence", torch.equal(confidence_orig, confidence_padd))
    print(confidence_orig - confidence_padd)
    print("location", torch.equal(location_orig, location_padd))
    print(location_orig - location_padd)
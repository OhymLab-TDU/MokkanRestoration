import torch
import torch.nn as nn


class EdgeAccuracy(nn.Module):
    """
    Measures the accuracy of the edge map
    """
    def __init__(self, threshold=0.5):
        super(EdgeAccuracy, self).__init__()
        self.threshold = threshold
    print("self.threshold")


    def __call__(self, inputs, outputs):
        labels = (inputs > self.threshold)
        #print("labels")
        outputs = (outputs > self.threshold)
        #print("outputs")

        relevant = torch.sum(labels.float())
        #print("relevant")
        selected = torch.sum(outputs.float())
        #print("selected")

        if relevant == torch.tensor(0) and selected == torch.tensor(0):
            return torch.tensor(1), torch.tensor(1)

        true_positive = ((outputs == labels) * labels).float()
        #print("true_positive")
        recall = torch.sum(true_positive) / (relevant + 1e-8)
        #print("recall")
        precision = torch.sum(true_positive) / (selected + 1e-8)
        #print("precision")

        return precision, recall
    

class PSNR(nn.Module):
    def __init__(self, max_val):
        super(PSNR, self).__init__()

        base10 = torch.log(torch.tensor(10.0))
        print("base10")
        max_val = torch.tensor(max_val).float()
        print("max_val")
        self.register_buffer('base10', base10)
        self.register_buffer('max_val', 20 * torch.log(max_val) / base10)

    def __call__(self, a, b):
        mse = torch.mean((a.float() - b.float()) ** 2)
        #print("mse")
        if mse == 0:
            return torch.tensor(0)

        return self.max_val - 10 * torch.log(mse) / self.base10
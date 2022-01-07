
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import transformer 

class AndModule(nn.Module):
    """ ‘And’ modules represent the <intersect> program element.It combines two attention masks
        using an elementwise minimum function"
    """
    def forward(self, attn1, attn2):
        out = torch.min(attn1, attn2)
        return out


class OrModule(nn.Module):
    """ ‘Or’ modules represent the <union> program element. It combines two attention masks using an
        elementwise maximum function"
    """
    def forward(self, attn1, attn2):
        out = torch.max(attn1, attn2)
        return out


class AttentionModule(nn.Module):
    """ ‘Attention’ modules are used to represent the program elements which act as an attribute
        filter for a program (i.e. <filter_color>, <filter_material>, <filter_size>, <filter_shape>).
        The module firstly performs an elementwise multiplication between the previous module output
        with the feature embeddings to produce an attention mask, before processing the resulting
        embeddings containing the ‘regions of interest’ through two layers of 3x3 kernel CNN + ReLU
        and finally a 1x1 kernel CNN + elementwise sigmoid to produce the final downsampled attention
        mask highlighting the probabilistic attended features, ready for use by the next module in the
        reasoning chain.
    """
    def __init__(self, dim):
        super().__init__()
        self.transformer = transformer.Transformer()

        self.convtrans = nn.Conv2d(dim, 4, kernel_size=3, padding=1) 
        torch.nn.init.kaiming_normal_(self.convtrans.weight)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1) 
        self.conv3 = nn.Conv2d(dim, 1, kernel_size=1, padding=0)
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        self.dim = dim

    def forward(self, feats, attn):
        repeat = int(feats.size(1)/attn.size(1))

        if (attn.size() == feats.size()):
           attended_feats = torch.mul(feats, attn)
        else:
           attended_feats = torch.mul(feats, attn.repeat(1, repeat, 1, 1))        
        out = F.relu(self.conv1(attended_feats))
        out = F.relu(self.conv2(out))
        out = F.sigmoid(self.conv3(out))
        return out
        

class QueryModule(nn.Module):
    """ ‘Query’ modules represent the program elements which query an attribute, the existing of an object
        or the quantity of objects in an image (i.e. <count>, <exist>, <query_color>, <query_material>,
        <query_shape>, <query_size>). The module works in similar manner to the ‘Attention’ module, although
        performs a simpler set of convolutions involving two layers 3x3 kernel CNN + ReLU. Query modules
        are often used as the last step in a module chain and the prior step to the MLP classifier,
        therefore, the sigmoid function used by the Attention module is not needed. The ‘Query’ module,
        in addition to the ‘Comparison’ module are they only modules which can end a reasoning chain.
    """
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        self.dim = dim

    def forward(self, feats, attn):
        print("Query Module")

        repeat = int(feats.size(1)/attn.size(1))
        print('repeat=',repeat)
        if (attn.size() == feats.size()):
           attended_feats = torch.mul(feats, attn)
        else:
           attended_feats = torch.mul(feats, attn.repeat(1, repeat, 1, 1))
       
        out = F.relu(self.conv1(attended_feats))
        out = F.relu(self.conv2(out))
        return out


class RelateModule(nn.Module):
    """ ‘Relate’ modules are used to represent spatial DSL terms such as <relate[left]>, <relate[right]>,
        <relate[front]> and <relate[behind]>
        A Vision Transformer is used as the attention module responsible for taking the previous modules
        attention map and attending the relevant features.
    """
    def __init__(self, dim):
        super().__init__()
        self.convtrans = nn.Conv2d(dim, 4, kernel_size=3, padding=1) 
        torch.nn.init.kaiming_normal_(self.convtrans.weight)
        self.transformer = transformer.Transformer()
        self.dim = dim

    def forward(self, feats, attn):

        repeat = int(feats.size(1)/attn.size(1))
        if (attn.size() == feats.size()):
            attended_feats = torch.mul(feats, attn)
        else:
           attended_feats = torch.mul(feats, attn.repeat(1, repeat, 1, 1))
        
        attended_feats = F.relu(self.convtrans(attended_feats))
        out = self.transformer(attended_feats)
        return out


class SameModule(nn.Module):
    """ ‘Same’ modules represent the program elements which perform a same comparison on object features
        (i.e. <same_color>, <same_material>, <same_shape>, <same_size>). The ‘Same’ module generates an
        attention map, again by performing an elementwise multiplication between the previous module output
        and the extracted features. The index of the maximally attended object is then located using maximum
        pooling, before extracting the respective feature vector. Cross-correlation is then performed between
        the feature vector and all other map locations in an attempt to find identical features. The output
        is then downsampled to produce the resulting attention mask.
    """
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim+1, 1, kernel_size=1)
        torch.nn.init.kaiming_normal_(self.conv.weight)
        self.dim = dim

    def forward(self, feats, attn):
        print("Same Module")
        size = attn.size()[2]
        the_max, the_idx = F.max_pool2d(attn, size, return_indices=True)
        attended_feats = feats.index_select(2, the_idx[0, 0, 0, 0] / size)
        attended_feats = attended_feats.index_select(3, the_idx[0, 0, 0, 0] % size)
        x = torch.mul(feats, attended_feats.repeat(1, 1, size, size))
        x = torch.cat([x, attn], dim=1)

        out = F.sigmoid(self.conv(x))
        return out


class ComparisonModule(nn.Module):
    """ ‘Comparison’ modules represent the program elements which perform and equal comparison between object
        attributes, or greater than or less than operation (i.e. <equal_color>, <equal_material>, <equal_integer>,
        <equal_shape>, <equal_size>, <greater_than>,  <less_than>). The module takes outputs from two previous
        modules in the module chain, concatenates them and then performs a downsampling convolution to determine
        whether the two feature maps contain the same property, before a final set of convolutions produce the
        resulting output attention mask. The ‘Comparison’ module, in addition to the ‘Query’ module are they only
        modules which can end a reasoning chain. 
    """
    def __init__(self, dim):
        print("Comparison Module")
        super().__init__()

        self.projection = nn.Conv2d(2*dim, dim, kernel_size=1, padding=0)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)

    def forward(self, in1, in2):
        out = torch.cat([in1, in2], 1)
        
        out = F.relu(self.projection(out))
        out = F.relu(self.conv1(out))
        out = F.relu(self.conv2(out))
        return out

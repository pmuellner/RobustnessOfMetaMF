import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter

class MetaRecommender(nn.Module):
    def __init__(self, user_num, item_num, item_emb_size=32, item_mem_num=8, user_emb_size=32, mem_size=128, hidden_size=512):
        super(MetaRecommender, self).__init__()
        self.item_num = item_num
        self.item_emb_size = item_emb_size
        self.item_mem_num = item_mem_num
        self.user_embedding = nn.Embedding(user_num, user_emb_size)
        self.memory = Parameter(nn.init.xavier_normal_(torch.Tensor(user_emb_size, mem_size)), requires_grad=True)
        self.hidden_layer_1, self.weight_layer_1, self.bias_layer_1 = self.define_one_layer(mem_size, hidden_size, item_emb_size, int(item_emb_size/4))
        self.hidden_layer_2, self.weight_layer_2, self.bias_layer_2 = self.define_one_layer(mem_size, hidden_size, int(item_emb_size/4), 1)
        self.hidden_layer_3, self.emb_layer_1, self.emb_layer_2 = self.define_item_embedding(item_num, item_emb_size, item_mem_num, mem_size, hidden_size)
    
    def define_one_layer(self, mem_size, hidden_size, int_size, out_size):
        hidden_layer = nn.Linear(mem_size, hidden_size)
        weight_layer = nn.Linear(hidden_size, int_size*out_size)
        bias_layer = nn.Linear(hidden_size, out_size)
        return hidden_layer, weight_layer, bias_layer
    
    def define_item_embedding(self, item_num, item_emb_size, item_mem_num, mem_size, hidden_size):
        hidden_layer = nn.Linear(mem_size, hidden_size)
        emb_layer_1 = nn.Linear(hidden_size, item_num*item_mem_num)
        emb_layer_2 = nn.Linear(hidden_size, item_mem_num*item_emb_size)
        return hidden_layer, emb_layer_1, emb_layer_2 
            
    def forward(self, user_id):
        #collaborative memory module
        user_emb = self.user_embedding(user_id)
        cf_vec = torch.matmul(user_emb, self.memory)

        #meta recommender module
        output_weight = []
        output_bias = []
        
        ## hypernetwork
        weight, bias = self.get_one_layer(self.hidden_layer_1, self.weight_layer_1, self.bias_layer_1, cf_vec, self.item_emb_size, int(self.item_emb_size/4))
        output_weight.append(weight)
        output_bias.append(bias) 
                
        weight, bias = self.get_one_layer(self.hidden_layer_2, self.weight_layer_2, self.bias_layer_2, cf_vec, int(self.item_emb_size/4), 1)
        output_weight.append(weight)
        output_bias.append(bias)
        
        item_embedding = self.get_item_embedding(self.hidden_layer_3, self.emb_layer_1, self.emb_layer_2, cf_vec, self.item_num, self.item_mem_num, self.item_emb_size)
        
        return output_weight, output_bias, item_embedding, cf_vec
    
    def get_one_layer(self, hidden_layer, weight_layer, bias_layer, cf_vec, int_size, out_size):
        hid = hidden_layer(cf_vec)
        hid = F.relu(hid)
        weight = weight_layer(hid)
        bias = bias_layer(hid)
        weight = weight.view(-1, int_size, out_size)
        bias = bias.view(-1, 1, out_size)
        return weight, bias
    
    def get_item_embedding(self, hidden_layer, emb_layer_1, emb_layer_2, cf_vec, item_num, item_mem_num, item_emb_size):
        hid = hidden_layer(cf_vec)
        hid = F.relu(hid)
        emb_left = emb_layer_1(hid)
        emb_right = emb_layer_2(hid)
        emb_left = emb_left.view(-1, item_num, item_mem_num)
        emb_right = emb_right.view(-1, item_mem_num, item_emb_size)
        item_embedding = torch.matmul(emb_left, emb_right)
        return item_embedding
    
    def fix_hypernetwork(self):
        self.hidden_layer_1.weight.requires_grad = False
        self.hidden_layer_1.bias.requires_grad = False
        self.weight_layer_1.weight.requires_grad = False
        self.weight_layer_1.bias.requires_grad = False
        self.bias_layer_1.weight.requires_grad = False
        self.bias_layer_1.bias.requires_grad = False

        self.hidden_layer_2.weight.requires_grad = False
        self.hidden_layer_2.bias.requires_grad = False
        self.weight_layer_2.weight.requires_grad = False
        self.weight_layer_2.bias.requires_grad = False
        self.bias_layer_2.weight.requires_grad = False
        self.bias_layer_2.bias.requires_grad = False
       
    
class MetaMF(nn.Module):
    def __init__(self, user_num, item_num, item_emb_size=32, item_mem_num=8, user_emb_size=32, mem_size=128, hidden_size=512):
        super(MetaMF, self).__init__()
        self.item_num = item_num
        self.metarecommender = MetaRecommender(user_num, item_num, item_emb_size, item_mem_num, user_emb_size, mem_size, hidden_size)
        
    def disable_meta_learning(self):
        self.metarecommender.fix_hypernetwork()
        
    def forward(self, user_id, item_id):
        #prediction module
        model_weight, model_bias, item_embedding, _ = self.metarecommender(user_id)
        item_id = item_id.view(-1, 1)
        item_one_hot = torch.zeros(len(item_id), self.item_num, device=item_id.device)
        item_one_hot.scatter_(1, item_id, 1)
        item_one_hot = torch.unsqueeze(item_one_hot, 1)
        item_emb = torch.matmul(item_one_hot, item_embedding)
        out = torch.matmul(item_emb, model_weight[0])
        out = out+model_bias[0]
        out = F.relu(out)
        out = torch.matmul(out, model_weight[1])
        out = out+model_bias[1]
        out = torch.squeeze(out)
        return out
    
    def loss(self, prediction, rating):
        return torch.mean(torch.pow(prediction-rating,2))
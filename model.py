from transformers import BertModel,BertTokenizer
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.distributions.bernoulli import Bernoulli
import numpy as np
import datetime


class Bert_RNP2_share(nn.Module):
    def __init__(self,args):
        super(Bert_RNP2_share, self).__init__()
        print('Bert_RNP2_share')
        self.encoder = BertModel.from_pretrained("bert-base-uncased",return_dict=False)
        self.encoder_word_embedding_fn = lambda t: self.encoder.embeddings.word_embeddings(t)
        self.re_encoder = self.encoder
        self.re_encoder_word_embedding_fn = lambda t: self.re_encoder.embeddings.word_embeddings(t)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.hidden_dim = args.hidden_dim
        self.num_tags = args.num_tags
        self.x_2_prob_z = nn.Linear(self.hidden_dim, self.num_tags)
        self.device = args.device
        self.alpha_rationle = args.alpha_rationle
        self.class_num = args.class_num
        self.classifier = nn.Linear(self.hidden_dim, self.class_num)

    def forward(self, input_ids, attention_mask):
        eps = 1e-8
        output_s = self.encoder(inputs_embeds = self.encoder_word_embedding_fn(input_ids), attention_mask = attention_mask)
        selector_out = output_s[0]
        batch_size = input_ids.size(0)
        feats = self.x_2_prob_z(selector_out)
        rationale_mask = []
        special_mask = torch.zeros(attention_mask.size())
        special_mask = special_mask.to(attention_mask.device)
        special_mask[:,0] = 1.0

        if self.training:
            sampled_seq = F.gumbel_softmax(feats,hard=False,dim=2)
            sampled_seq = sampled_seq[:,:,-1].unsqueeze(2)
            sampled_seq = sampled_seq * attention_mask.unsqueeze(2)
            sampled_seq = sampled_seq.squeeze(-1)
        else:
            mask_length = attention_mask.sum(-1).cpu().numpy()
            rationales = feats.cpu().argmax(dim=-1).numpy()
            rationale_mask = []
            for k_index,rationale in enumerate(rationales):
                rationale_mask.append(rationale[0:mask_length[k_index]])
            sampled_seq = torch.tensor(rationales)
            # print(sampled_seq)
            sampled_seq = sampled_seq.to(attention_mask.device)
            sampled_seq = sampled_seq * attention_mask

        sampled_seq = 1 - (1 - sampled_seq) * (1 - special_mask)

        predictor_inputs_embeds = self.re_encoder_word_embedding_fn(input_ids)

        mask_embedding = self.re_encoder_word_embedding_fn(torch.scalar_tensor(self.tokenizer.mask_token_id,dtype=torch.long,device=sampled_seq.device))
        
        masked_inputs_embeds = predictor_inputs_embeds * sampled_seq.unsqueeze(2) + mask_embedding * (1 - sampled_seq.unsqueeze(2))
        output_p = self.re_encoder(inputs_embeds = masked_inputs_embeds,attention_mask=attention_mask)
        # s_w_feature = output_p[0]
        # s_w_feature = s_w_feature * sampled_seq.unsqueeze(-1)
        # pred_out = torch.sum(s_w_feature, dim = 1)/ sampled_seq.sum(-1).unsqueeze(1)

        pred_out = output_p[1]

        output = self.classifier(pred_out) 

        # output the non-rationale rep
        non_rationale_masked_inputs_embeds = mask_embedding * sampled_seq.unsqueeze(2) + predictor_inputs_embeds * (1 - sampled_seq.unsqueeze(2))
        
        output_non_rationale = self.re_encoder(inputs_embeds = non_rationale_masked_inputs_embeds,attention_mask=attention_mask)[1]


        infor_loss = (sampled_seq.sum(-1) / (attention_mask.sum(1)+eps) ) - self.alpha_rationle
        self.infor_loss = torch.abs(infor_loss).mean()
        # self.infor_loss = infor_loss.mean()
        regular =  torch.abs(sampled_seq[:,1:] - sampled_seq[:,:-1]).sum(1) / (attention_mask.sum(1)-1+eps)
        self.regular = regular.mean()
        # return output , output_p
        return output , rationale_mask, output_non_rationale  #,output , rationale_mask



class INVRAT(nn.Module):
    def __init__(self,args):
        super(INVRAT, self).__init__()
        print('INVRAT')
        self.encoder = BertModel.from_pretrained("bert-base-uncased",return_dict=False)
        self.encoder_word_embedding_fn = lambda t: self.encoder.embeddings.word_embeddings(t)
        self.re_encoder = self.encoder
        self.re_encoder_word_embedding_fn = lambda t: self.re_encoder.embeddings.word_embeddings(t)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.hidden_dim = args.hidden_dim
        self.num_tags = args.num_tags
        self.x_2_prob_z = nn.Linear(self.hidden_dim, self.num_tags)
        self.device = args.device
        self.alpha_rationle = args.alpha_rationle
        self.class_num = args.class_num
        self.classifier = nn.Linear(self.hidden_dim, self.class_num)


        ###
        self.re_encoder_env = BertModel.from_pretrained("bert-base-uncased",return_dict=False)
        self.re_encoder_env_word_embedding_fn = lambda t: self.re_encoder_env.embeddings.word_embeddings(t)
        self.classifier_env = nn.Linear(self.hidden_dim+2, 2)


    def forward(self, input_ids, attention_mask,env_id):
        eps = 1e-8
        output_s = self.encoder(inputs_embeds = self.encoder_word_embedding_fn(input_ids), attention_mask = attention_mask)
        selector_out = output_s[0]
        batch_size = input_ids.size(0)
        feats = self.x_2_prob_z(selector_out)
        rationale_mask = []
        special_mask = torch.zeros(attention_mask.size())
        special_mask = special_mask.to(attention_mask.device)
        special_mask[:,0] = 1.0

        if self.training:
            sampled_seq = F.gumbel_softmax(feats,hard=False,dim=2)
            sampled_seq = sampled_seq[:,:,-1].unsqueeze(2)
            sampled_seq = sampled_seq * attention_mask.unsqueeze(2)
            sampled_seq = sampled_seq.squeeze(-1)
        else:
            mask_length = attention_mask.sum(-1).cpu().numpy()
            rationales = feats.cpu().argmax(dim=-1).numpy()
            rationale_mask = []
            for k_index,rationale in enumerate(rationales):
                rationale_mask.append(rationale[0:mask_length[k_index]])
            sampled_seq = torch.tensor(rationales)
            # print(sampled_seq)
            sampled_seq = sampled_seq.to(attention_mask.device)
            sampled_seq = sampled_seq * attention_mask

        sampled_seq = 1 - (1 - sampled_seq) * (1 - special_mask)

        predictor_inputs_embeds = self.re_encoder_word_embedding_fn(input_ids)

        mask_embedding = self.re_encoder_word_embedding_fn(torch.scalar_tensor(self.tokenizer.mask_token_id,dtype=torch.long,device=sampled_seq.device))
        
        masked_inputs_embeds = predictor_inputs_embeds * sampled_seq.unsqueeze(2) + mask_embedding * (1 - sampled_seq.unsqueeze(2))
        output_p = self.re_encoder(inputs_embeds = masked_inputs_embeds,attention_mask=attention_mask)
        # s_w_feature = output_p[0]
        # s_w_feature = s_w_feature * sampled_seq.unsqueeze(-1)
        # pred_out = torch.sum(s_w_feature, dim = 1)/ sampled_seq.sum(-1).unsqueeze(1)

        pred_out = output_p[1]

        output = self.classifier(pred_out) 


        #####

        np_label = np.expand_dims(np.array(env_id, dtype=np.float32), axis=1)
        one_hot_label = np.concatenate([1. - np_label, np_label], axis=1)
        env = torch.tensor(one_hot_label)
        env = env.to(self.device)

        predictor_inputs_embeds = self.re_encoder_env_word_embedding_fn(input_ids)

        mask_embedding = self.re_encoder_env_word_embedding_fn(torch.scalar_tensor(self.tokenizer.mask_token_id,dtype=torch.long,device=sampled_seq.device))
        
        masked_inputs_embeds = predictor_inputs_embeds * sampled_seq.unsqueeze(2) + mask_embedding * (1 - sampled_seq.unsqueeze(2))
        output_envs = self.re_encoder_env(inputs_embeds = masked_inputs_embeds,attention_mask=attention_mask)

        pred_out_env = output_envs[1]
        # xx = torch.tensor(one_hot_label)
        # expanded_tensor = xx.unsqueeze(1)
        # env_expanded_tensor = expanded_tensor.expand(pred_out_env.size())

        output_env = self.classifier_env(torch.cat([pred_out_env,env],-1)) 

        infor_loss = (sampled_seq.sum(-1) / (attention_mask.sum(1)+eps) ) - self.alpha_rationle
        self.infor_loss = torch.abs(infor_loss).mean()
        # self.infor_loss = infor_loss.mean()
        regular =  torch.abs(sampled_seq[:,1:] - sampled_seq[:,:-1]).sum(1) / (attention_mask.sum(1)-1+eps)
        self.regular = regular.mean()
        # return output , output_p
        if self.training:
            return [output , output_env],rationale_mask
        else:
            return output,rationale_mask,output,rationale_mask




r"""Functional interface"""
import warnings
import math


import torch
from torch import Tensor
from torch.overrides import has_torch_function, handle_torch_function

def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):

    if not torch.jit.is_scripting():
        if type(logits) is not Tensor and has_torch_function((logits,)):
            return handle_torch_function(
                gumbel_softmax, (logits,), logits, tau=tau, hard=hard, eps=eps, dim=dim)
    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    random_values = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
    gumbels = random_values.exponential_().log()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft

    if torch.any(torch.isnan(ret)):
        raise Exception('NaN result returned by Gumbel softmax function')

    return ret


def phi_to_rationale(phi: torch.Tensor,
                     binarization_method:str,
                     training:bool=True,
                     gumbel_train_temperature:float=None,
                     gumbel_eval_temperature:float=None,
                     hard:bool=False
                     ):
    if binarization_method == 'gumbel_softmax':
        # zero_phi = torch.log(1 - self.sigmoid(phi))

        if training:
            zero_phi = torch.zeros_like(phi)
            # mathematically this works out to a situation where the probabilities (which the gumbel softmax will approximate sampling from) will be = sigmoid(phi)
            both_phi = torch.stack([zero_phi, phi], dim=2)
            temperature = gumbel_train_temperature if training else gumbel_eval_temperature
            predicted_rationale = gumbel_softmax(both_phi, hard=hard,tau=temperature, dim=2)[:, :, 1]
            predicted_rationale = torch.nan_to_num(predicted_rationale, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            predicted_rationale = torch.sigmoid(phi).round()

    # phi = phi[:, :, 1]
    elif binarization_method == 'bernoulli':
        predicted_rationale_probs = torch.sigmoid(phi)
        # result['predicted_rationale_c_probs'] = predicted_rationale_probs
        predicted_rationale = torch.bernoulli(predicted_rationale_probs).detach()
    elif binarization_method == 'sigmoid':
        predicted_rationale = torch.sigmoid(phi)

    return predicted_rationale


def mask_embeddings(inputs_embeds,mask,padding_mask,word_embedding_function,mask_token_id,masking_strategy):

    if masking_strategy == 'multiply_zero':
        # Just zero out the masked token embeddings
        masked_inputs_embeds = inputs_embeds * mask.unsqueeze(2)
        # masked_input_ids = input_ids
        masked_padding_mask = padding_mask
    # masked_special_mask = special_mask
    elif masking_strategy == 'multiply_mask':
        # Replace masked token embeddings with the embedding for the [MASK] token
        # mask_token_ids = torch.ones_like(mask) * mask_token_id
        # with torch.no_grad():
        #   mask_embeds = word_embedding_function(mask_token_ids)
        mask_embedding = word_embedding_function(torch.scalar_tensor(mask_token_id,dtype=torch.long,device=mask.device))
        masked_inputs_embeds = inputs_embeds * mask.unsqueeze(2) + mask_embedding * (1 - mask.unsqueeze(2))
        # rounded_mask = (mask >= 0.5).int()
        # masked_input_ids = input_ids * rounded_mask + mask_token_ids * (1 - rounded_mask)
        masked_padding_mask = padding_mask
    # masked_special_mask = special_mask

    elif masking_strategy == 'bert_attention':
        #Just apply mask to the padding mask, so that bert ignores the indicated tokens
        masked_inputs_embeds = inputs_embeds
        # masked_input_ids = input_ids
        # masked_padding_mask = padding_mask * mask
        masked_padding_mask = padding_mask - (1-mask)

    # # masked_special_mask = special_mask
    # elif masking_strategy == 'token_type_ids':
    #   zero_embeddings = rationale_embedding_function(torch.zeros_like(mask,dtype=torch.long))
    #   type_embeddings = zero_embeddings * (1 - mask).unsqueeze(2) + \
    #                     rationale_embedding_function(torch.ones_like(mask,dtype=torch.long)) * mask.unsqueeze(2)
    #   #the subtraction is needed because the BERT embedding layer adds it by default
    #   masked_inputs_embeds = inputs_embeds - zero_embeddings+ type_embeddings
    #   # masked_inputs_embeds = inputs_embeds - type_embeddings

    #   # masked_input_ids = input_ids
    #   masked_padding_mask = padding_mask
    # # masked_special_mask = special_mask
    # elif masking_strategy == '0_1_embeddings' or masking_strategy == 'embeddings':
    #   type_embeddings = rationale_embedding_function(torch.zeros_like(mask)) * (1 - mask).unsqueeze(2) + rationale_embedding_function(torch.ones_like(mask)) * mask.unsqueeze(2)
    #   masked_inputs_embeds = inputs_embeds + type_embeddings
    #   # masked_input_ids = input_ids
    #   masked_padding_mask = padding_mask
    # # masked_special_mask = special_mask
    else:
        raise Exception(f'Unknown masking strategy {masking_strategy}')

    result ={'masked_inputs_embeds':masked_inputs_embeds,
             'masked_padding_mask':masked_padding_mask}

    return result















import logging
import os
import argparse
import random
from tqdm import tqdm, trange
import csv
import glob 
import json
from sklearn import metrics
import numpy as np
import torch
import os
import torch.nn as nn
from utils import *
from model import INVRAT
from torch.utils.data import TensorDataset, DataLoader
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import prettytable as pt
from sklearn.metrics import accuracy_score, auc, average_precision_score, classification_report, precision_recall_curve, roc_auc_score,f1_score
from sklearn.metrics import precision_recall_fscore_support
from torch.utils import data
from short_easy_utils import *

parser = argparse.ArgumentParser(description='VMask classificer')

parser.add_argument('--model_name', type=str, default='RNP', help='model name')
parser.add_argument('--lr', type=float, default=5e-5, help='initial learning rate')
parser.add_argument('--max_len', type=int, default=300, help='max_len')
parser.add_argument('--weight_decay', default=0, type=float, help='adding l2 regularization')
parser.add_argument('--dropout', type=float, default=0.2, help='the probability for dropout')
parser.add_argument('--alpha_rationle', type=float, default=0.2, help='alpha_rationle')
parser.add_argument('--hidden_dim', type=int, default=768, help='number of hidden dimension')
parser.add_argument("--activation", type=str, dest="activation", default="tanh", help='the choice of \
        non-linearity transfer function')
parser.add_argument('--gpu_id', default='0', type=str, help='gpu id')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--data_dir', type=str)
parser.add_argument('--dev_data_dir', type=str)
parser.add_argument('--types', type=str, default='legal', help='data_type')
parser.add_argument('--epochs', type=int, default=16, help='number of epochs for training')
parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
parser.add_argument('--class_num', type=int, default=2, help='class_num')
parser.add_argument('--save_path', type=str, default='', help='save_path')
parser.add_argument('--alpha', type=float, default=0)
parser.add_argument('--beta', type=float, default=0)
parser.add_argument('--use_crf', type=int, default=0)
parser.add_argument('--loss_type', type=str, default='mse')
parser.add_argument("--warmup_steps", default=0, type=int, 
                        help="Linear warmup over warmup_steps.")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="gradient_accumulation_steps")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, 
                        help="Epsilon for Adam optimizer.")
parser.add_argument('--fraction_rationales', type=float, default=1.0,
                        help='what fraction of sentences have rationales')
parser.add_argument('--weight_decay_finetune', type=float, default=1e-5,
                        help='weight decay finetune') 
parser.add_argument("--max_grad_norm", default=1.0, type=float, 
                        help="Max gradient norm.")
parser.add_argument("--dataset", default="movie_reviews", type=str,
                        help="dataset")
parser.add_argument("--date", default="1009", type=str)
parser.add_argument("--fixed", default="yes", type=str)
parser.add_argument('--num_tags', type=int, default=2)
parser.add_argument('--is_da', type=str, default='no')
parser.add_argument("--pretraining", default="yes", type=str)
args = parser.parse_args()

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO,)
logger = logging.getLogger(__name__)
# logger_combine_AT

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

def random_seed():
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return

args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(args.device)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
dataset_processor = MovieReviewsProcessor()


dataset_processor.set_fraction_rationales(args.fraction_rationales)

train_examples = dataset_processor.get_train_env_examples(args.data_dir)
dev_examples = dataset_processor.get_dev_env_examples(args.data_dir)
test_examples = dataset_processor.get_test_env_examples(args.data_dir)


print(len(train_examples))

tag_map = dataset_processor.get_tag_map()
num_labels = dataset_processor.get_num_labels()
num_tags = dataset_processor.get_num_tags()
args.num_tags = num_tags


train_dataset = DatasetWitheasyRationalesEnv(train_examples, tokenizer, tag_map, args.max_len, \
    args.dataset)
dev_dataset = DatasetWitheasyRationalesEnv(dev_examples, tokenizer, tag_map, args.max_len, \
    args.dataset)
test_dataset = DatasetWitheasyRationalesEnv(test_examples, tokenizer, tag_map, args.max_len, \
    args.dataset)


train_dataloader = data.DataLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    collate_fn=DatasetWitheasyRationalesEnv.pad,
    worker_init_fn=np.random.seed(args.seed),
)

dev_dataloader = data.DataLoader(
    dataset=dev_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    collate_fn= DatasetWitheasyRationalesEnv.pad,
    worker_init_fn=np.random.seed(args.seed),
)

test_dataloader = data.DataLoader(
    dataset=test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    #num_workers=8,
    #pin_memory=True,
    collate_fn= DatasetWitheasyRationalesEnv.pad,
    worker_init_fn=np.random.seed(args.seed),
)



for k, v in vars(args).items():
    logger.info("{:20} : {:10}".format(k, str(v)))

def main():

    model = eval(args.model_name)(args)
    model = model.to(args.device)

    num_training_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.epochs

    named_params = list(model.encoder.parameters()) + list(model.x_2_prob_z.parameters()) 

    no_decay = ['bias', 'LayerNorm.weight']


    optimizer = AdamW(named_params, lr=args.lr, eps=args.adam_epsilon)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps
    )
    
    optimizer_inv = AdamW(list(model.re_encoder.parameters())+list(model.classifier.parameters()), lr=args.lr)

    scheduler_inv = get_linear_schedule_with_warmup(
        optimizer_inv, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps
    )


    optimizer_env = AdamW(list(model.re_encoder_env.parameters())+list(model.classifier_env.parameters()), lr=args.lr,eps=args.adam_epsilon)

    scheduler_env = get_linear_schedule_with_warmup(
        optimizer_env, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps
    )

    for epoch in range(1, args.epochs+1):
        model.train()
        logger.info("Trianing Epoch: {}/{}".format(epoch, int(args.epochs)))
        #  for bert cross small dataset

        if args.pretraining == 'yes' and epoch in [1,2,3]:
            named_params_pre = list(model.re_encoder.named_parameters()) 
            named_params_pre.extend(list(model.classifier.named_parameters()))
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters_pre = [
                {'params': [p for n, p in named_params_pre \
                    if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay_finetune}
            ]
            optimizer_pre = AdamW(optimizer_grouped_parameters_pre, lr=args.lr, eps=args.adam_epsilon)
        
            scheduler_pre = get_linear_schedule_with_warmup(
            optimizer_pre, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps
            )

            for step,batch in enumerate(tqdm(train_dataloader)):

                input_ids, attention_mask, label_ids, tag_ids, has_rationale, start_labels, end_labels, span_labels = batch
                # print(input_ids)
                # if step>10:break
                
                optimizer_pre.zero_grad()
                input_ids = input_ids.to(args.device)
                tag_ids  =  tag_ids.to(args.device)
                attention_mask = attention_mask.to(args.device)
                label_ids = label_ids.to(args.device)
                predictor_inputs_embeds = model.re_encoder_word_embedding_fn(input_ids)
                output_p = model.re_encoder(inputs_embeds = predictor_inputs_embeds,attention_mask=attention_mask)
                pred_out = output_p[1]
                # print(pred_out)
                output = model.classifier(pred_out) 
                # print(label_ids)
                # print(output)
                loss = criterion(output,label_ids) 

                loss = loss / args.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:

                    torch.nn.utils.clip_grad_norm_(model.re_encoder.parameters(), args.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), args.max_grad_norm)
                    optimizer_pre.step()
                    scheduler_pre.step()
                    
            if epoch>0:
                model.eval()
                ##############################  dev  ##############################
                predictions_charge = []
                true_charge = []
                for step,batch in enumerate(tqdm(dev_dataloader)):
                    input_ids, attention_mask, label_ids, tag_ids, has_rationale, start_labels, end_labels, span_labels = batch

                    input_ids = input_ids.to(args.device)
                    
                    attention_mask = attention_mask.to(args.device)
                    true_charge.extend(label_ids.cpu().numpy())
                    
                    with torch.no_grad():
                        predictor_inputs_embeds = model.re_encoder_word_embedding_fn(input_ids)
                        output_p = model.re_encoder(inputs_embeds = predictor_inputs_embeds,attention_mask=attention_mask)
                        pred_out = output_p[1]
                        output = model.classifier(pred_out) 

                    pred = output.cpu().argmax(dim=1).numpy()
                    predictions_charge.extend(pred)
                dev_eval = {}
                print(true_charge)
                print(predictions_charge)
                class_micro_f1, class_macro_precision, class_macro_recall, class_macro_f1 = eval_data_types(true_charge,predictions_charge,num_labels=args.class_num)

                table = pt.PrettyTable(['types ','     Acc   ','          P          ', '          R          ', '      F1          ' ]) 
                table.add_row(['dev',  class_micro_f1, class_macro_precision, class_macro_recall, class_macro_f1 ])
                logger.info(table)

        else:


            for step,batch in enumerate(tqdm(train_dataloader)):

                path = step % 7
                input_ids, attention_mask, label_ids, tag_ids, has_rationale, start_labels, end_labels, env_ids = batch

                optimizer.zero_grad()
                
                input_ids = input_ids.to(args.device)
                tag_ids  =  tag_ids.to(args.device)
                attention_mask = attention_mask.to(args.device)
                label_ids = label_ids.to(args.device)
                
                output,_ = model(input_ids,attention_mask,env_ids)

            
                env_inv_loss = criterion(output[0],label_ids)
                env_enable_loss = criterion(output[1],label_ids)
                diff_loss = torch.max(torch.zeros_like(env_inv_loss), env_inv_loss - env_enable_loss)
                
                gen_loss = 10 * diff_loss + env_inv_loss

                gen_loss = gen_loss + args.alpha*model.infor_loss + args.beta*model.regular
                gen_loss = gen_loss / args.gradient_accumulation_steps

                if path in [0]:
                # update the generator
                    optimizer.zero_grad()
                    gen_loss.backward()
                    if (step + 1) % args.gradient_accumulation_steps == 0:

                        torch.nn.utils.clip_grad_norm_(list(model.encoder.parameters()) + list(model.x_2_prob_z.parameters()) +list(model.classifier.parameters()), args.max_grad_norm)
                        optimizer.step()
                        scheduler.step()

                elif path in [1,2,3]:
                    # update the env inv predictor
                    optimizer_inv.zero_grad() 
                    env_inv_loss.backward()
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(list(model.re_encoder.parameters()) + list(model.classifier.parameters()), args.max_grad_norm)
                        optimizer_inv.step()
                        scheduler_inv.step()

                    
                    
                elif path in [4,5,6]:
                    optimizer_env.zero_grad()
                    env_enable_loss.backward()
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(list(model.re_encoder_env.parameters()) + list(model.classifier_env.parameters()), args.max_grad_norm)
                        optimizer_env.step()
                        scheduler_env.step()


            if epoch>0:
                model.eval()
                ##############################  dev  ##############################
                percents = 0
                predictions_charge_cross = []
                true_charge_cross = []

                predictions_charge_rnp = []

                all_tag_preds_cross = []
                all_tag_gold_cross = []

                all_tag_preds_rnp = []
                all_tag_gold_rnp = []


                for step,batch in enumerate(tqdm(dev_dataloader)):

                    input_ids, attention_mask, label_ids, tag_ids, has_rationale, start_labels, end_labels, env_ids = batch
                    true_charge_cross.extend(label_ids.cpu().numpy())
                    input_ids = input_ids.to(args.device)
                    attention_mask = attention_mask.to(args.device)
                    tag_ids = tag_ids.to(args.device)
                    label_ids = label_ids.to(args.device)
                    with torch.no_grad():
                        output_bert_cross, rationale_mask_bert_cross, output_bert_rnp, rationale_mask_bert_rnp = model(input_ids,attention_mask,env_ids)

                    pred_cross = output_bert_cross.cpu().argmax(dim=1).numpy()
                    predictions_charge_cross.extend(pred_cross)

                    pred_rnp = output_bert_rnp.cpu().argmax(dim=1).numpy()
                    predictions_charge_rnp.extend(pred_rnp)

                    pred_tags_flat = [val for sublist in rationale_mask_bert_cross for val in sublist]
                    gold_tags_flat = torch.masked_select(tag_ids, attention_mask==1).tolist()
                    all_tag_preds_cross.extend(pred_tags_flat)
                    all_tag_gold_cross.extend(gold_tags_flat)
                    
                    assert len(gold_tags_flat) == len(pred_tags_flat)
                    pred_tags_flat = [val for sublist in rationale_mask_bert_rnp for val in sublist]
                    gold_tags_flat = torch.masked_select(tag_ids, attention_mask==1).tolist()
                    all_tag_preds_rnp.extend(pred_tags_flat)
                    all_tag_gold_rnp.extend(gold_tags_flat)

                    assert len(gold_tags_flat) == len(pred_tags_flat)

                precision_tagging, recall_tagging, f1_tagging, _ = precision_recall_fscore_support(np.array(all_tag_gold_cross), np.array(all_tag_preds_cross), labels=[1])
                
                precision_tagging2, recall_tagging2, f1_tagging2, _ = precision_recall_fscore_support(np.array(all_tag_gold_rnp), np.array(all_tag_preds_rnp), labels=[1])

                class_macro_precision, class_macro_recall, class_macro_f1,_ = precision_recall_fscore_support(true_charge_cross,predictions_charge_cross,average='weighted')
                class_macro_precision2, class_macro_recall2, class_macro_f12,_ = precision_recall_fscore_support(true_charge_cross,predictions_charge_rnp,average='weighted')



                table = pt.PrettyTable(['types ','    toekn_P   ', '   toekn_R     ', '  toekn_F1   ' , '      F1     ' , '      percents     ' ]) 

                table.add_row(['dev-rnp',precision_tagging2[0], recall_tagging2[0], f1_tagging2[0], class_macro_f12, np.array(all_tag_preds_cross).sum()/len(all_tag_gold_cross) ])



                logger.info(table)


                ##############################  test  ##############################
                percents = 0
                predictions_charge_cross = []
                true_charge_cross = []

                predictions_charge_rnp = []

                all_tag_preds_cross = []
                all_tag_gold_cross = []

                all_tag_preds_rnp = []
                all_tag_gold_rnp = []


                for step,batch in enumerate(tqdm(test_dataloader)):

                    input_ids, attention_mask, label_ids, tag_ids, has_rationale, start_labels, end_labels, env_ids = batch
                    true_charge_cross.extend(label_ids.cpu().numpy())
                    input_ids = input_ids.to(args.device)
                    attention_mask = attention_mask.to(args.device)
                    tag_ids = tag_ids.to(args.device)
                    label_ids = label_ids.to(args.device)
                    with torch.no_grad():
                        output_bert_cross, rationale_mask_bert_cross, output_bert_rnp, rationale_mask_bert_rnp = model(input_ids,attention_mask,env_ids)

                    pred_cross = output_bert_cross.cpu().argmax(dim=1).numpy()
                    predictions_charge_cross.extend(pred_cross)

                    pred_rnp = output_bert_rnp.cpu().argmax(dim=1).numpy()
                    predictions_charge_rnp.extend(pred_rnp)

                    pred_tags_flat = [val for sublist in rationale_mask_bert_cross for val in sublist]
                    gold_tags_flat = torch.masked_select(tag_ids, attention_mask==1).tolist()
                    all_tag_preds_cross.extend(pred_tags_flat)
                    all_tag_gold_cross.extend(gold_tags_flat)
                    
                    assert len(gold_tags_flat) == len(pred_tags_flat)
                    pred_tags_flat = [val for sublist in rationale_mask_bert_rnp for val in sublist]
                    gold_tags_flat = torch.masked_select(tag_ids, attention_mask==1).tolist()
                    all_tag_preds_rnp.extend(pred_tags_flat)
                    all_tag_gold_rnp.extend(gold_tags_flat)

                    assert len(gold_tags_flat) == len(pred_tags_flat)

                test_eval = {}
                precision_tagging, recall_tagging, f1_tagging, _ = precision_recall_fscore_support(np.array(all_tag_gold_cross), np.array(all_tag_preds_cross), labels=[1])
                print(true_charge_cross)
                print(predictions_charge_cross)
                print(predictions_charge_rnp)
                precision_tagging2, recall_tagging2, f1_tagging2, _ = precision_recall_fscore_support(np.array(all_tag_gold_rnp), np.array(all_tag_preds_rnp), labels=[1])

                class_macro_precision, class_macro_recall, class_macro_f1, _  = precision_recall_fscore_support(true_charge_cross,predictions_charge_cross,average='weighted')
                class_macro_precision2, class_macro_recall2, class_macro_f12, _  = precision_recall_fscore_support(true_charge_cross,predictions_charge_rnp,average='weighted')


                table = pt.PrettyTable(['types ','    toekn_P   ', '   toekn_R     ', '  toekn_F1   ' ,   '      F1     ' , '      percents     ' ]) 

                table.add_row(['test-rnp',precision_tagging2[0], recall_tagging2[0], f1_tagging2[0],  class_macro_f12, np.array(all_tag_preds_cross).sum()/len(all_tag_gold_cross) ])


                logger.info(table)



if __name__ == "__main__":
    random_seed()
    criterion = nn.CrossEntropyLoss()
    main()

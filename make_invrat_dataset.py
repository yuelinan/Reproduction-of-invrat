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
from model import *
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
from data_utils import *
from kmeans_pytorch import kmeans
parser = argparse.ArgumentParser(description='VMask classificer')

parser.add_argument('--model_name', type=str, default='RNP', help='model name')
parser.add_argument('--lr', type=float, default=5e-5, help='initial learning rate')
parser.add_argument('--max_len', type=int, default=512, help='max_len')
parser.add_argument('--weight_decay', default=0, type=float, help='adding l2 regularization')
parser.add_argument('--dropout', type=float, default=0.2, help='the probability for dropout')
parser.add_argument('--alpha_rationle', type=float, default=0.2, help='alpha_rationle')
parser.add_argument('--hidden_dim', type=int, default=768, help='number of hidden dimension')
parser.add_argument("--activation", type=str, dest="activation", default="tanh", help='the choice of \
        non-linearity transfer function')
parser.add_argument('--gpu_id', default='0', type=str, help='gpu id')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--data_dir', type=str)
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


args = parser.parse_args()

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO,
                    )
logger = logging.getLogger(__name__)

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

# set fraction rationales
dataset_processor.set_fraction_rationales(args.fraction_rationales)

# get training examples 

train_examples = dataset_processor.get_train_examples(args.data_dir)


tag_map = dataset_processor.get_tag_map()
num_labels = dataset_processor.get_num_labels()
num_tags = dataset_processor.get_num_tags()
args.num_tags = num_tags



train_dataset = DatasetWithRationales(train_examples, tokenizer, tag_map, args.max_len, \
    args.dataset)

train_dataloader = data.DataLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=0,
    # pin_memory=True,
    collate_fn=DatasetWithRationales.pad,
    worker_init_fn=np.random.seed(args.seed),
)



for k, v in vars(args).items():
    logger.info("{:20} : {:10}".format(k, str(v)))



def make_shortcut():
    
    
    model = eval(args.model_name)(args)
    model.load_state_dict(torch.load(args.save_path))
    model = model.to(args.device)
    
    model.eval()

    ##############################  test  ##############################
    percents = 0
    predictions_charge = []
    true_charge = []

    all_tag_preds = []
    all_tag_gold = []
    all_env =  []
    for step,batch in enumerate(tqdm(train_dataloader)):
        # if step>10:break
        input_ids, attention_mask, label_ids, tag_ids, has_rationale, start_labels, end_labels, span_labels = batch
        
        true_charge.extend(label_ids.cpu().numpy())
        input_ids = input_ids.to(args.device)
        attention_mask = attention_mask.to(args.device)
        
        with torch.no_grad():
            output,pred_tags,h_env = model(input_ids,attention_mask)
        
        all_env.append(h_env)
        
    
    all_env_tensor = torch.cat(all_env,0)
    num_clusters = 2
    # kmeans
    cluster_ids_x, cluster_centers = kmeans(X=all_env_tensor, num_clusters=num_clusters, distance='euclidean',device = args.device)
    # print(cluster_ids_x)
    # print(cluster_centers)


    # generate the dataset
    f_json = open('./dataset/movie/'+args.dataset+'_invrat.json','a+')
    for step,batch in enumerate(tqdm(train_dataloader)):
        # if step>10:break
        input_ids, attention_mask, label_ids, tag_ids, has_rationale, start_labels, end_labels, span_labels = batch
        
        dataset_short = {}
        dataset_short['label_ids'] = [str(label_ids.cpu().numpy().tolist()[0])]
        
        dataset_short['input_ids'] = input_ids.cpu().numpy().tolist()
        dataset_short['tag_ids'] = tag_ids[0].cpu().numpy().tolist()
        dataset_short['env_id'] = cluster_ids_x[step].cpu().numpy().tolist()

        json_str = json.dumps(dataset_short, ensure_ascii=False)
        f_json.write(json_str)
        f_json.write('\n')


if __name__ == "__main__":
    random_seed()
    criterion = nn.CrossEntropyLoss()
    make_shortcut()

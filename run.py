# 拼接 prompt
# 输入到模型
# 输出 mask对应的hidden state

import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd

template = "[s1] is [MASK]"
def convert_prompt(sentence):
    s_new = template.replace("[s1]",sentence)
    return s_new

def output_hidden(inputs,tokenizer,model,device):
    inputs = convert_prompt(inputs)
    inputs = tokenizer(inputs,return_tensors="pt")
    masked_index = inputs["input_ids"].tolist()[0].index(mask_id)
    
    inputs = inputs.to(device)
    outputs = model(**inputs)
    cls = outputs["pooler_output"][0]
    last_hidden_state = outputs["last_hidden_state"].cpu()
    mask_hidden_state = last_hidden_state[0][masked_index].cpu()
    # print(masked_index)
    # print(cls.shape)
    # print(mask_hidden_state.shape)
    
    return cls,mask_hidden_state

def cal_cos(input1,input2):
    return torch.cosine_similarity(input1,input2,dim=0).item()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").to(device)
    mask_id = tokenizer.convert_tokens_to_ids("[MASK]")
    print(mask_id)

    cls_sims = []
    mask_sims = []
    labels = []
    with open("train.tsv","r") as f:
        lines = f.readlines()
        for i,line in enumerate(lines):
            line = line.split("\t")
            assert len(line)==10
            input1 = line[7]
            input2 = line[8]
            label = float(line[9])
            cls_state_1,mask_state_1 = output_hidden(input1,tokenizer,model,device)
            cls_state_2,mask_state_2 = output_hidden(input2,tokenizer,model,device)
            cls_sim = cal_cos(cls_state_1,cls_state_2)
            mask_sim = cal_cos(mask_state_1,mask_state_2)

            cls_sims.append(cls_sim)
            mask_sims.append(mask_sim)
            labels.append(label)

    

    #计算相关系数
    df = pd.DataFrame(list(zip(cls_sims,mask_sims,labels)), columns =['cls', 'mask', 'label'])
    print(df.shape)
    corr = df.corr('spearman')
    print(corr.shape)
    print("template: {}".format(template))
    print("cls to label:{}".format(corr.iat[2,0]))
    print("mask to label:{}".format(corr.iat[2,1]))
    #save
    cls_sims = np.asarray(cls_sims)
    mask_sims = np.asarray(mask_sims)
    labels = np.asarray(labels)
    np.save("cls_sims.npy",cls_sims)
    np.save("mask_sims.npy",mask_sims)
    np.save("labels.npy",labels)

    


    


    



    



# 拼接 prompt
# 输入到模型
# 输出 mask对应的hidden state

import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd

template = "[s1] is [MASK]"
template_list = ["[s1] [MASK]", "[s1] is [MASK]", "[s1] mean [MASK]", "[s1] means [MASK]",
                 "[s1] [MASK].", "[s1] is [MASK].", "[s1] mean [MASK].", "[s1] means [MASK].",
                 "This [s1] means [MASK].", "This sentence of [s1] means [MASK].",
                 "This sentence of \"[s1]\" means [MASK].", "This sentence: \"[s1]\" means [MASK]."]


def convert_prompt(sentence, template, with_mask):
    if not with_mask:
        s_new = sentence
    else:
        s_new = template.replace("[s1]", sentence)
    return s_new


def output_hidden(inputs, tokenizer, model, device, template):

    # with [mask] token, for MLM rep
    inputs_w_mask = convert_prompt(inputs, template, with_mask=True)
    inputs_w = tokenizer(inputs_w_mask, return_tensors="pt")
    masked_index = inputs_w["input_ids"].tolist()[0].index(mask_id)
    inputs_w = inputs_w.to(device)
    outputs_w = model(**inputs_w)

    last_hidden_state_w = outputs_w["last_hidden_state"].cpu()
    mask_hidden_state = last_hidden_state_w[0][masked_index].cpu()

    # without [MASK] token, for CLS and mean
    inputs_wo_mask = convert_prompt(inputs, template, with_mask=False)
    inputs_wo = tokenizer(inputs_wo_mask, return_tensors="pt")
    inputs_wo = inputs_wo.to(device)
    outputs_wo = model(**inputs_wo)

    cls = outputs_wo["pooler_output"][0]
    last_hidden_state_wo = outputs_wo["last_hidden_state"].cpu()
    #mask_hidden_state = last_hidden_state[0][masked_index].cpu()
    mean_pooled = torch.sum(last_hidden_state_wo[0], dim=0) / 768

    return cls, mask_hidden_state, mean_pooled


def cal_cos(input1, input2):
    return torch.cosine_similarity(input1, input2, dim=0).item()


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").to(device)
    mask_id = tokenizer.convert_tokens_to_ids("[MASK]")
    # print(mask_id)

    cls_sims = []
    mask_sims = []
    mean_sims = []
    labels = []
    for template in template_list:
        with open("train.tsv", "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.split("\t")
                assert len(line) == 10
                input1 = line[7]
                input2 = line[8]
                label = float(line[9])
                cls_state_1, mask_state_1, mean_state_1 = output_hidden(input1, tokenizer, model, device, template)
                cls_state_2, mask_state_2, mean_state_2 = output_hidden(input2, tokenizer, model, device, template)

                cls_sim = cal_cos(cls_state_1, cls_state_2)
                mask_sim = cal_cos(mask_state_1, mask_state_2)
                mean_sim = cal_cos(mean_state_1, mean_state_2)

                cls_sims.append(cls_sim)
                mask_sims.append(mask_sim)
                mean_sims.append(mean_sim)
                labels.append(label)

        # 计算相关系数
        df = pd.DataFrame(list(zip(cls_sims, mask_sims, mean_sims, labels)), columns=['cls', 'mask', 'mean', 'label'])
        #print(df.shape)
        corr = df.corr('spearman')
        #print(corr.shape)
        #print(corr)
        print("template: {}".format(template))
        print("cls to label:{}".format(corr.iat[3, 0]))
        print("mean to label:{}".format(corr.iat[3, 2]))
        print("mask to label:{}".format(corr.iat[3, 1]))

        # save
        '''
        cls_sims = np.asarray(cls_sims)
        mask_sims = np.asarray(mask_sims)
        labels = np.asarray(labels)
        np.save("cls_sims.npy", cls_sims)
        np.save("mask_sims.npy", mask_sims)
        np.save("labels.npy", labels)
        '''

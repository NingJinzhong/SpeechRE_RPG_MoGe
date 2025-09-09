from torchmetrics import Metric
import torch
import copy
import string
def process_string(s):
    # 转换为小写
    s = s.lower()
    # 去掉标点符号
    s = s.translate(str.maketrans('', '', string.punctuation))
    return s

class TripletF1(Metric):
    def __init__(self, hypernum,order_view):
        super().__init__()
        self.hypernum = hypernum
        self.tokenizer = self.hypernum.processor.tokenizer
        self.order_view = order_view
        self.add_state("glod_num", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("pred_num", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("predtrue_num", default=torch.tensor(0), dist_reduce_fx="sum")    
    def update(self,pred_triplets,triplet_labels):
        pred_triplets = [[process_string(s) for s in triplet] for triplet in pred_triplets]
        triplet_labels = [[process_string(s) for s in triplet] for triplet in triplet_labels]
        new_triple_list = []
        new_triple_label_list = []
        for pt in pred_triplets:
            new_triple_list.append([pt[0],pt[2],pt[3]])

        for tl in triplet_labels:
            new_triple_label_list.append([tl[0],tl[2],tl[3]])
        self.glod_num+=len(new_triple_label_list)
        self.pred_num+=len(new_triple_list)
        for pt in new_triple_list:
            if pt in new_triple_label_list:
                self.predtrue_num+=1
        
    def compute(self):
        P = self.predtrue_num.float()/(self.pred_num.float()+1e-8)
        R = self.predtrue_num.float()/(self.glod_num.float()+1e-8)
        return 2/(1/P+1/R)

class RelationF1(Metric):
    def __init__(self, hypernum,order_view):
        super().__init__()
        self.hypernum = hypernum
        self.tokenizer = self.hypernum.processor.tokenizer
        self.order_view = order_view
        self.add_state("glod_num", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("pred_num", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("predtrue_num", default=torch.tensor(0), dist_reduce_fx="sum")    
    def update(self,pred_triplets,triplet_labels):
        pred_triplets = [[process_string(s) for s in triplet] for triplet in pred_triplets]
        triplet_labels = [[process_string(s) for s in triplet] for triplet in triplet_labels]
        new_relation_list = []
        new_relation_label_list = []
        for pt in pred_triplets:
            new_relation_list.append(pt[2])

        for tl in triplet_labels:
            new_relation_label_list.append(tl[2])
        
        self.glod_num+=len(new_relation_label_list)
        self.pred_num+=len(new_relation_list)

        for pt in new_relation_list:
            if pt in new_relation_label_list:
                self.predtrue_num+=1
                
    def compute(self):
        P = self.predtrue_num.float()/(self.pred_num.float()+1e-8)
        R = self.predtrue_num.float()/(self.glod_num.float()+1e-8)
        return 2/(1/P+1/R)

class EncoderRelationF1(Metric):
    def __init__(self, hypernum,order_view):
        super().__init__()
        self.hypernum = hypernum
        self.tokenizer = self.hypernum.processor.tokenizer
        self.order_view = order_view
        self.add_state("glod_num", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("pred_num", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("predtrue_num", default=torch.tensor(0), dist_reduce_fx="sum")    
    def update(self,predicted_relations_id_list,relation_label_id_list):
        for pred_r_sample_list,label_r_sample_list in zip(predicted_relations_id_list,relation_label_id_list):
            pred_r_sample_list = [item for item in pred_r_sample_list if item !="<relation_type_padding_token>"]
            label_r_sample_list = [item for item in label_r_sample_list if item !="<relation_type_padding_token>"]
            self.glod_num+=len(label_r_sample_list)
            self.pred_num+=len(pred_r_sample_list)
            for pr in pred_r_sample_list:
                if pr in label_r_sample_list:
                    self.predtrue_num+=1    
                
    def compute(self):
        P = self.predtrue_num.float()/(self.pred_num.float()+1e-8)
        R = self.predtrue_num.float()/(self.glod_num.float()+1e-8)
        return 2/(1/P+1/R)
    
class EntityF1(Metric):
    def __init__(self, hypernum,order_view):
        super().__init__()
        self.hypernum = hypernum
        self.tokenizer = self.hypernum.processor.tokenizer
        self.order_view = order_view
        self.add_state("glod_num", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("pred_num", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("predtrue_num", default=torch.tensor(0), dist_reduce_fx="sum")    
    def update(self,pred_triplets,triplet_labels):
        pred_triplets = [[process_string(s) for s in triplet] for triplet in pred_triplets]
        triplet_labels = [[process_string(s) for s in triplet] for triplet in triplet_labels]
        new_entity_list = []
        new_entity_label_list = []
        for pt in pred_triplets:
            if pt[0] not in new_entity_list:
                new_entity_list.append(pt[0])
            if pt[3] not in new_entity_list:
                new_entity_list.append(pt[3])

        for tl in triplet_labels:
            if tl[0] not in new_entity_label_list:
                new_entity_label_list.append(tl[0])
            if tl[3] not in new_entity_label_list:
                new_entity_label_list.append(tl[3])
        
        self.glod_num+=len(new_entity_label_list)
        self.pred_num+=len(new_entity_list)

        for pt in new_entity_list:
            if pt in new_entity_label_list:
                self.predtrue_num+=1
        
    def compute(self):
        P = self.predtrue_num.float()/(self.pred_num.float()+1e-8)
        R = self.predtrue_num.float()/(self.glod_num.float()+1e-8)
        return 2/(1/P+1/R)
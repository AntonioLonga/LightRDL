
import argparse
import copy
import json
import math
import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch_frame import stype

from relbench.base import Dataset, EntityTask
from relbench.datasets import get_dataset
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task

import pandas as pd
from preprocessing.build_graph import build_graph_dnf,build_graph_position
import torch
import time



def main():
    parser = argparse.ArgumentParser(description="Official Code for ICLR submission GRAPH Construction")

    parser.add_argument("--dataset", type=str, default="rel-f1")
    parser.add_argument("--task", type=str, default="driver-dnf")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=os.path.expanduser("~/.cache/relbench_examples"),
    )
    args = parser.parse_args()
    
    if args.task == "driver-dnf":
        graph_builder = build_graph_dnf
    elif args.task == "driver-position":
        graph_builder = build_graph_position
    else:
        print("task not implemented yet!!!")
        assert False


    # get the database and task from relbench
    dataset: Dataset = get_dataset(args.dataset, download=True)
    task: EntityTask = get_task(args.dataset, args.task, download=True)
    stypes_cache_path = Path(f"{args.cache_dir}/{args.dataset}/stypes.json")

    try:
        with open(stypes_cache_path, "r") as f:
            col_to_stype_dict = json.load(f)
        for table, col_to_stype in col_to_stype_dict.items():
            for col, stype_str in col_to_stype.items():
                col_to_stype[col] = stype(stype_str)
    except FileNotFoundError:
        col_to_stype_dict = get_stype_proposal(dataset.get_db())
        Path(stypes_cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(stypes_cache_path, "w") as f:
            json.dump(col_to_stype_dict, f, indent=2, default=str)


    db = dataset.get_db()

    # load embedding of the features
    emb_cos_stan = pd.read_pickle("static_networks/f1/embeddings/emb_constructor_standings.pkl")
    emb_cos_res = pd.read_pickle("static_networks/f1/embeddings/emb_constructor_results.pkl")
    emb_cost = pd.read_pickle("static_networks/f1/embeddings/emb_constructors.pkl")
    emb_driv = pd.read_pickle("static_networks/f1/embeddings/emb_drivers.pkl")
    emb_stan = pd.read_pickle("static_networks/f1/embeddings/emb_standings.pkl")
    emb_qual = pd.read_pickle("static_networks/f1/embeddings/emb_qualifying.pkl")
    emb_resu = pd.read_pickle("static_networks/f1/embeddings/emb_results.pkl")
    emb_race = pd.read_pickle("static_networks/f1/embeddings/emb_races.pkl")
    emb_circ = pd.read_pickle("static_networks/f1/embeddings/emb_circuits.pkl")


    delta = task.timedelta
    test_table = task.get_table("test",mask_input_cols=False).df
    val_table = task.get_table("val",mask_input_cols=False).df
    train_table = task.get_table("train",mask_input_cols=False).df


    # load the distilled features from LightGBM
    emb_ligth_in = pd.read_csv("static_networks/f1/"+args.task+"/distilled_LIGHTGBM/lightgbm_emb.csv")
    emb_ligth = pd.DataFrame({
        'driverId': emb_ligth_in['driverId'],
        'date': emb_ligth_in['date'],
        'emb': emb_ligth_in[["0","1","2","3","4","5","6","7","8","9"]].values.tolist()  # List of 0-9 features
    })

    # compute and store test graphs
    count = len(np.flip(np.sort(test_table.date.unique())))
    i = 0
    for tt1 in np.flip(np.sort(test_table.date.unique())):
        t1 = pd.Timestamp("2009-11-01") - (delta * (i))
        t0 = pd.Timestamp("2009-11-01") - (delta * (5+i))
        
        current_embs = emb_ligth[emb_ligth.date == str(tt1)[:10]]
        curr_train_table = test_table[test_table.date == tt1]
        driverIds = curr_train_table.driverId.to_numpy()
        curr_train_table = curr_train_table.merge(current_embs,on="driverId")
        data = graph_builder(t0,t1,curr_train_table,driverIds,db,emb_cos_stan ,emb_cos_res,emb_cost, emb_driv,emb_stan, emb_qual, emb_resu,emb_race,emb_circ,delta)
        assert(data["drivers"].x.shape[0] == len(driverIds))
        torch.save(data, 'static_networks/f1/'+args.task+'/data_obj/data_obj_TEST_'+str(count-i)+'.pth')
        i = i + 1
        
        

    # compute and store validation graphs
    START_CONSTRUCTION = time.time()
    print("start building test graphs")
    count = len(np.flip(np.sort(val_table.date.unique())))
    i = 0
    for tt1 in np.flip(np.sort(val_table.date.unique())):
        t1 = pd.Timestamp("2007-11-01") - (delta * (i))
        t0 = pd.Timestamp("2007-11-01") - (delta * (5+i))    
        current_embs = emb_ligth[emb_ligth.date == str(tt1)[:10]]
        curr_train_table = val_table[val_table.date == tt1]
        curr_train_table = curr_train_table.merge(current_embs,on="driverId")
        driverIds = curr_train_table.driverId.to_numpy()
        data = graph_builder(t0,t1,curr_train_table,driverIds,db,emb_cos_stan ,emb_cos_res,emb_cost, emb_driv,emb_stan, emb_qual, emb_resu,emb_race,emb_circ,delta)
        assert(data["drivers"].x.shape[0] == len(driverIds))
        torch.save(data, 'static_networks/f1/'+args.task+'/data_obj/data_obj_VAL_'+str(count-i)+'.pth')
        i = i + 1

    print("start building validation graphs")  
    
    # compute and store train graphs   
    def my_validate(data):
        node_types = set(data.node_types)
        num_src_node_types = {src for src, _, _ in data.edge_types}
        num_dst_node_types = {dst for _, _, dst in data.edge_types}

        dangling_types = node_types - (num_src_node_types | num_dst_node_types)
        if len(dangling_types)==0:
            return True
        else:
            return False
    
    print("start building train graphs")     
    count = len(np.flip(np.sort(train_table.date.unique())))
    i = 0
    for tt1 in np.flip(np.sort(train_table.date.unique())):
        t1 = pd.Timestamp("2004-11-01") - (delta * (i))
        t0 = pd.Timestamp("2004-11-01") - (delta * (5+i))
        current_embs = emb_ligth[emb_ligth.date == str(tt1)[:10]]
        curr_train_table = train_table[train_table.date == tt1]
        curr_train_table = curr_train_table.merge(current_embs,on="driverId")
        driverIds = curr_train_table.driverId.to_numpy()
        data = graph_builder(t0,t1,curr_train_table,driverIds,db,emb_cos_stan ,emb_cos_res,emb_cost, emb_driv,emb_stan, emb_qual, emb_resu,emb_race,emb_circ,delta)
        assert(data["drivers"].x.shape[0] == len(driverIds))
        if my_validate(data):
            torch.save(data, 'static_networks/f1/'+args.task+'/data_obj/data_obj_TRAIN_'+str(count-i)+'.pth')
            i = i + 1
            
            
    print(f"graphs builded in {time.time()-START_CONSTRUCTION:.2f} seconds")
    

if __name__ == "__main__":
    main()
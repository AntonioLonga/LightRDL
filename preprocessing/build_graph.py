import numpy as np
from torch_geometric.data import HeteroData
import torch


def build_graph_position(t0,t1,curr_train_table,driverIds,db,emb_cos_stan ,emb_cos_res,emb_cost, emb_driv,emb_stan, emb_qual, emb_resu,emb_race,emb_circ,delta):

    # constructor_standings 
    table_name = "constructor_standings"
    df = db.table_dict[table_name].df.copy()
    filtered_constructor_standings = df[(df.date > (t0 + delta)) & (df.date <=(t1 + delta))].reset_index(drop=True)
    mappting_constructor_standings = {j:i for i,j in enumerate(filtered_constructor_standings.constructorStandingsId)}
    filtered_constructor_standings["constructorStandingsId_new"] = np.array([mappting_constructor_standings[i] for i in filtered_constructor_standings.constructorStandingsId])
    # to fix raceid, constructionid # done constructionID done raceid

    # constructor_results
    table_name = "constructor_results"
    df = db.table_dict[table_name].df.copy()
    filtered_constructor_results = df[(df.date > (t0 + delta)) & (df.date <=(t1 + delta))].reset_index(drop=True)
    mappting_constructor_results = {j:i for i,j in enumerate(filtered_constructor_results.constructorResultsId)}
    filtered_constructor_results["constructorResultsId_new"] = np.array([mappting_constructor_results[i] for i in filtered_constructor_results.constructorResultsId])
    # to fix raceid, constructionid # done constructionID don raceid

    # qualifying
    table_name = "qualifying"
    df = db.table_dict[table_name].df.copy()
    filtered_qualifying = df[(df.date > (t0 + delta)) & (df.date <=(t1 + delta))]
    filtered_qualifying = filtered_qualifying[filtered_qualifying.driverId.isin(driverIds)].reset_index(drop=True)
    mappting_qualifying = {j:i for i,j in enumerate(filtered_qualifying.qualifyId)}
    filtered_qualifying["qualifyId_new"] = np.array([mappting_qualifying[i] for i in filtered_qualifying.qualifyId])
    # in qulifying fix raceid driverid constructorID # done constructorID # done raceid

    # result
    table_name = "results"
    df = db.table_dict[table_name].df.copy()
    filtered_results = df[(df.date > (t0 + delta)) & (df.date <=(t1 + delta))]
    filtered_results = filtered_results[filtered_results.driverId.isin(driverIds)].reset_index(drop=True)
    mappting_results = {j:i for i,j in enumerate(filtered_results.resultId)}
    filtered_results["resultId_new"] = np.array([mappting_results[i] for i in filtered_results.resultId])
    # in results fix raceid driverid constructorID # done constructorID # done raceid

    # constructions
    table_name = "constructors"
    df = db.table_dict[table_name].df
    a = filtered_results.constructorId.to_numpy()
    b = filtered_qualifying.constructorId.to_numpy()
    c = filtered_constructor_results.constructorId.to_numpy()
    d = filtered_constructor_standings.constructorId.to_numpy()
    constructorIds = np.unique(np.concatenate([a,b,c,d]))
    filtered_constructor = df[df['constructorId'].isin(constructorIds)].reset_index(drop=True)
    mappting_constructor = {j:i for i,j in enumerate(filtered_constructor.constructorId)}
    filtered_constructor["constructorId_new"] = np.array([mappting_constructor[i] for i in filtered_constructor.constructorId])
    # fix constructorID in result
    filtered_results["constructorId_new"] = np.array([mappting_constructor[i] for i in filtered_results.constructorId])
    # fix constructorID in qualifying
    filtered_qualifying["constructorId_new"] = np.array([mappting_constructor[i] for i in filtered_qualifying.constructorId])
    # fix constructorID in result
    filtered_constructor_standings["constructorId_new"] = np.array([mappting_constructor[i] for i in filtered_constructor_standings.constructorId])
    # fix constructorID in qualifying
    filtered_constructor_results["constructorId_new"] = np.array([mappting_constructor[i] for i in filtered_constructor_results.constructorId])

    # standings
    table_name = "standings"
    df = db.table_dict[table_name].df
    filtered_standings =df[(df.date > (t0 + delta)) & (df.date <=(t1 + delta))].reset_index(drop=True)
    filtered_standings = filtered_standings[filtered_standings.driverId.isin(driverIds)].reset_index(drop=True)
    mappting_standings = {j:i for i,j in enumerate(filtered_standings.driverStandingsId)}
    filtered_standings["driverStandingsId_new"] = np.array([mappting_standings[i] for i in filtered_standings.driverStandingsId])
    # in standings fix raceid driverid # done races


    # filter race
    table_name = "races"
    df = db.table_dict[table_name].df
    df = df[(df.date > (t0 + delta)) & (df.date <=(t1 + delta))]
    a = filtered_results.raceId.to_numpy()
    b = filtered_qualifying.raceId.to_numpy()
    c = filtered_constructor_results.raceId.to_numpy()
    d = filtered_constructor_standings.raceId.to_numpy()
    f = filtered_standings.raceId.to_numpy()
    racesIds = np.unique(np.concatenate([a,b,c,d,f]))
    filtered_races = df[df['raceId'].isin(racesIds)].reset_index(drop=True)
    mappting_races = {j:i for i,j in enumerate(filtered_races.raceId)}
    filtered_races["raceId_new"] = np.array([mappting_races[i] for i in filtered_races.raceId])
    # fix racesID in result
    filtered_results["raceId_new"] = np.array([mappting_races[i] for i in filtered_results.raceId])
    # fix racesID in qualifying
    filtered_qualifying = filtered_qualifying[filtered_qualifying.raceId.isin(filtered_races.raceId)].reset_index(drop=True)

    filtered_qualifying["raceId_new"] = np.array([mappting_races[i] for i in filtered_qualifying.raceId])
    # fix racesID in standings
    filtered_standings["raceId_new"] = np.array([mappting_races[i] for i in filtered_standings.raceId])
    # fix racesID in constructor_standings
    filtered_constructor_standings["raceId_new"] = np.array([mappting_races[i] for i in filtered_constructor_standings.raceId])
    # fix racesID in constructor_results
    filtered_constructor_results["raceId_new"] = np.array([mappting_races[i] for i in filtered_constructor_results.raceId])
    # to fix circuitId # done circuitId

    # circuits 
    table_name = "circuits"
    df = db.table_dict[table_name].df
    filtered_circuits = df[df['circuitId'].isin(filtered_races.circuitId)].reset_index(drop=True)
    mappting_circuits = {j:i for i,j in enumerate(filtered_circuits.circuitId)}
    filtered_circuits["circuitId_new"] = np.array([mappting_circuits[i] for i in filtered_circuits.circuitId])
    # fix circuitid in races
    filtered_races["circuitId_new"] = np.array([mappting_circuits[i] for i in filtered_races.circuitId])

    # drivers
    table_name = "drivers"
    df = db.table_dict[table_name].df
    filtered_drivers = df[df['driverId'].isin(driverIds)].reset_index(drop=True)
    mappting_drivers = {j:i for i,j in enumerate(filtered_drivers.driverId)}
    filtered_drivers["driverId_new"] = np.array([mappting_drivers[i] for i in filtered_drivers.driverId])
    # fix driverId in qualifying
    filtered_qualifying["driverId_new"] = np.array([mappting_drivers[i] for i in filtered_qualifying.driverId])
    # fix driverId in standings
    filtered_standings["driverId_new"] = np.array([mappting_drivers[i] for i in filtered_standings.driverId])
    # fix driverId in results
    filtered_results["driverId_new"] = np.array([mappting_drivers[i] for i in filtered_results.driverId])

    # add y to driver
    filtered_drivers =filtered_drivers.merge(curr_train_table, on="driverId")



    data = HeteroData()
    tmp = emb_driv[emb_driv.driverId.isin(filtered_drivers.driverId)]
    tmp = tmp.set_index('driverId').loc[filtered_drivers.driverId].reset_index().emb.to_numpy().tolist()
    tmp = np.array(tmp)
    tmp = np.concatenate([tmp,np.array(filtered_drivers.emb.to_numpy().tolist())],axis=1)


    if len(tmp) > 0:
        
        data["drivers"].x = torch.tensor(tmp, dtype=torch.float) 
        data["drivers"].y = torch.tensor(filtered_drivers.position.to_numpy(), dtype=torch.long)

        tmp = emb_race[emb_race.raceId.isin(filtered_races.raceId)]
        tmp = tmp.set_index('raceId').loc[filtered_races.raceId].reset_index().emb.to_numpy().tolist()
        if len(tmp) > 0:
            data["races"].x = torch.tensor(tmp, dtype=torch.float)
            


        tmp = emb_cos_stan[emb_cos_stan.constructorStandingsId.isin(filtered_constructor_standings.constructorStandingsId)]
        tmp = tmp.set_index('constructorStandingsId').loc[filtered_constructor_standings.constructorStandingsId].reset_index().emb.to_numpy().tolist()
        if len(tmp) > 0:
            data["constructor_standings"].x = torch.tensor(tmp, dtype=torch.float) 



        tmp = emb_stan[emb_stan.driverStandingsId.isin(filtered_standings.driverStandingsId)]
        tmp = tmp.set_index('driverStandingsId').loc[filtered_standings.driverStandingsId].reset_index().emb.to_numpy().tolist()
        if len(tmp) > 0:
            data["standings"].x = torch.tensor(tmp, dtype=torch.float) 



        tmp = emb_cost[emb_cost.constructorId.isin(filtered_constructor.constructorId)]
        tmp = tmp.set_index('constructorId').loc[filtered_constructor.constructorId].reset_index().emb.to_numpy().tolist()
        if len(tmp) > 0:
            data["constructors"].x = torch.tensor(tmp, dtype=torch.float) 



        tmp = emb_cos_res[emb_cos_res.constructorResultsId.isin(filtered_constructor_results.constructorResultsId)]
        tmp = tmp.set_index('constructorResultsId').loc[filtered_constructor_results.constructorResultsId].reset_index().emb.to_numpy().tolist()
        if len(tmp) > 0:
            data["constructor_results"].x = torch.tensor(tmp, dtype=torch.float) 



        tmp = emb_resu[emb_resu.resultId.isin(filtered_results.resultId)]
        tmp = tmp.set_index('resultId').loc[filtered_results.resultId].reset_index().emb.to_numpy().tolist()
        if len(tmp) > 0:
            data["results"].x = torch.tensor(tmp, dtype=torch.float) 



        tmp = emb_qual[emb_qual.qualifyId.isin(filtered_qualifying.qualifyId)]
        tmp = tmp.set_index('qualifyId').loc[filtered_qualifying.qualifyId].reset_index().emb.to_numpy().tolist()
        if len(tmp) > 0:
            data["qualifying"].x = torch.tensor(tmp, dtype=torch.float) 



        tmp = emb_circ[emb_circ.circuitId.isin(filtered_circuits.circuitId)]
        tmp = tmp.set_index('circuitId').loc[filtered_circuits.circuitId].reset_index().emb.to_numpy().tolist()
        if len(tmp) > 0:
            data["circuits"].x = torch.tensor(tmp, dtype=torch.float) 


        if "standings" in data.node_types:
            if "races" in data.node_types:
                data['standings', 'r1', 'races'].edge_index = torch.tensor(filtered_standings[["driverStandingsId_new","raceId_new"]].to_numpy().T, dtype=torch.long)
                data['races', 'rev_r1', 'standings'].edge_index = torch.tensor(filtered_standings[["raceId_new","driverStandingsId_new"]].to_numpy().T, dtype=torch.long)
            if "drivers" in data.node_types:
                data['standings', 'r2', 'drivers'].edge_index = torch.tensor(filtered_standings[["driverStandingsId_new","driverId_new"]].to_numpy().T, dtype=torch.long)
                data['drivers', 'rev_r2', 'standings'].edge_index = torch.tensor(filtered_standings[["driverId_new","driverStandingsId_new"]].to_numpy().T, dtype=torch.long)

        if "results" in data.node_types:
            if "drivers" in data.node_types:
                data['results', 'r3', 'drivers'].edge_index = torch.tensor(filtered_results[["resultId_new","driverId_new"]].to_numpy().T, dtype=torch.long)
                data['drivers', 'rev_r3', 'results'].edge_index = torch.tensor(filtered_results[["driverId_new","resultId_new"]].to_numpy().T, dtype=torch.long)
            if "races" in data.node_types:
                data['results', 'r4', 'races'].edge_index = torch.tensor(filtered_results[["resultId_new","raceId_new"]].to_numpy().T, dtype=torch.long)
                data['races', 'rev_r4', 'results'].edge_index = torch.tensor(filtered_results[["raceId_new","resultId_new"]].to_numpy().T, dtype=torch.long)
            if "constructors" in data.node_types:
                data['results', 'r5', 'constructors'].edge_index = torch.tensor(filtered_results[["resultId_new","constructorId_new"]].to_numpy().T, dtype=torch.long)
                data['constructors', 'rev_r5', 'results'].edge_index = torch.tensor(filtered_results[["constructorId_new","resultId_new"]].to_numpy().T, dtype=torch.long)

        if "qualifying" in data.node_types:
            if "drivers" in data.node_types:
                data['qualifying', 'r6', 'drivers'].edge_index = torch.tensor(filtered_qualifying[["qualifyId_new","driverId_new"]].to_numpy().T, dtype=torch.long)
                data['drivers', 'rev_r6', 'qualifying'].edge_index = torch.tensor(filtered_qualifying[["driverId_new","qualifyId_new"]].to_numpy().T, dtype=torch.long)
            if "races" in data.node_types:        
                data['qualifying', 'r7', 'races'].edge_index = torch.tensor(filtered_qualifying[["qualifyId_new","raceId_new"]].to_numpy().T, dtype=torch.long)
                data['races', 'rev_r7', 'qualifying'].edge_index = torch.tensor(filtered_qualifying[["raceId_new","qualifyId_new"]].to_numpy().T, dtype=torch.long)
            if "constructors" in data.node_types:
                data['qualifying', 'r8', 'constructors'].edge_index = torch.tensor(filtered_qualifying[["qualifyId_new","constructorId_new"]].to_numpy().T, dtype=torch.long)
                data['constructors', 'rev_r8', 'qualifying'].edge_index = torch.tensor(filtered_qualifying[["constructorId_new","qualifyId_new"]].to_numpy().T, dtype=torch.long)


        if "races" in data.node_types:
            if "circuits" in data.node_types:
                data['races', 'r9', 'circuits'].edge_index = torch.tensor(filtered_races[["raceId_new","circuitId_new"]].to_numpy().T, dtype=torch.long)
                data['circuits', 'rev_r9', 'races'].edge_index = torch.tensor(filtered_races[["circuitId_new","raceId_new"]].to_numpy().T, dtype=torch.long)

        if "constructor_results" in data.node_types:
            if "constructors" in data.node_types:
                data['constructor_results', 'r10', 'constructors'].edge_index = torch.tensor(filtered_constructor_results[["constructorResultsId_new","constructorId_new"]].to_numpy().T, dtype=torch.long)
                data['constructors', 'rev_r10', 'constructor_results'].edge_index = torch.tensor(filtered_constructor_results[["constructorId_new","constructorResultsId_new"]].to_numpy().T, dtype=torch.long)
            if "races" in data.node_types:
                data['constructor_results', 'r11', 'races'].edge_index = torch.tensor(filtered_constructor_results[["constructorResultsId_new","raceId_new"]].to_numpy().T, dtype=torch.long)
                data['races', 'rev_r11', 'constructor_results'].edge_index = torch.tensor(filtered_constructor_results[["raceId_new","constructorResultsId_new"]].to_numpy().T, dtype=torch.long)

        if "constructor_standings" in data.node_types:
            if "constructors" in data.node_types:
                data['constructor_standings', 'r10', 'constructors'].edge_index = torch.tensor(filtered_constructor_standings[["constructorStandingsId_new","constructorId_new"]].to_numpy().T, dtype=torch.long)
                data['constructors', 'rev_r10', 'constructor_standings'].edge_index = torch.tensor(filtered_constructor_standings[["constructorId_new","constructorStandingsId_new"]].to_numpy().T, dtype=torch.long)
            if "races" in data.node_types:
                data['constructor_standings', 'r11', 'races'].edge_index = torch.tensor(filtered_constructor_standings[["constructorStandingsId_new","raceId_new"]].to_numpy().T, dtype=torch.long)
                data['races', 'rev_r11', 'constructor_standings'].edge_index = torch.tensor(filtered_constructor_standings[["raceId_new","constructorStandingsId_new"]].to_numpy().T, dtype=torch.long)
        

    else:
        data = -1
        
    return data

def build_graph_dnf(t0,t1,curr_train_table,driverIds,db,emb_cos_stan ,emb_cos_res,emb_cost, emb_driv,emb_stan, emb_qual, emb_resu,emb_race,emb_circ,delta):

    # constructor_standings 
    table_name = "constructor_standings"
    df = db.table_dict[table_name].df.copy()
    filtered_constructor_standings = df[(df.date > (t0 + delta)) & (df.date <=(t1 + delta))].reset_index(drop=True)
    mappting_constructor_standings = {j:i for i,j in enumerate(filtered_constructor_standings.constructorStandingsId)}
    filtered_constructor_standings["constructorStandingsId_new"] = np.array([mappting_constructor_standings[i] for i in filtered_constructor_standings.constructorStandingsId])
    # to fix raceid, constructionid # done constructionID done raceid

    # constructor_results
    table_name = "constructor_results"
    df = db.table_dict[table_name].df.copy()
    filtered_constructor_results = df[(df.date > (t0 + delta)) & (df.date <=(t1 + delta))].reset_index(drop=True)
    mappting_constructor_results = {j:i for i,j in enumerate(filtered_constructor_results.constructorResultsId)}
    filtered_constructor_results["constructorResultsId_new"] = np.array([mappting_constructor_results[i] for i in filtered_constructor_results.constructorResultsId])
    # to fix raceid, constructionid # done constructionID don raceid

    # qualifying
    table_name = "qualifying"
    df = db.table_dict[table_name].df.copy()
    filtered_qualifying = df[(df.date > (t0 + delta)) & (df.date <=(t1 + delta))]
    filtered_qualifying = filtered_qualifying[filtered_qualifying.driverId.isin(driverIds)].reset_index(drop=True)
    mappting_qualifying = {j:i for i,j in enumerate(filtered_qualifying.qualifyId)}
    filtered_qualifying["qualifyId_new"] = np.array([mappting_qualifying[i] for i in filtered_qualifying.qualifyId])
    # in qulifying fix raceid driverid constructorID # done constructorID # done raceid

    # result
    table_name = "results"
    df = db.table_dict[table_name].df.copy()
    filtered_results = df[(df.date > (t0 + delta)) & (df.date <=(t1 + delta))]
    filtered_results = filtered_results[filtered_results.driverId.isin(driverIds)].reset_index(drop=True)
    mappting_results = {j:i for i,j in enumerate(filtered_results.resultId)}
    filtered_results["resultId_new"] = np.array([mappting_results[i] for i in filtered_results.resultId])
    # in results fix raceid driverid constructorID # done constructorID # done raceid

    # constructions
    table_name = "constructors"
    df = db.table_dict[table_name].df
    a = filtered_results.constructorId.to_numpy()
    b = filtered_qualifying.constructorId.to_numpy()
    c = filtered_constructor_results.constructorId.to_numpy()
    d = filtered_constructor_standings.constructorId.to_numpy()
    constructorIds = np.unique(np.concatenate([a,b,c,d]))
    filtered_constructor = df[df['constructorId'].isin(constructorIds)].reset_index(drop=True)
    mappting_constructor = {j:i for i,j in enumerate(filtered_constructor.constructorId)}
    filtered_constructor["constructorId_new"] = np.array([mappting_constructor[i] for i in filtered_constructor.constructorId])
    # fix constructorID in result
    filtered_results["constructorId_new"] = np.array([mappting_constructor[i] for i in filtered_results.constructorId])
    # fix constructorID in qualifying
    filtered_qualifying["constructorId_new"] = np.array([mappting_constructor[i] for i in filtered_qualifying.constructorId])
    # fix constructorID in result
    filtered_constructor_standings["constructorId_new"] = np.array([mappting_constructor[i] for i in filtered_constructor_standings.constructorId])
    # fix constructorID in qualifying
    filtered_constructor_results["constructorId_new"] = np.array([mappting_constructor[i] for i in filtered_constructor_results.constructorId])

    # standings
    table_name = "standings"
    df = db.table_dict[table_name].df
    filtered_standings =df[(df.date > (t0 + delta)) & (df.date <=(t1 + delta))].reset_index(drop=True)
    filtered_standings = filtered_standings[filtered_standings.driverId.isin(driverIds)].reset_index(drop=True)
    mappting_standings = {j:i for i,j in enumerate(filtered_standings.driverStandingsId)}
    filtered_standings["driverStandingsId_new"] = np.array([mappting_standings[i] for i in filtered_standings.driverStandingsId])
    # in standings fix raceid driverid # done races


    # filter race
    table_name = "races"
    df = db.table_dict[table_name].df
    df = df[(df.date > (t0 + delta)) & (df.date <=(t1 + delta))]
    a = filtered_results.raceId.to_numpy()
    b = filtered_qualifying.raceId.to_numpy()
    c = filtered_constructor_results.raceId.to_numpy()
    d = filtered_constructor_standings.raceId.to_numpy()
    f = filtered_standings.raceId.to_numpy()
    racesIds = np.unique(np.concatenate([a,b,c,d,f]))
    filtered_races = df[df['raceId'].isin(racesIds)].reset_index(drop=True)
    mappting_races = {j:i for i,j in enumerate(filtered_races.raceId)}
    filtered_races["raceId_new"] = np.array([mappting_races[i] for i in filtered_races.raceId])
    # fix racesID in result
    filtered_results["raceId_new"] = np.array([mappting_races[i] for i in filtered_results.raceId])
    # fix racesID in qualifying
    filtered_qualifying = filtered_qualifying[filtered_qualifying.raceId.isin(filtered_races.raceId)].reset_index(drop=True)

    filtered_qualifying["raceId_new"] = np.array([mappting_races[i] for i in filtered_qualifying.raceId])
    # fix racesID in standings
    filtered_standings["raceId_new"] = np.array([mappting_races[i] for i in filtered_standings.raceId])
    # fix racesID in constructor_standings
    filtered_constructor_standings["raceId_new"] = np.array([mappting_races[i] for i in filtered_constructor_standings.raceId])
    # fix racesID in constructor_results
    filtered_constructor_results["raceId_new"] = np.array([mappting_races[i] for i in filtered_constructor_results.raceId])
    # to fix circuitId # done circuitId

    # circuits 
    table_name = "circuits"
    df = db.table_dict[table_name].df
    filtered_circuits = df[df['circuitId'].isin(filtered_races.circuitId)].reset_index(drop=True)
    mappting_circuits = {j:i for i,j in enumerate(filtered_circuits.circuitId)}
    filtered_circuits["circuitId_new"] = np.array([mappting_circuits[i] for i in filtered_circuits.circuitId])
    # fix circuitid in races
    filtered_races["circuitId_new"] = np.array([mappting_circuits[i] for i in filtered_races.circuitId])

    # drivers
    table_name = "drivers"
    df = db.table_dict[table_name].df
    filtered_drivers = df[df['driverId'].isin(driverIds)].reset_index(drop=True)
    mappting_drivers = {j:i for i,j in enumerate(filtered_drivers.driverId)}
    filtered_drivers["driverId_new"] = np.array([mappting_drivers[i] for i in filtered_drivers.driverId])
    # fix driverId in qualifying
    filtered_qualifying["driverId_new"] = np.array([mappting_drivers[i] for i in filtered_qualifying.driverId])
    # fix driverId in standings
    filtered_standings["driverId_new"] = np.array([mappting_drivers[i] for i in filtered_standings.driverId])
    # fix driverId in results
    filtered_results["driverId_new"] = np.array([mappting_drivers[i] for i in filtered_results.driverId])

    # add y to driver
    filtered_drivers =filtered_drivers.merge(curr_train_table, on="driverId")



    data = HeteroData()
    tmp = emb_driv[emb_driv.driverId.isin(filtered_drivers.driverId)]
    tmp = tmp.set_index('driverId').loc[filtered_drivers.driverId].reset_index().emb.to_numpy().tolist()
    tmp = np.array(tmp)
    tmp = np.concatenate([tmp,np.array(filtered_drivers.emb.to_numpy().tolist())],axis=1)


    if len(tmp) > 0:
        
        data["drivers"].x = torch.tensor(tmp, dtype=torch.float) 
        data["drivers"].y = torch.tensor(filtered_drivers.did_not_finish.to_numpy(), dtype=torch.long)

        tmp = emb_race[emb_race.raceId.isin(filtered_races.raceId)]
        tmp = tmp.set_index('raceId').loc[filtered_races.raceId].reset_index().emb.to_numpy().tolist()
        if len(tmp) > 0:
            data["races"].x = torch.tensor(tmp, dtype=torch.float)
            


        tmp = emb_cos_stan[emb_cos_stan.constructorStandingsId.isin(filtered_constructor_standings.constructorStandingsId)]
        tmp = tmp.set_index('constructorStandingsId').loc[filtered_constructor_standings.constructorStandingsId].reset_index().emb.to_numpy().tolist()
        if len(tmp) > 0:
            data["constructor_standings"].x = torch.tensor(tmp, dtype=torch.float) 



        tmp = emb_stan[emb_stan.driverStandingsId.isin(filtered_standings.driverStandingsId)]
        tmp = tmp.set_index('driverStandingsId').loc[filtered_standings.driverStandingsId].reset_index().emb.to_numpy().tolist()
        if len(tmp) > 0:
            data["standings"].x = torch.tensor(tmp, dtype=torch.float) 



        tmp = emb_cost[emb_cost.constructorId.isin(filtered_constructor.constructorId)]
        tmp = tmp.set_index('constructorId').loc[filtered_constructor.constructorId].reset_index().emb.to_numpy().tolist()
        if len(tmp) > 0:
            data["constructors"].x = torch.tensor(tmp, dtype=torch.float) 



        tmp = emb_cos_res[emb_cos_res.constructorResultsId.isin(filtered_constructor_results.constructorResultsId)]
        tmp = tmp.set_index('constructorResultsId').loc[filtered_constructor_results.constructorResultsId].reset_index().emb.to_numpy().tolist()
        if len(tmp) > 0:
            data["constructor_results"].x = torch.tensor(tmp, dtype=torch.float) 



        tmp = emb_resu[emb_resu.resultId.isin(filtered_results.resultId)]
        tmp = tmp.set_index('resultId').loc[filtered_results.resultId].reset_index().emb.to_numpy().tolist()
        if len(tmp) > 0:
            data["results"].x = torch.tensor(tmp, dtype=torch.float) 



        tmp = emb_qual[emb_qual.qualifyId.isin(filtered_qualifying.qualifyId)]
        tmp = tmp.set_index('qualifyId').loc[filtered_qualifying.qualifyId].reset_index().emb.to_numpy().tolist()
        if len(tmp) > 0:
            data["qualifying"].x = torch.tensor(tmp, dtype=torch.float) 



        tmp = emb_circ[emb_circ.circuitId.isin(filtered_circuits.circuitId)]
        tmp = tmp.set_index('circuitId').loc[filtered_circuits.circuitId].reset_index().emb.to_numpy().tolist()
        if len(tmp) > 0:
            data["circuits"].x = torch.tensor(tmp, dtype=torch.float) 


        if "standings" in data.node_types:
            if "races" in data.node_types:
                data['standings', 'r1', 'races'].edge_index = torch.tensor(filtered_standings[["driverStandingsId_new","raceId_new"]].to_numpy().T, dtype=torch.long)
                data['races', 'rev_r1', 'standings'].edge_index = torch.tensor(filtered_standings[["raceId_new","driverStandingsId_new"]].to_numpy().T, dtype=torch.long)
            if "drivers" in data.node_types:
                data['standings', 'r2', 'drivers'].edge_index = torch.tensor(filtered_standings[["driverStandingsId_new","driverId_new"]].to_numpy().T, dtype=torch.long)
                data['drivers', 'rev_r2', 'standings'].edge_index = torch.tensor(filtered_standings[["driverId_new","driverStandingsId_new"]].to_numpy().T, dtype=torch.long)

        if "results" in data.node_types:
            if "drivers" in data.node_types:
                data['results', 'r3', 'drivers'].edge_index = torch.tensor(filtered_results[["resultId_new","driverId_new"]].to_numpy().T, dtype=torch.long)
                data['drivers', 'rev_r3', 'results'].edge_index = torch.tensor(filtered_results[["driverId_new","resultId_new"]].to_numpy().T, dtype=torch.long)
            if "races" in data.node_types:
                data['results', 'r4', 'races'].edge_index = torch.tensor(filtered_results[["resultId_new","raceId_new"]].to_numpy().T, dtype=torch.long)
                data['races', 'rev_r4', 'results'].edge_index = torch.tensor(filtered_results[["raceId_new","resultId_new"]].to_numpy().T, dtype=torch.long)
            if "constructors" in data.node_types:
                data['results', 'r5', 'constructors'].edge_index = torch.tensor(filtered_results[["resultId_new","constructorId_new"]].to_numpy().T, dtype=torch.long)
                data['constructors', 'rev_r5', 'results'].edge_index = torch.tensor(filtered_results[["constructorId_new","resultId_new"]].to_numpy().T, dtype=torch.long)

        if "qualifying" in data.node_types:
            if "drivers" in data.node_types:
                data['qualifying', 'r6', 'drivers'].edge_index = torch.tensor(filtered_qualifying[["qualifyId_new","driverId_new"]].to_numpy().T, dtype=torch.long)
                data['drivers', 'rev_r6', 'qualifying'].edge_index = torch.tensor(filtered_qualifying[["driverId_new","qualifyId_new"]].to_numpy().T, dtype=torch.long)
            if "races" in data.node_types:        
                data['qualifying', 'r7', 'races'].edge_index = torch.tensor(filtered_qualifying[["qualifyId_new","raceId_new"]].to_numpy().T, dtype=torch.long)
                data['races', 'rev_r7', 'qualifying'].edge_index = torch.tensor(filtered_qualifying[["raceId_new","qualifyId_new"]].to_numpy().T, dtype=torch.long)
            if "constructors" in data.node_types:
                data['qualifying', 'r8', 'constructors'].edge_index = torch.tensor(filtered_qualifying[["qualifyId_new","constructorId_new"]].to_numpy().T, dtype=torch.long)
                data['constructors', 'rev_r8', 'qualifying'].edge_index = torch.tensor(filtered_qualifying[["constructorId_new","qualifyId_new"]].to_numpy().T, dtype=torch.long)


        if "races" in data.node_types:
            if "circuits" in data.node_types:
                data['races', 'r9', 'circuits'].edge_index = torch.tensor(filtered_races[["raceId_new","circuitId_new"]].to_numpy().T, dtype=torch.long)
                data['circuits', 'rev_r9', 'races'].edge_index = torch.tensor(filtered_races[["circuitId_new","raceId_new"]].to_numpy().T, dtype=torch.long)

        if "constructor_results" in data.node_types:
            if "constructors" in data.node_types:
                data['constructor_results', 'r10', 'constructors'].edge_index = torch.tensor(filtered_constructor_results[["constructorResultsId_new","constructorId_new"]].to_numpy().T, dtype=torch.long)
                data['constructors', 'rev_r10', 'constructor_results'].edge_index = torch.tensor(filtered_constructor_results[["constructorId_new","constructorResultsId_new"]].to_numpy().T, dtype=torch.long)
            if "races" in data.node_types:
                data['constructor_results', 'r11', 'races'].edge_index = torch.tensor(filtered_constructor_results[["constructorResultsId_new","raceId_new"]].to_numpy().T, dtype=torch.long)
                data['races', 'rev_r11', 'constructor_results'].edge_index = torch.tensor(filtered_constructor_results[["raceId_new","constructorResultsId_new"]].to_numpy().T, dtype=torch.long)

        if "constructor_standings" in data.node_types:
            if "constructors" in data.node_types:
                data['constructor_standings', 'r10', 'constructors'].edge_index = torch.tensor(filtered_constructor_standings[["constructorStandingsId_new","constructorId_new"]].to_numpy().T, dtype=torch.long)
                data['constructors', 'rev_r10', 'constructor_standings'].edge_index = torch.tensor(filtered_constructor_standings[["constructorId_new","constructorStandingsId_new"]].to_numpy().T, dtype=torch.long)
            if "races" in data.node_types:
                data['constructor_standings', 'r11', 'races'].edge_index = torch.tensor(filtered_constructor_standings[["constructorStandingsId_new","raceId_new"]].to_numpy().T, dtype=torch.long)
                data['races', 'rev_r11', 'constructor_standings'].edge_index = torch.tensor(filtered_constructor_standings[["raceId_new","constructorStandingsId_new"]].to_numpy().T, dtype=torch.long)

        

    else:
        data = -1
        
    return data




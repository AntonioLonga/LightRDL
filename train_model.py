from nn.model import LightRDL, evaluate, train
from preprocessing.utils import load_graphs
from torch_geometric.loader import DataLoader
import torch
from sklearn.metrics import roc_auc_score
import argparse



def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Official Code for ICML submission TRAIN model")

    # Add arguments
    parser.add_argument('--dataset', type=str, help="rel-bench dataset name", default="rel-f1")
    parser.add_argument('--task', type=str, help="rel-bench task name", default="driver-dnf")
    parser.add_argument('--batch_size', type=int, help="Batch size",default=1)
    parser.add_argument('--num_layers', type=int, help="number of layer for our model",default=3)
    parser.add_argument('--dropout_prob', type=float, help="dropout",default=0.1)
    parser.add_argument('--lr', type=float, help="learning rate",default=0.0001)
    parser.add_argument('--hidden_channels', type=int, help="hidden channels dim.",default=64)
    parser.add_argument('--weight_decay', type=float, help="weight decay",default=0.000001)
    parser.add_argument('--device', type=str, help="device",default="cuda")
    parser.add_argument('--compute_val_every', type=int, help="Evaluate validation every X epochs",default=10)
    parser.add_argument('--patience', type=int, help="Earlystopping patiente",default=50)
    parser.add_argument('--nb_epochs', type=int, help="number of epochs",default=1000)
    parser.add_argument('--mode', type=str, help="Either training or evaluate",default="training")
    parser.add_argument('--task_type', type=str, help="Either CLASSIFICATION or REGRESSION",default="CLASSIFICATION") #the task type depends on the task!
    parser.add_argument('--target_table',type=str, help="""Here you have to put the target table. (i.e.
                                                            the name of the table containing the target nodes)""", default="drivers")
    
    # Parse the arguments
    args = parser.parse_args()
    
    print("I'm loading...")
    print("DATASET:\t", args.dataset)
    print("TASK:   \t", args.task)
    
    print("the mode is:\n\t",args.mode)
        
    path = "static_networks/f1/"+args.task+"/data_obj/"


    ## load the graphs
    dataset_train = load_graphs(path,mode="TRAIN",pk=args.target_table)
    dataset_val = load_graphs(path,mode="VAL",pk=args.target_table)
    dataset_test = load_graphs(path,mode="TEST",pk=args.target_table)

    # build the data loader
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)



    # Initialize the model
    DIM_EMB = dataset_train[0][args.target_table].x.shape[1]

    model = LightRDL(dataset_train[0].metadata(), 
                    hidden_channels=args.hidden_channels, 
                    out_channels=1,
                    num_layers=args.num_layers, 
                    dropout_prob=args.dropout_prob,
                    DIM_EMB=DIM_EMB,
                    pk = args.target_table).to(args.device)


    with torch.no_grad():  # Initialize lazy modules.
        for batch in train_loader:
            batch = batch.to(args.device)
            out = model(batch.x_dict, batch.edge_index_dict)


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.task_type == "CLASSIFICATION":
        criterion = torch.nn.BCEWithLogitsLoss()
        metric = "AUROC"
        best_val = -float("inf")
    elif args.task_type == "REGRESSION":
        criterion = torch.nn.L1Loss()
        best_val = float("inf")
        
    # train the model
    if args.mode == "training":
        model.train()
        for epoch in range(args.nb_epochs):
            train_loss = train(model,train_loader,args.device,optimizer,args.target_table,criterion)
            if epoch % args.compute_val_every == 0:
                val_loss,ys, preds = evaluate(model,val_loader,args.device,args.target_table,criterion)
                if args.task_type == "CLASSIFICATION":
                    val_auc = roc_auc_score(ys,preds)
                    print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val {metric}: {val_auc:.4f}')
                    if val_auc > best_val:
                        best_val = val_auc
                        best_epoch = epoch
                        torch.save(model.state_dict(), "trained_models/best_current_model.pth")
                        counter = 0 
                    else: 
                        counter += args.compute_val_every  
                    if counter >= args.patience:
                        break
                    
                elif args.task_type == "REGRESSION":
                    print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
                    if val_loss < best_val:
                        best_val = val_loss
                        best_epoch = epoch
                        torch.save(model.state_dict(), "trained_models/best_current_model.pth")
                        counter = 0 
                    else: 
                        counter += args.compute_val_every  
                    if counter >= args.patience:
                        break
                
    else:
        model.load_state_dict(torch.load("trained_models/best_current_model.pth"))
        model.eval()



    # eval
    if args.task_type == "CLASSIFICATION":
        train_loss, ys, preds = evaluate(model,train_loader,args.device,args.target_table,criterion)
        train_auc = roc_auc_score(ys,preds)
        val_loss,  ys, preds = evaluate(model,val_loader,args.device,args.target_table,criterion)
        val_auc = roc_auc_score(ys,preds)
        test_loss,  ys, preds = evaluate(model,test_loader,args.device,args.target_table,criterion)
        test_auc = roc_auc_score(ys,preds)

        print(f'Train Loss: {train_loss:.4f}, Train AUROC: {train_auc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val AUROC: {val_auc:.4f}')
        print(f'Test Loss: {test_loss:.4f}, Test AUROC: {test_auc:.4f}')
    elif args.task_type == "REGRESSION":
        train_loss, ys, preds = evaluate(model,train_loader,args.device,args.target_table,criterion)
        val_loss,  ys, preds = evaluate(model,val_loader,args.device,args.target_table,criterion)
        test_loss,  ys, preds = evaluate(model,test_loader,args.device,args.target_table,criterion)

        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print(f'Test Loss: {test_loss:.4f}')


if __name__ == "__main__":
    main()
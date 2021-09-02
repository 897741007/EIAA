from trainer import trainer
from GAT_prediction import GAT_predictor
import os

if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    for b_f in (1,2,0):
        for l_n in (6,5,4,3,2,1):
            """
            # edge_type:
            # le --> layer edge weight
            # fe --> fixed edge weight
            # sle --> stacked layer edge weight
            # ne --> Non edge weight
            """
            edge_type = 'se'
            if b_f == 0:
                edge_type = 'ne'
            task_id = 'layer_{0}_l{1}'.format(edge_type, b_f)

            hidden_dim = 512
            head_num = 8
            layer_num = l_n
            bond_influence = b_f
            dropout = 0.2
            prediction_class = 1
            
            if bond_influence:
                lower_aromatic = False
                specify_bond = True
            else:
                lower_aromatic = True
                specify_bond = False
            #lower_aromatic = False
            
            warm_up = 0
            epoch = 300
            batch_size = 32
            lr = 1e-4
            do_random = False
            log_record = 10
            valid = 30
            device = 'cuda'
            show_process = False

            training_data_file = '../data/lipo/lipo_train_unnorm.csv'
            valid_data_file = '../data/lipo/lipo_valid_unnorm.csv'
            test_data_file = '../data/lipo/lipo_test_unnorm.csv'


            model_trainer = trainer(task_name=task_id, model=None, epoch=epoch, batch_size=batch_size, lr=lr,
            log_record=log_record, valid=valid, device=device, show_process=show_process, warm_up=warm_up,
            lower_aromatic=lower_aromatic, specify_bond=specify_bond, shuffle_data=do_random)
            model_trainer.load_data(training_data_file, valid_data_file, test_data_file)
            GAT_model = GAT_predictor(hidden_dim, layer_num, head_num, model_trainer.data_provider.dict_size, dropout, bond_influence, prediction_class, device)
            model_trainer.model = GAT_model
            model_trainer.training_model()

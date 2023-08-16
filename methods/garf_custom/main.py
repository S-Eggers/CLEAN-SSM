import time
from methods.garf_custom.SeqGAN.train import Trainer
from methods.garf_custom.SeqGAN.get_config import get_config
from methods.garf_custom.att_reverse import att_reverse
from methods.garf_custom.rule_sample import rule_sample


def main(flag=2, order=1, remove_amount_of_error_tuples=0.0, dataset="Hosp_rules", g_h=64, g_e=64):
    config = get_config('config.ini')
    path = f"{dataset}_copy"
    f = open('data/save/log_evaluation.txt', 'w')
    f.write("")
    f.close()

    print("Load the config and define the variables")
    
    att_reverse(path, order)
    v = 0
    if flag == 0 or flag == 2:
        trainer = Trainer(order,
                        config["batch_size"],
                        config["max_length"],
                        g_e,
                        g_h,
                        g_e,
                        g_h,
                        config["d_dropout"],
                        config["generate_samples"],
                        path_pos=path,
                        path_neg=config["path_neg"],
                        g_lr=config["g_lr"],
                        d_lr=config["d_lr"],
                        n_sample=config["n_sample"],
                        path_rules=config["path_rules"],
                        remove_amount_of_error_tuples=remove_amount_of_error_tuples)
        
        v = trainer.V
        pretraining_time = time.time()
        # Pretraining for adversarial training To run this part against the trained pretrain, batch_size = 32
        print("Start pre-training")
        # insert_error.insert_error(1000)
        trainer.pre_train(g_epochs=config["g_pre_epochs"],  # 50
                        d_epochs=config["d_pre_epochs"],  # 1
                        g_pre_path=config["g_pre_weights_path"],  # data/save/generator_pre.hdf5
                        d_pre_path=config["d_pre_weights_path"],  # data/save/discriminator_pre.hdf5
                        g_lr=config["g_pre_lr"],  # 1e-2
                        d_lr=config["d_pre_lr"])  # 1e-4

        lstm_weights = trainer.get_parameter_count()
        
        pretraining_time_end = time.time()
        trainer.load_pre_train(config["g_pre_weights_path"], config["d_pre_weights_path"])
        trainer.reflect_pre_train()  # Mapping layer weights to agent


        trainer.train(steps=1,  # A total of 1 big round, each big round in the generator training 1 time, discriminator 
                                # training 1 time
                    g_steps=1,
                    head=10,
                    g_weights_path=config["g_weights_path"],
                    d_weights_path=config["d_weights_path"])
        adversarial_time = time.time()

        trainer.save(config["g_weights_path"], config["d_weights_path"])

    if flag == 1 or flag == 2:
        trainer = Trainer(order,
                        1,
                        config["max_length"],
                        g_e,
                        g_h,
                        g_e,
                        g_h,
                        config["d_dropout"],
                        config["generate_samples"],
                        path_pos=path,
                        path_neg=config["path_neg"],
                        g_lr=config["g_lr"],
                        d_lr=config["d_lr"],
                        n_sample=config["n_sample"],
                        path_rules=config["path_rules"])
        trainer.load(config["g_weights_path"], config["d_weights_path"])

        rule_len = rule_sample(config["path_rules"], path, order)
        # trainer.generate_rules(config["path_rules"], config["generate_samples"])  
        # Production of rule sequence, i.e. rules.txt

        rule_time = time.time()
        trainer.train_rules(rule_len, config["path_rules"])  # For production rules, generate rules_final.txt from rules.txt
        
        rule_time_end = time.time()
        trainer.filter(path)
        rule_filter_time_end = time.time()

        f = open('data/save/log.txt', 'w')
        f.write("")
        f.close()
        att_reverse(path, 1)
        repair_time = time.time()
        trainer.repair(path)  # Used for two-way repair, the rules are from rules_final.txt, the repair data 
        # is Hosp_rules_copy, which is a full backup of Hosp_rules + random noise
        # trainer.repair_SeqGAN()
        repair_time_end = time.time()
        
        pretraining = pretraining_time_end - pretraining_time
        print("Pretraining time: ", pretraining)
        adversarial = adversarial_time - pretraining_time_end
        print("Adversarial time: ", adversarial)
        rule = rule_time_end - rule_time
        print("Rule time: ", rule)
        rule_filter = rule_filter_time_end - rule_time_end
        print("Rule filter time: ", rule_filter)
        repair = repair_time_end - repair_time
        print("Repair time: ", repair)
        
        return pretraining, adversarial, rule, rule_filter, repair, v, lstm_weights

if __name__ == '__main__':
    config = get_config('config.ini')
    error_rate = 0.1
    flag = config["flag"]  # 0 for training SeqGAN, 1 for repairing part, 2 for doing it simultaneously
    order = config["order"]  # Order, 1 for positive order, 0 for negative order
    insert_errors = bool(config["insert_errors"])
    main(error_rate, insert_errors, flag, order)
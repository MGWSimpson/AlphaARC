

# log file for tracking all the stats
# also just track the more meta level stuff.
def make_run_log():
    return {
        "training_logs": [],
        "eval_logs": [], 
        "episode_logs": []
    }

def make_train_log(epoch): 
    return {
        "epoch": epoch,
        "policy_epoch_mean": 0,
        "value_epoch_mean": 0,
        "supervised_epoch_mean":0,
        "policy_batch_loss": [],
        "value_batch_loss": [],
        "supervised_batch_loss": [],
        
    }



def make_train_log_means(train_log):
    train_log['policy_epoch_mean'] = sum(train_log['policy_batch_loss']) / len(train_log['policy_batch_loss'])
    train_log['value_epoch_mean'] =  sum(train_log['value_batch_loss']) / len(train_log['value_batch_loss'])
    train_log['supervised_epoch_mean'] = sum(train_log['supervised_batch_loss']) / len(train_log['supervised_batch_loss'])
    return train_log

# collect information at the start of the episode
def make_episode_log(task_id, solved): 
    return {
        "task_id": task_id,
        "solved": 0,

    }


def make_eval_log(): 
    pass
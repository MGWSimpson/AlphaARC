

# log file for tracking all the stats
# also just track the more meta level stuff.
def make_run_log():
    return {
        "training_logs": [],
        "eval_logs": [], 
        "training_episode_logs": [],
        "eval_episode_logs": []
    }

def make_train_log(epoch): 
    return {
        "epoch": epoch,
        "policy_epoch_mean": 0,
        "value_epoch_mean": 0,
        "supervised_epoch_mean":0,
        "policy_batch_loss": [],
        "value_batch_loss": [],
        "total_batch_loss": [],
        "supervised_batch_loss": [],
        "supervised_buffer_capacity": 0,
        "rl_buffer_capacity":0
        
    }



def make_train_log_means(train_log):
    def safe_mean(lst):
        return sum(lst) / len(lst) if lst else 0.0
     
    train_log['policy_epoch_mean'] = safe_mean(train_log['policy_batch_loss'])
    train_log['value_epoch_mean'] =  safe_mean(train_log['value_batch_loss'])
    train_log['total_epoch_mean'] = safe_mean(train_log['total_batch_loss'])
    train_log['supervised_epoch_mean'] = safe_mean(train_log['supervised_batch_loss'])
    return train_log

# collect information at the start of the episode
def make_episode_log(task_id, solved): 
    return {
        "task_id": task_id,
        "solved": 0,
    }


def make_eval_log(): 
    pass


# log file for tracking all the stats
# also just track the more meta level stuff.
def make_run_log(train_log, train_episodes_log,
                 eval_log, eval_episodes_log):
    
    return {
        "policy_epoch_mean": train_log['policy_epoch_mean'],
        "value_epoch_mean": train_log['value_epoch_mean'],
        "supervised_epoch_mean": train_log['supervised_epoch_mean'],
        "supervised_buffer_capacity": train_log['supervised_buffer_capacity'],
        "rl_buffer_capacity": train_log['rl_buffer_capacity'] ,   
        "train_solve_rate": summarize_episodes(train_episodes_log),
        "eval_solve_rate": eval_log['solve_rate'],
        "training_correct_syntax_rate": summarize_syntax_rate(train_episodes_log),
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
        "rl_buffer_capacity": 0   
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
# 
def make_episode_log(task_id): 
    return {
        "task_id": task_id,
        "solved": 0.0,
        "depth": 0,
        "correct_syntax_ratio": 0.0
        }



def summarize_episodes(episode_logs):
    solve_rate = sum([x['solved'] for x in episode_logs]) / len(episode_logs)
    return solve_rate


def summarize_syntax_rate(episode_logs): 
    syntax_rate = sum(x['correct_syntax_ratio'] for x in episode_logs) / len(episode_logs)
    return syntax_rate

def make_eval_log(): 
    return {
        "solve_rate": 0.0,
    }
import os
import json
from alphaarc.task import Task, from_dict

class Curriculum:
    def __init__(self, dir_paths=[], file_paths=[]):
        self.tasks = []
        self. current_task_counter = 0
        self._add_data_sources(dirs=dir_paths, files=file_paths)
        print(f"loaded {len(self.tasks)} tasks")

    def _load_tasks_from_folders(self, dir_path):
        file_paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]
        new_tasks = [Task.from_json(path) for path in file_paths]
        self.tasks.extend(new_tasks)


    def _load_tasks_from_files(self, file_path):
        # load the JSON and iterate through the keys basically.
        with open(file_path) as fp:
            json_object = json.load(fp)
        task_keys = json_object.keys()
        new_tasks = [from_dict(json_object[key]) for key in task_keys]
        self.tasks.extend(new_tasks)


    def _add_data_sources(self, dirs=None, files=None): 
        for folder_path in dirs:
            self._load_tasks_from_folders(folder_path)
        for file_path in files:
            self._load_tasks_from_files(file_path)

    
    # simply loop the tasks for now
    # TODO: explore curriculum strategies 
    def select_task(self): 
        selected_task = self.tasks[self.current_task_counter]  
        self.current_task_counter +=1
        if self.current_task_counter > len(self.tasks): 
            self.current_task_counter = 0
        
        return selected_task




if __name__ == "__main__": 
    directories = ['data/training', 'data/evaluation']
    files = ['data/mutated_tasks_train_9600.json']
    c = Curriculum(dir_paths=directories, file_paths=files)
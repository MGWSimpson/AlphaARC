from alphaarc.task import Task, from_dict
import os
import json


class BaseCurriculum:
    def __init__(self, dir_paths=[], file_paths=[], is_eval=False):
        self.tasks = []
        self.solved_tasks = []
        self.is_eval = is_eval

        self._add_data_sources(dirs=dir_paths, files=file_paths)
        print(f"loaded {len(self.tasks)} tasks, is eval: {self.is_eval}")

    def _load_tasks_from_folders(self, dir_path):
        file_paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]
        new_tasks = [Task.from_json(path, self.is_eval) for path in file_paths]
        self.tasks.extend(new_tasks)

    def _load_tasks_from_files(self, file_path):
        # load the JSON and iterate through the keys basically.
        with open(file_path) as fp:
            json_object = json.load(fp)
        task_keys = json_object.keys()
        new_tasks = [from_dict(json_object[key], self.is_eval) for key in task_keys]
        self.tasks.extend(new_tasks)


    def _add_data_sources(self, dirs=None, files=None): 
        for folder_path in dirs:
            self._load_tasks_from_folders(folder_path)
        for file_path in files:
            self._load_tasks_from_files(file_path)


    def __len__(self): 
        return len(self.tasks)
    
    def get_total_n(self):
        return len(self.solved_tasks) + len(self.tasks)

    def get_n_solved(self): 
        return len(self.solved_tasks)
    

    def handle_solved_tasks(self, task):
        self.solved_tasks.append(task)
        self.tasks.remove(task)


    def generate_curriculum(self) -> list[Task]: 
        raise NotImplementedError




class BaselineCurriculum(BaseCurriculum):
    def __init__(self, dir_paths=[], file_paths=[], is_eval=False):
        super().__init__(dir_paths, file_paths, is_eval)
    

    def generate_curriculum(self) -> list[Task]:
        return self.tasks

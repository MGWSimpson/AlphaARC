from __future__ import annotations

import csv
import json
import datetime as dt
from pathlib import Path
from typing import Mapping, Any, Final




class TrainingLogger:
    pass
 

class Logger:
    def __init__(
        self,
        root_dir = "logs",
        timestamp_subdir= True,
        timestamp_fmt = "%Y-%m-%d_%H-%M-%S" ,
    ):
        base = Path(root_dir)
        self.log_dir = (
            base / dt.datetime.now().strftime(timestamp_fmt)
            if timestamp_subdir
            else base
        )
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self._init_training_log()

 
    
    def _init_training_log(self): 
        self._training_log_path = self.log_dir / "training_log.csv"
        self._initialise_csv(path=self._training_log_path,row=[
                        "step",
                        "policy_loss",
                        "value_loss",
                        "supervised_loss",
                    ])

   

    def _initialise_csv(self, path, row) -> None:
        if not path.exists():
            with path.open("w", newline="") as fp:
                csv.writer(fp).writerow(
                    row
                )

    def _write_row(self, path, row: list[Any]) -> None:
        with path.open("a", newline="") as fp:
            csv.writer(fp).writerow(row)

    def log_training_data(
        self,
        policy_loss: float,
        value_loss: float,
        supervised_loss: float,
        step: int | None = None,
    ) -> None:
        self._write_row(self._training_log_path, [step, policy_loss, value_loss, supervised_loss])

  

if __name__ == "__main__":  # pragma: no cover
    import random
    
    log = Logger('./runs')

    for step in range(3):
            log.log_training_data(
                policy_loss=random.random(),
                value_loss=random.random(),
                supervised_loss=random.random(),
                step=step,
            )

import torch
import numpy as np
import os
import json
import logging
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from numbers import Number

log = logging.getLogger("epoch_logger")

def is_json_convertible(obj):
    return any([isinstance(obj, t) for t in (dict, tuple, list, str, Number)])
        
def convert_json(obj):
    if isinstance(obj, dict):
        return { convert_json(k) : convert_json(v) for k,v in obj.items()
                     if isinstance(k, str) and is_json_convertible(v) }
    if isinstance(obj, tuple) or isinstance(obj, list):
        return [ convert_json(x) for x in obj if is_json_convertible(x) ]
    if isinstance(obj, str) or isinstance(obj, Number):
        return obj
    return None

class EpochLogger:
    def __init__(self, save_dir, read_only=False):
        if not os.path.isdir(save_dir) and not read_only:
            os.makedirs(save_dir)
        self._save_dir = save_dir
        self._info = defaultdict(list)
        if not read_only:
            self._summary_writer = SummaryWriter(log_dir=self._save_dir)

    def save_settings(self, settings):
        """Saves settings to a json file if it does not exist, otherwise
        it loads the existing json file and compares settings between
        argumens and the saved file.
        """
        save_file = os.path.join(self._save_dir, "settings.json")
        settings = convert_json(vars(settings))
        if os.path.isfile(save_file):
            with open(save_file, "rt") as f:
                saved_settings = json.load(f)
            if saved_settings != settings:
                log.warn("SETTINGS CHANGED BETWEEN RUNS!")
                for k in settings:
                    if k not in saved_settings:
                        log.warn(f"{k}: {settings[k]} is new")
                for k in saved_settings:
                    if k not in settings:
                        log.warn(f"{k}: {saved_settings[k]} is missing")
                for k in settings:
                    if k in saved_settings and settings[k] != saved_settings[k]:
                        log.warn(f"{k}: {saved_settings[k]} -> {settings[k]}")
                raise ValueError("settings changed between runs")
        else:
            s = json.dumps(settings, separators=(',', ':\t'), indent=4, sort_keys=True)
            log.info(f"Settings = {s}")
            with open(save_file, "wt") as f:
                f.write(s)

    def accum(self, **kwargs):
        """Accumulates key-value pairs for epoch logging.

        Each key corresponds to an initially empty list.  Each value is
        appended to that list.  The epoch() call saves the results for
        logging, and resets the lists for the next epoch.
        """
        for k,v in kwargs.items():
            self._info[k].append(v)

    def save_checkpoint(self, epoch_no, **kwargs):
        save_file = os.path.join(self._save_dir, f"checkpoint_{epoch_no:03d}.pt")
        cp = {k:v for k,v in kwargs.items()}
        cp['epoch_no'] = epoch_no
        torch.save(cp, save_file)

    def load_checkpoint(self, epoch_no=None):
        if epoch_no is None:
            epoch_no = 1
            while os.path.isfile(os.path.join(self._save_dir, f"checkpoint_{epoch_no:03d}.pt")):
                epoch_no += 1
            epoch_no -= 1
        
            if epoch_no == 0:
                return None
        
        return torch.load(os.path.join(self._save_dir, f"checkpoint_{epoch_no:03d}.pt"))

    def epoch(self, epoch_no, **kwargs):
        """End-of-epoch logging.

        Writes accumulated data, and key-value argument pairs to the
        log/event file.
        """
        log.info(f"Epoch {epoch_no} done")
        for k,v in self._info.items():
            values = np.array(v, dtype=np.float32)
            if len(values) > 0:
                self._summary_writer.add_histogram("RLQP/"+k, values, epoch_no)
                log.info(f"  {k}: range=[{np.amin(values)}, {np.amax(values)}], mean={np.mean(values)} + {np.std(values)}")
                
        for k,v in kwargs.items():
            self._summary_writer.add_scalar("RLQP/"+k, v, epoch_no)
            log.info(f"  {k}: {v}")
        
        self._summary_writer.flush()
        self._info = defaultdict(list)


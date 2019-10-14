import os
import logging
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime

from utils.logger import setup_logging
from utils.utils import read_json, write_json

class ConfigParser:
    def __init__(self, args, options=""):
        """
        - class to parse configuration json file. Handles hyperparameters for training, initializations of modules, checkpoint saving
        and logging module.

        input:
            args: Dict containing configurations, hyperparameters for training. contents of `parameters.json` file for example.
            options: Dict keychain:value, specifying position values to be replaced from config dict.
        """
        # parse default and custom cli options
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        args = args.parse_args()

        self.cfg_fname = Path(args.config)

        # load json file as python dictionary 
        config = read_json(self.cfg_fname)

        config["src_data"] = args.src_data 
        config["tgt_data"] = args.tgt_data 

        config["src_data_prefix"] = args.src_data_prefix 
        config["tgt_data_prefix"] = args.tgt_data_prefix 

        # load config file and apply custom cli options
        self._config = _update_config(config, options, args)

        # set save directory where trained embedding and log will be saved
        save_dir = Path(args.save) / ( config["src_data_prefix"] + "_" + config["tgt_data_prefix"])

        timestamp = datetime.now().strftime(r'%m%d_%H%M%S') 

        exper_name = self.config['name']

        print(f"Result will be saved in {save_dir}")
        
        self._save_dir = save_dir / 'embed' / exper_name / timestamp
        self._log_dir = save_dir / 'log' / exper_name / timestamp

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # save updated config file to the checkpoint dir 
        write_json(self.config, self.save_dir / "parameters.json")

        # configure logging module
        setup_logging(self.log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    def save_file(self, content, fname):

        write_json(content, self.save_dir / fname)

        
    def __getitem__(self, name):
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir

# helper functions used to update config dict with custom cli options
def _update_config(config, options, args):
    for opt in options:
        value = getattr(args, _get_opt_name(opt.flags))
        if value is not None:
            _set_by_path(config, opt.target, value)
    return config

def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')

def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    _get_by_path(tree, keys[:-1])[keys[-1]] = value

def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
import sys
sys.path.append("..")

from detectors import *
from evaluator import Evaluator
from inspect import signature
from random import choice
from tensorflow.python.client.device_lib import list_local_devices as lsdev

ADJECTIVES = [
    "attractive", "bald", "beautiful", "chubby", "clean", "dazzling", "drab",
    "elegant", "fancy", "fit", "flabby", "glamorous", "gorgeous", "handsome",
    "long", "magnificent", "muscular", "plain", "plump", "quaint", "scruffy",
    "shapely", "short", "skinny", "stocky", "ugly", "unkempt", "unsightly",
    "zealous", "fly", "amazing", "sly"
]

ANIMALS = [
    "chameleon", "panda", "raccoon", "tapir", "elephant", "tiger", "lion",
    "penguin", "alpaca", "cat", "dog", "mouse", "snake", "gnu", "fox", "badger",
    "eagle", "hedgehog", "dragon", "hyena", "falcon", "dove", "ant", "bear",
    "horse", "sheep", "elk", "cow", "wolf", "whale", "dolphin", "shark",
    "platypus", "kangaroo", "koala", "sloth", "seal", "reindeer", "pig",
    "spider", "octopus", "snail", "mosquito", "fly", "frog", "axolotl"
]

class NameBuilder():
    def __init__(self):
        self.already_in_use = []

    def name(self):
        name = "_".join([choice(ADJECTIVES), choice(ANIMALS)])
        if name not in self.already_in_use:
            self.already_in_use.append(name)
            return name
        else:
            return self.name()


def evaluator_from_dict(conf):
    if not "input_dir" in conf:
        raise KeyError("No input directory specified in configuration.")
        
    if not "output_dir" in conf:
        raise KeyError("No output directory specified in configuration.")

    name_builder = NameBuilder()

    evaluator = __call_constructor(Evaluator, conf)
    evaluator.add_detectors(*[
        detector_from_dict(detector_conf, name_builder)
        for detector_conf in conf["detectors"]])
    evaluator.add_records(*(conf["records"]))

    return evaluator

def detector_from_dict(conf, name_builder):
    if not "type" in conf:
        raise KeyError("No detector type specified in configuration.")

    detector_class = eval(conf["type"])
    if not "name" in conf:
        conf["name"] = name_builder.name()
    if not "gpus" in conf:
        conf["gpus"] = len([d for d in lsdev() if d.device_type == "GPU"])
    return __call_constructor(detector_class, conf)

def __call_constructor(klass, conf):
    constructor_keys = signature(klass.__init__).parameters.keys()
    kwargs = {key: conf[key] for key in constructor_keys if key in conf}
    return klass(**kwargs)
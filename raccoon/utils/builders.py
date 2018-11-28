import sys
sys.path.append("..")

from detectors import *
from evaluator import Evaluator
from inspect import signature

def evaluator_from_dict(conf):
    if not "input_dir" in conf:
        raise KeyError("No input directory specified in configuration.")
        
    if not "output_dir" in conf:
        raise KeyError("No output directory specified in configuration.")

    evaluator = __call_constructor(Evaluator, conf)
    evaluator.add_detectors(*[detector_from_dict(c) for c in conf["detectors"]])
    evaluator.add_records(*(conf["records"]))

    return evaluator

def detector_from_dict(conf):
    if not "type" in conf:
        raise KeyError("No detector type specified in configuration.")

    detector_class = eval(conf["type"])
    return __call_constructor(detector_class, conf)

def __call_constructor(klass, conf):
    constructor_keys = signature(klass.__init__).parameters.keys()
    kwargs = {key: conf[key] for key in constructor_keys if key in conf}
    return klass(**kwargs)
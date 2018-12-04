from ..detectors import *
from ..evaluator import Evaluator

from inspect import signature, Parameter
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
    "spider", "octopus", "snail", "mosquito", "fly", "frog", "axolotl",
    "aardvark", "beaver", "otter", "snake", "python", "anaconda", "goat",
    "zebra", "rhino", "hippo", "pony", "chicken", "turkey", "turtle", "rat",
    "tortoise", "porpoise", "butterfly", "caterpillar", "wolverine", "squirrel",
    "monkey", "ape", "gorilla", "chimpanzee", "bonobo", "baboon", "organgutan",
    "boar", "cheetah", "katta", "fossa", "jaguar", "leopard", "panther", "orca",
    "beluga", "ocelot", "mole", "scorpion", "jellyfish", "bat", "harpy", "raven",
    "bateleur", "owl", "magpie", "macaw", "parrot", "cockatoo", "crocodile",
    "eel", "nightingale", "dormouse", "worm", "chinchilla", "ferret", "toad",
    "donkey", "weasel", "coral", "ray", "pelican", "flamingo", "echidna", "kiwi",
    "ostrich", "emu", "piranha", "seagull", "peafowl", "opossum", "llama",
    "giraffe", "porcupine", "anteater", "capybara", "rabbit", "bunny", "toucan",
    "duck", "goose", "lynx", "coelacanth", "crab", "squid", "camel", "koi",
    "catfish", "swordfish", "sailfish", "marlin", "merlin", "chipmunk",
    "gopher", "groundhog", "baboon", "meerkat", "warthog", "puffin", "gazelle",
    "antelope", "deer", "shrew", "woodpecker", "slug", "crow", "wasp", "bee",
    "millipede", "centipede", "ermine", "tanuki", "salamander", "lizard",
    "cobra", "beetle", "ladybug", "liger", "tigon", "mule", "plankton", "shrimp",
    "mite", "tick", "flea", "lionfish", "tuna", "anglerfish", "seadragon",
    "seahorse", "anemone", "dragonfly", "mantis", "crane", "swan", "sparrow",
    "stork", "bongo", "salmon", "skunk", "badger", "gerbil", "coyote", "cougar",
    "cockroach", "dingo", "pangolin", "caracal", "lemur", "aye-aye", "bilby",
    "binturong", "cricket", "civet", "possum", "hamster", "lorikeet", "gecko",
    "firefly", "mink", "kookaburra", "tarantula", "wombat", "bison", "buffalo",
    "locust", "mandrill", "newt", "oyster", "quail", "sardine", "sandpiper",
    "yak", "moose", "hare", "skink", "guanaco", "dugong", "manatee", "iguana",
    "tadpole", "lungfish", "mudskipper", "basilisk", "kea", "kaka", "kakapo",
    "unicornfish", "moth", "grasshopper", "mongoose", "lemming", "robin",
    "cormorant", "barracuda", "hornet", "ibis", "jackal", "impala", "cavy",
    "cassowary", "rhea", "albatross", "armadillo", "blackbird", "cod",
    "hawk", "herring", "hummingbird", "kudu", "lark", "loris", "termite",
    "lobster", "mallard", "walrus", "narwhale", "baiji", "mara", "tamarin",
    "capuchin", "boa", "viper", "alligator", "rattlesnake", "sturgeon",
    "arowana", "swordtail", "loach", "bandicoot", "wallaby", "leech"
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

class InsufficientConfiguration(Exception):
    pass

def evaluator_from_dict(conf):
    if not "input_dir" in conf:
        raise InsufficientConfiguration(
            "No input directory specified in configuration.")
        
    if not "output_dir" in conf:
        raise InsufficientConfiguration(
            "No output directory specified in configuration.")

    name_builder = NameBuilder()

    evaluator = __call_constructor(Evaluator, conf)
    
    if "detectors" in conf: 
        evaluator.add_detectors(*[
            detector_from_dict(detector_conf, name_builder)
            for detector_conf in conf["detectors"]])
    else: raise InsufficientConfiguration("No detectors specified.")

    if "records" in conf: evaluator.add_records(*(conf["records"]))
    else: raise InsufficientConfiguration("No records specified.")

    return evaluator

def detector_from_dict(conf, name_builder):
    if not "type" in conf:
        raise InsufficientConfiguration(
            "No detector type specified in configuration.")

    detector_class = eval(conf["type"])
    if not "name" in conf:
        conf["name"] = name_builder.name()
    if not "gpus" in conf:
        conf["gpus"] = len([d for d in lsdev() if d.device_type == "GPU"])
    return __call_constructor(detector_class, conf)

def __call_constructor(klass, conf):
    ctor_params = signature(klass).parameters
    kwargs = {}
    for key, param in ctor_params.items():
        if key in conf: kwargs[key] = conf[key]
        elif param.default == Parameter.empty:
            raise InsufficientConfiguration(
                "{} missing in detector configuration".format(key))
    return klass(**kwargs)

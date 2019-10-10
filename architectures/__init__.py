import importlib
from architectures.base_architecture import BaseArchitecture


def find_architecture_using_name(arch_name):
    # Given the option --architecture [architecture],
    # the file "networks/{}_architecture.py" will be imported.
    arch_filename = "architectures." + arch_name + "_architecture"
    arch_lib = importlib.import_module(arch_filename)

    # In the file, the class called [ArchiterctureName]Architecture() will
    # be instantiated. It has to be a subclass of BaseArchitecture, and it is case-insensitive.
    architecture = None
    target_model_name = arch_name.replace('_', '') + 'architecture'
    for name, cls in arch_lib.__dict__.items():
        if name.lower() == target_model_name.lower() and issubclass(cls, BaseArchitecture):
            architecture = cls

    if architecture is None:
        print("No architecture class with name {} was found in {}.py,".format(target_model_name, arch_filename))
        exit(0)

    return architecture


def create_architecture(args):
    model = find_architecture_using_name(args.architecture)
    instance = model(args)
    print("architecture [{}] was created".format(instance.architecture))
    return instance

import pickle


class Unpickler(pickle.Unpickler):
    def __init__(self, *args, **kwargs):
        super(Unpickler, self).__init__(*args, **kwargs)

    def find_class(self, __module_name: str, __global_name: str):
        if __global_name == 'MobileNetSkipAdd':
            from fast_depth.models import MobileNetSkipAdd
            return MobileNetSkipAdd
        if __module_name == "metrics":
            return super().find_class(".".join(["fast_depth", __module_name]), __global_name)
        else:
            return super().find_class(__module_name, __global_name)


class Pickler(pickle.Pickler):
    def __init__(self, *args, **kwargs):
        super(Pickler, self).__init__(*args, **kwargs)


def load(*args, **kwargs):
    return pickle.load(*args, **kwargs)

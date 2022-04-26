
class BenchmarkPOD:

    """
        Plain Old Dataobject corralling data generated by benchmarking different wavelength selection methods
    """

    def __init__(self):
        self.data = dict()
        self.meta = dict()

    def register_meta(self, **kwargs):
        for key in kwargs.keys():
            self.meta[key] = kwargs[key]

    def register(self, method_key: str, *keys, **kwargs):
        data = self.data.setdefault(method_key, dict())
        for key in keys:
            data = data.setdefault(key, dict())
        for key in kwargs.keys():
            values = data.setdefault(key, [])
            values.append(kwargs[key])

    def get_methods(self):
        return list(self.data.keys())

    def get_item(self, method: str, *item_keys):
        d = self.data[method]
        for item_key in item_keys:
            d = d[item_key]
        return d

    def get_meta_item(self, key: str):
        return self.meta[key]


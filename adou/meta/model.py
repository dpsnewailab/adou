class ModelType(type):
    """
    Meta class of a generic model, for future extend.
    """
    def __new__(cls, name, bases, body):
        if '__doc__' not in body:
            raise TypeError('Your model must be have some explanations.')

        if '__schema__' not in body:
            raise TypeError('Define your model input/output schema.')
        elif '__schema__' in body and ('input' not in body['__schema__'] or 'output' not in body['__schema__']):
            raise TypeError('Invalid schema, you should define both input and output specific field.')
        
        return super().__new__(cls, name, bases, body)


class Model:
    def load(self, *args, **kwargs): pass
    def summary(self, *args, **kwargs): pass
    def train(self, *args, **kwargs): pass
    def predict(self, *args, **kwargs): pass


class MyModel(Model, metaclass=ModelType):
    """
    asb
    """
    __schema__ = {
        'input': '1',
        'output': '2'
    }

    def __init__(self):
        super().__init__()

    def load(self, *args, **kwargs): pass
    def summary(self, *args, **kwargs): pass
    def train(self, *args, **kwargs): pass
    def predict(self, *args, **kwargs): pass

my_model = MyModel()

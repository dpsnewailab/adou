Just some typical **A**pproaches for **Do**cument **U**nderstanding and related tasks :book:

---

#### Image

- [ ] Document Layout Analysis.
- [ ] OCR pipeline.

#### Text ETL Process (Extract, Transform, Load)

- [ ] Preprocess.
- [ ] Vietnamese specific text mining toolbox. 

#### Design Patterns

```python
class MyModel(adou.Model):
    __doc__ = ...
    schema = {'input':..., 'output':...}
    
    def load(self, *args, **kwargs)
    def summary(self, *args, **kwargs)
    def train(self, *args, **kwargs)
    def predict(self, *args, **kwargs)
```

```python
class MyModelTestCase(unittest.TestCase):
    def test_case_01(self, *args, **kwargs)
    def test_case_02(self, *args, **kwargs)
    ...
```
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
    def load(self, *args, **kwargs)
    def summary(self, *args, **kwargs)
    def train(self, *args, **kwargs)
    def predict(self, *args, **kwargs)
    def help(self, *args, **kwargs):
        def pipeline()
        def input_schema()
        def output_schema()
```

```python
class MyModelTestCase(unittest.TestCase):
    def test_case_01(self, *args, **kwargs)
    def test_case_02(self, *args, **kwargs)
    ...
```
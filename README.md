# handcrafted_feature_builder

handcrafted feature builder

## How to use

```python
x = np.random.rand(2, 3, 10)
builder = HandcraftedFeatureBuilder(target_axis=-1)
builder = builder.Max().Min()
trans = builder.build()
t = trans(x)
assert t.shape == (2, 3, 2)
# t.shape == (2, 3, [max val, min val])

x = np.random.rand(2, 10)
builder = HandcraftedFeatureBuilder.from_str('max,min', target_axis=-1)
trans = builder.build()
t = trans(x)
assert t.shape == (2, 2)
```

## Features

* max
* min
* mean
* std
* var
* sum
* median
* percentile
* cv
* abs\_mean
* abs\_max
* abs\_min
* abs\_std
* intensity
* zcr


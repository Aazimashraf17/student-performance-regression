# Data provenance

`student-mat.csv` is the unmodified mathematics CSV from the UCI Student Performance download. It contains 395 records and 33 columns. The project uses only G1 and G2 as predictors of G3. The separate Portuguese-language dataset is not combined with it.

- Creator: Paulo Cortez.
- Source: [UCI Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance).
- Download: [official ZIP](https://archive.ics.uci.edu/static/public/320/student+performance.zip).
- Citation: Cortez, P. (2008). *Student Performance* [Dataset]. UCI Machine Learning Repository. [https://doi.org/10.24432/C5TG7T](https://doi.org/10.24432/C5TG7T).
- License: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
- Retrieved: 2026-09-12.
- Source changes: none. Analysis outputs and `split.csv` were created for this project; no endorsement by the data creator is implied.

The source CSV uses semicolon separators. Its SHA-256 is:

```text
e47f9ee225e1ee6e69b7564e6dac7123e80b8486677fe111f351964cef5dec80
```

Its MD5, used by base R's built-in checksum function, is:

```text
4dc304be95c60de6ee13fb8769469dd7
```

## Shared split

`row_id` is the original data-row number, beginning at 1 after the header; it is a project identifier, not a source student ID. `split` contains `train` or `test`. No rows are dropped, imputed, or merged.

Reproduce the split in Python with:

```python
import numpy as np
import pandas as pd

row_ids = np.arange(1, 396)
test_ids = np.random.default_rng(42).permutation(row_ids)[:79]
split = pd.DataFrame({
    "row_id": row_ids,
    "split": np.where(np.isin(row_ids, test_ids), "test", "train"),
})
```

The committed split is authoritative. The data checks found no missing cells, no fully duplicated source records, and 38 final grades equal to zero. Those are observed data checks, not assumptions about why any student received a particular grade.

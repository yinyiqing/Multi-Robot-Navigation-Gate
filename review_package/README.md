# Review package

`piroute-review-package-20260907.tar.zst` is a reduced working copy for reviewing the ICRA paper
and checking the frozen results. It contains the manuscript, protocols, current model artifacts,
G25 sealed results, G17 development results, G26 exploratory results, corrected qualitative captures,
and the necessary analysis code.

It excludes historical training runs, replay/student shards, TensorBoard files, ROS build caches,
old invalid B2 captures, `readpaper` PDFs, and `trash/`.

Verify and extract:

```bash
sha256sum -c piroute-review-package-20260907.tar.zst.sha256
tar --zstd -xf piroute-review-package-20260907.tar.zst
```

The archive is intended for local paper review and lightweight reproduction. It is not a complete
archive of every historical training artifact.

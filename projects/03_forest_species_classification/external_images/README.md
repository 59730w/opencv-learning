# External evaluation images

The 26 external image files are intentionally not committed to keep the repository lightweight.

`metadata.csv` records the expected class or negative type, Wikimedia Commons source page, direct image URL, creator and license for every frozen sample.

Recreate and validate the frozen set with:

```bash
python scripts/download_external_images.py
```

Do not replace failed images or use this frozen set for training, checkpoint selection or parameter tuning.

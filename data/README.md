# Data

This directory holds datasets. Its contents are not versioned (see `.gitignore`); only the
folder structure is kept.

- `raw/` — original, immutable data as collected or downloaded
- `interim/` — intermediate, partially processed data
- `processed/` — final datasets ready for modelling

Scraped social media data can carry terms-of-service and privacy constraints. Keep raw
captures here, write down where each dataset came from, and do not commit them.

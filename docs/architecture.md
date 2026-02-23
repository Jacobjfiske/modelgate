# Architecture

```text
+--------------------+        +-----------------------+
| train/train_*.py   |        | models/registry.json  |
| produces artifact  +------->+ stable/canary pointers|
+--------------------+        +-----------+-----------+
                                          |
                                          v
                                +----------------------+
                                | inference/main.py    |
                                | FastAPI runtime      |
                                +----------+-----------+
                                           |
               +---------------------------+-------------------------+
               |                                                     |
               v                                                     v
   +--------------------------+                           +----------------------+
   | feature_validation.py    |                           | metrics.py           |
   | schema/range guard       |                           | latency/errors/success|
   +--------------------------+                           +----------------------+
```

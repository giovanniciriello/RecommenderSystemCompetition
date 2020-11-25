import pandas as pd

good = pd.read_csv('submission.csv').user_id.tolist()
bad = pd.read_csv('data/submissions.csv').user_id.tolist()

for x in bad:
  if x not in good:
    print(x)
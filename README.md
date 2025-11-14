# Graph Tensor Convolutional Network


## Project Structure

| File Name          | Description                                  |
|--------------------|----------------------------------------------|
| read_data.py       | Generate training data.                       |
| main_train.py     | Train models.                                |
| read_data_utils.py | Provide data processing utilities for training. |
| help.py            | Contain auxiliary functions for model training. |
| GTCN.py            | Implementation of the GTCN model.             |


## Data Format

```
From,To,Value,TimeStamp
1,401,1,0
9,270,8,0
9,969,8,0
53,118,5,0
112,53,4,0
118,1,8,0
118,53,5,0

118,270,8,0

# Argumentative Causal Discovery
This repository provides the code for the paper "ArgCaausalDiscovery: Constraint-based Causal Structure Learning with Argumentation". 

### Example usage
To run the CausalABA algorithm from python at the root folder, run:
```
# Imports
python causalaba.py

OUTPUT:
Answer: 
Answer: arrow(1,2)
Answer: arrow(2,1)
Answer: arrow(2,0)
Answer: arrow(2,0) arrow(1,2)
Answer: arrow(2,0) arrow(2,1)
Answer: arrow(0,2)
Answer: arrow(0,2) arrow(1,2)
Answer: arrow(0,2) arrow(2,1)
Answer: arrow(1,0) arrow(2,0)
Answer: arrow(1,0) arrow(2,0) arrow(1,2)
Answer: arrow(1,0) arrow(2,0) arrow(2,1)
Answer: arrow(1,0)
Answer: arrow(1,0) arrow(1,2)
Answer: arrow(1,0) arrow(0,2)
Answer: arrow(1,0) arrow(0,2) arrow(1,2)
Answer: arrow(1,0) arrow(2,1)
Answer: arrow(0,1) arrow(2,0)
Answer: arrow(0,1) arrow(2,0) arrow(2,1)
Answer: arrow(0,1)
Answer: arrow(0,1) arrow(0,2)
Answer: arrow(0,1) arrow(1,2)
Answer: arrow(0,1) arrow(2,1)
Answer: arrow(0,1) arrow(0,2) arrow(1,2)
Answer: arrow(0,1) arrow(0,2) arrow(2,1)
RunTime: 0:00:00.000782
Number of models: 25
```

### Environment
The code was tested with Python 3.10. `requirements.txt` provides the necessary python packages. Run `pip install -r requirements.txt` from a terminal at the root folder to install all packages in your virtual environment. You will need clingo installed from the potassco repository via conda (command provided in requirements.txt).

# rl_constraint
How to run:
1. Go to one of the two sub-directories: inventory or VRP
2. Create a docker image using Dockerfile in each sub-directory
3. Run the following command inside the docker
  3.1 For VRP: python dqn_vrp.py --iters 2000 --problem E-n101-k14 --constraint-id 0
  3.2 For inventory: python pipeline_ppo/inventory_train_constraint_pipeline.py --config random

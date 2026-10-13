# master todo list
# 1. implement synthesize_pairs() in generate_preference.data.py -- PART 1
# 2. in configs/train_config.py, you can tune hyperparameter values.
# 3. in utils/masking_utils.py, you must implement tokenize_with_response_mask() -- PART 1
#
# 4. in train_sft.py, you have to implement the training loop -- PART 2
# 5. similarly, in train_dpo.py, implement the training loop -- PART 2
#
# 6. in evaluate.py, you will just run this to write the results.json file. -- PART 3
# 7. ./eval has a classifier, results.json file (once you write it), a filled template (?), and the eval contract. 
#    the held-out eval sets and mmlu slide are the hidden test set and are not included.
#    (confusing) you can self-check by dropping your own dev prompts... apparently some clarifying instructions
#    are in eval/REAME.md.
#
#
# 8. slurm/ has your job templates.
# 

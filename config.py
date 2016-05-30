TRAINSET_PATH = 'train_pdtb.json'
WORD_PAIRS = 'dict/word_pairs.txt'
PARSETREE_DICT = 'dict/parsetree.txt'
PRODUCTION_RULES = 'dict/production_rules.txt'
ARG1_PARSETREE = 'dict/arg1_parsetree.txt'
ARG2_PARSETREE = 'dict/arg2_parsetree.txt'
ARG1_PRODUCTOIN_RULE = 'dict/arg1_production_rules.txt'
ARG2_PRODUCTOIN_RULE = 'dict/arg2_production_rules.txt'
BOTH_PRODUCTOIN_RULE = 'dict/both_production_rules.txt'
DEPENDENCY_RULES = 'dict/dependency_rule.txt'
STANFORD_PARSER_JAR_PATH = 'lib/stanford-parser/stanford-parser.jar'
STANFORD_PARSER_MODEL_PATH = 'lib/stanford-parser/stanford-parser-3.6.0-models.jar'

USE_DEPENDENCY_RULE_NO = 600 # max, 500, accuracy = 0.31726
USE_PRODUCTION_RULE_NO = 500 # max, 500, accuracy = 0.4076305221
USE_WORD_PAIR_NO = 1000 		 # max, 500, accuracy = 0.329317269076
MODEL = 'model/altogether_wp_pr_dr.model'
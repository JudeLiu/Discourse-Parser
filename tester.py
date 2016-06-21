# -*- coding: utf-8 -*- 
import util, config, feature_functions
import nltk, codecs, json, pickle, time, sys, getopt

class Tester:
	def __init__(self, fname, **kargs):
		self.file_name = fname
		self.all_relations = util.read_all_data_utf8(self.file_name)


	def test_word_pair_train(self):
		#print(self.dict_word_pairs)
		model = pickle.load(open('model/word_pair.model', 'rb'))
		self.dict_word_pairs = util.load_dict_word_pairs(config.WORD_PAIRS, config.USE_WORD_PAIR_NO)

		features = []
		for rel in self.all_relations:
			num_sense = util.map_sense_to_number(rel['Sense'][0])

			feat = {}
			feat.update(feature_functions.word_pairs(rel, self.dict_word_pairs))

			if num_sense != -1:
				features.append( (feat, num_sense) )
		
		print( 'Accuracy is: %1.10f' % nltk.classify.util.accuracy(model, features))

	def test_production_rule_train(self, write_parse_tree = False):
		model = pickle.load(open('model/production_rule.model', 'rb'))
		if write_parse_tree:
			util.write_parse_tree_to_file('dev_pdtb.json')

		arg1_production_rule_dict = util.load_production_rule_dict(config.ARG1_PRODUCTOIN_RULE, config.USE_DEPENDENCY_RULE_NO)
		arg2_production_rule_dict = util.load_production_rule_dict(config.ARG2_PRODUCTOIN_RULE, config.USE_DEPENDENCY_RULE_NO)
		both_production_rule_dict = util.load_production_rule_dict(config.BOTH_PRODUCTOIN_RULE, config.USE_DEPENDENCY_RULE_NO)

		with codecs.open('tmp/arg1_parsetree.txt', encoding = 'utf-8') as file:
			arg1_parsetree = file.read().split('\n')

		with codecs.open('tmp/arg2_parsetree.txt', encoding = 'utf-8') as file:
			arg2_parsetree = file.read().split('\n')
		
		features = []
		for rel_idx, rel in enumerate(self.all_relations):
			num_sense = util.map_sense_to_number(rel['Sense'][0])
			feat = {}
			feat.update( feature_functions.production_rules
							(rel_idx, 
								[arg1_production_rule_dict, arg2_production_rule_dict, both_production_rule_dict], 
								[arg1_parsetree, arg2_parsetree]
 							)
						)

			if num_sense >= 1 and num_sense <= 7:
				features.append( (feat, num_sense) )

		print( 'Accuracy is: %1.10f' % nltk.classify.util.accuracy(model, features))


	def test_dependency_rule_train(self, write_dependency_rule = False):
		model = pickle.load(open('model/dependency_rule.model', 'rb'))
		dict_dependency_rule = util.load_dependency_rule_dict(config.DEPENDENCY_RULES, config.USE_DEPENDENCY_RULE_NO)

		if write_dependency_rule:
			feature_functions.write_dependency_rule_by_line(self.file_name)

		with codecs.open('tmp/dep_rule_%s.txt'% (self.file_name), 'r', encoding = 'utf-8') as file:
			dependency_rule_by_relation = file.read().split('\n')

		feature = []
		for index, relation in enumerate(self.all_relations):
			feat = {}
			feat.update( feature_functions.dependency_rules(dependency_rule_by_relation[index], dict_dependency_rule) )

			num_sense = util.map_sense_to_number(relation['Sense'][0])
			if num_sense != -1:
				feature.append( (feat, num_sense) )

		print( 'Accuracy is: %1.10f' % nltk.classify.util.accuracy(model, feature))


	def test_together_train(self, write_parse_tree = True, write_dependency_rule = True):
		'''
		Phase 1. Preparation
		'''
		# load word pair dict
		self.dict_word_pairs = util.load_dict_word_pairs(config.WORD_PAIRS, config.USE_WORD_PAIR_NO)

		if write_parse_tree or write_dependency_rule:
			print('Step1: Pre-extract feature\n-----------------------------')
		else:
			print('Step1: Pre-extract feature skipped\n-----------------------------')

		# prepare production rule
		if write_parse_tree:
			util.write_parse_tree_to_file(self.file_name)

		arg1_production_rule_dict = util.load_production_rule_dict(config.ARG1_PRODUCTOIN_RULE, config.USE_DEPENDENCY_RULE_NO)
		arg2_production_rule_dict = util.load_production_rule_dict(config.ARG2_PRODUCTOIN_RULE, config.USE_DEPENDENCY_RULE_NO)
		both_production_rule_dict = util.load_production_rule_dict(config.BOTH_PRODUCTOIN_RULE, config.USE_DEPENDENCY_RULE_NO)

		with codecs.open('tmp/arg1_parsetree.txt', encoding = 'utf-8') as file:
			arg1_parsetree = file.read().split('\n')

		with codecs.open('tmp/arg2_parsetree.txt', encoding = 'utf-8') as file:
			arg2_parsetree = file.read().split('\n')
		
		# prepare dependency rule
		dict_dependency_rule = util.load_dependency_rule_dict(config.DEPENDENCY_RULES, config.USE_DEPENDENCY_RULE_NO)

		if write_dependency_rule:
			feature_functions.write_dependency_rule_by_line(self.file_name)

		with codecs.open('tmp/dep_rule_%s.txt'% (self.file_name), 'r', encoding = 'utf-8') as file:
			dependency_rule_by_relation = file.read().split('\n')

		'''
		Phase 2. Extract feature
		'''
		print('\nStep2: extract feature\n-----------------------------')
		model = pickle.load(open(config.MODEL, 'rb'))
		features = []
		multi_sense_no = 0
		correct_no = 0
		pred_label = []

		start = time.time()
		for index, relation in enumerate(self.all_relations):
			if relation['Type'] == 'EntRel':
				pred_label.append('EntRel')
				continue

			feat = {}
			# Phase 2.1 Extract word pair feature
			feat.update(feature_functions.word_pairs(relation, self.dict_word_pairs))

			# Phase 2.2 Extract production rule feature
			feat.update( feature_functions.production_rules
							(index, 
								[arg1_production_rule_dict, arg2_production_rule_dict, both_production_rule_dict], 
								[arg1_parsetree, arg2_parsetree]
 							)
						)

			# Phase 2.3 Extract dependency rule feature
			feat.update( feature_functions.dependency_rules(dependency_rule_by_relation[index], dict_dependency_rule) )
			pred_label.append( model.classify(feat) )

			'''
			if len(relation['Sense']) == 1:
				#if num_sense >= 1 and num_sense <= 7:
				if util.valid_sense(relation['Sense'][0]):
					features.append( (feat, num_sense) )
				else:
					features.append( (feat, 4))
			else:
				one_feature = []
				for sense in relation['Sense']:
					if util.valid_sense(sense):
						one_feature.append( (feat, util.map_sense_to_number(sense)) )
					else:
						one_feature.append( (feat, 4) )
				#one_feature = [ (feat, util.map_sense_to_number(sense)) for sense in relation['Sense'] \
				#					if util.map_sense_to_number(sense) in range(1,8)]

				multi_sense_no += 1
				if util.predict_correct(model, one_feature) > 0:
					correct_no += 1
				'''
		pass
		end = time.time()
		print('Extract features cost %1.10fs' % (end-start))

		'''
		Phase 3. Predict
		'''
		print('\nStep3: predict\n-----------------------------')
		correct_no += util.predict_correct(model, features)

		with codecs.open('%s_predict.json' % self.file_name, 'w', encoding = 'utf8', errors = 'ignore') as file:
			write_data = range(len(pred_label))
			for index, plabel in enumerate(pred_label):
				write_data[index] = self.all_relations[index]
				if util.valid_sense(plabel):
					write_data[index]['Sense'] = [util.map_number_to_sense(plabel)]
				else:
					write_data[index]['Sense'] = [util.map_number_to_sense(4)]

			write_data = [json.dumps(wd) for wd in write_data]
			file.write('\n'.join((write_data)))

		#print( 'Accuracy is: %1.10f' % (correct_no*1.0 / (multi_sense_no + len(features)) ) )
		#print( 'Accuracy is: %1.10f' % nltk.classify.util.accuracy() )
		#util.individual_predict(model, features)


def test_together():
	helpMsg = \
'''Usage: tester.py [-p] [-d] test_file_name

Optional parameters:
  -p : disable writing production rules
  -d : disable writing dependency rules
  -h --help : print help message'''

	write_drule = write_prule = True

	try:
		options, arg = getopt.getopt(sys.argv[1:], 'hdp', ['help'])
	except getopt.GetoptError:
		print(helpMsg)
		quit()

	for name, value in options:
		if name in ['-h', '--help']:
			print(helpMsg)
			quit()
		elif name == '-d':
				write_drule = False
		elif name == '-p':
			write_prule = False

	if len(arg) != 1:
		print(helpMsg)
		quit()

	test_file_name = arg[0]
	Tester(test_file_name).test_together_train(write_parse_tree = write_prule, write_dependency_rule = write_drule)

if __name__ == '__main__':
	Tester('dev_pdtb.json').test_word_pair_train()
	
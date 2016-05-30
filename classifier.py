# -*- coding: utf-8 -*- 
import util, config, feature_functions
import nltk, codecs, json, pickle, time, sys


class ImplicitDiscourseClassifierModelTrainer:
	def __init__(self, fname):
		self.trainData = []
		self.file_name = fname
		self.dict_word_pairs = {}
		self.dict_production_rules = {}

	def train_together(self, **kargs):
		'''
		Phase 1. Preprocess, preparation
		'''

		
		## Phase 1.1 Load word pair dict
		self.dict_word_pairs = util.load_dict_word_pairs(config.WORD_PAIRS, config.USE_WORD_PAIR_NO)

		## Phase 1.2 Prepare for production rule
		arg1_production_rule_dict = util.load_production_rule_dict(config.ARG1_PRODUCTOIN_RULE, config.USE_PRODUCTION_RULE_NO)
		arg2_production_rule_dict = util.load_production_rule_dict(config.ARG2_PRODUCTOIN_RULE, config.USE_PRODUCTION_RULE_NO)
		both_production_rule_dict = util.load_production_rule_dict(config.BOTH_PRODUCTOIN_RULE, config.USE_PRODUCTION_RULE_NO)

		with codecs.open(config.ARG1_PARSETREE) as file:
			arg1_parsetree = file.read().split('\n')

		with codecs.open(config.ARG2_PARSETREE) as file:
			arg2_parsetree = file.read().split('\n')

		##Phase 1.3 Prepare for dependency rule
		dependency_rule_dict = util.load_dependency_rule_dict(config.DEPENDENCY_RULES, config.USE_DEPENDENCY_RULE_NO)
		self.trainData = util.read_data_utf8(self.file_name)
		with codecs.open('dict/dependency_rule_by_relation.txt', 'r', encoding = 'utf-8', errors = 'ignore') as file:
			dependency_rule_by_relation = file.read().split('\n')

		'''
		Phase 2. Extract features, including word pair, production rule and dependency rule
		'''
		self.trainData = util.read_data_utf8(self.file_name)
		start = time.time()
		feature = []
		for index, relation in enumerate(self.trainData):
			if index % 1000 == 0: print index
			feat = {}

			# Phase 2.1 Extract word pair feature
			feat.update(feature_functions.word_pairs(relation, self.dict_word_pairs))

			# Phase 2.2 Extract production rule feature
			feat.update(	\
				feature_functions._train_production_rules(index, \
					[ arg1_production_rule_dict, arg2_production_rule_dict, both_production_rule_dict ], \
					[arg1_parsetree, arg2_parsetree])
					)

			# Phase 2.3 Extract dependency rule feature
			feat.update(feature_functions.dependency_rules(dependency_rule_by_relation[index], dependency_rule_dict))

			# Phase 2.4 Combine all the features
			for sense in relation['Sense']:
				num_sense = util.map_sense_to_number(sense)
				if num_sense != -1:
					feature.append( (feat,  num_sense) )

		end=time.time()
		print('extract word pair costs: %f seconds' %(end-start))

		'''
		Phase 3. Trian model and store model into hard disk
		'''
		start = time.time()
		try:
			model = nltk.MaxentClassifier.train(feature, **kargs)
		except:
			util.store_model(model, 'model/altogether_wp_pr_dr.model')
		end = time.time()
		print('train model costs: %f seconds' %(end-start))

		util.store_model(model, 'model/altogether_wp_pr_dr.model')
		return model

	def word_pair_train(self, **kargs):	
		self.dict_word_pairs = util.load_dict_word_pairs(config.WORD_PAIRS, 500)

		"""
			:type feature: list(tuple(dict, str/int))
										^		^
										|		|
									  feature  sense
		"""
		
		self.trainData = util.read_data(self.file_name)

		start = time.time()
		feature = []
		for relation in self.trainData:
			feat = {}
			feat.update(feature_functions.word_pairs(relation, self.dict_word_pairs))

			for sense in relation['Sense']:
				num_sense = util.map_sense_to_number(sense)
				if num_sense != -1:
					feature.append( (feat,  num_sense) )

		end=time.time()
		print('extract word pair costs: %f seconds' %(end-start))
			
		start = time.time()
		model = nltk.MaxentClassifier.train(feature, **kargs)
		end = time.time()
		print('train word pair model costs: %f seconds' %(end-start))

		util.store_model(model, 'model/word_pair.model')
		return model
		"""
			alldata = util.read_data('dev_pdtb.json')
			for data in alldata:
				attr = {}
				attr['WordPair'] = Feature(word_pair(data))
				#attr['ProductionRule'] = 
				feat = [(attr, s) for s in data['Sense']]
		"""
		

		#classifier = nltk.classify.MaxentClassifier.train(feat)


	def production_rule_train(self, **kargs):
		arg1_production_rule_dict = util.load_production_rule_dict(config.ARG1_PRODUCTOIN_RULE, config.USE_PRODUCTION_RULE_NO)
		arg2_production_rule_dict = util.load_production_rule_dict(config.ARG2_PRODUCTOIN_RULE, config.USE_PRODUCTION_RULE_NO)
		both_production_rule_dict = util.load_production_rule_dict(config.BOTH_PRODUCTOIN_RULE, config.USE_PRODUCTION_RULE_NO)

		with codecs.open(config.ARG1_PARSETREE) as file:
			arg1_parsetree = file.read().split('\n')
			#for line in file:
			#	arg1_parsetree.append(line)

		with codecs.open(config.ARG2_PARSETREE) as file:
			arg2_parsetree = file.read().split('\n')

		feature = []

		self.trainData = util.read_data(self.file_name)
		for index, relation in enumerate(self.trainData):
			feat = {}
			feat.update(	\
				feature_functions._train_production_rules(index, \
					[ arg1_production_rule_dict, arg2_production_rule_dict, both_production_rule_dict ], \
					[arg1_parsetree, arg2_parsetree])
					)

			for sense in relation['Sense']:
				num_sense = util.map_sense_to_number(sense)
				if num_sense != -1:
					feature.append( (feat,  num_sense) )

		start = time.time()
		try:
			model = nltk.MaxentClassifier.train(feature, **kargs)
		except:
			util.store_model(model, 'model/production_rule.model')

		end = time.time()
		print('train production rules model costs: %f seconds' %(end-start))

		util.store_model(model, 'model/production_rule.model')
		return model


	def dependency_rule_train(self, **kargs):
		dependency_rule_dict = util.load_dependency_rule_dict(config.DEPENDENCY_RULES, config.USE_DEPENDENCY_RULE_NO)
		self.trainData = util.read_data_utf8(self.file_name)
		with codecs.open('dict/dependency_rule_by_relation.txt', 'r', encoding = 'utf-8', errors = 'ignore') as file:
			dependency_rule_by_relation = file.read().split('\n')
		
		'''
		from nltk.parse.stanford import StanfordDependencyParser
		jar = 'lib/stanford-parser-full-2015-12-09/stanford-parser.jar'
		models_jar = 'lib/stanford-parser-full-2015-12-09/stanford-parser-3.6.0-models.jar'
		dependency_parser = StanfordDependencyParser(path_to_jar = jar, path_to_models_jar = models_jar, java_options='-mx3000m')
 		'''

		feature = []
		for index, relation in enumerate(self.trainData):
			feat = {}
			feat.update(feature_functions.dependency_rules(dependency_rule_by_relation[index], dependency_rule_dict))

			for sense in relation['Sense']:
				num_sense = util.map_sense_to_number(sense)
				if num_sense <= 7:
					feature.append( (feat, num_sense) )

		start = time.time()
		model = nltk.MaxentClassifier.train(feature, **kargs)
		end = time.time()
		print('train dependency rules model costs: %f seconds' %(end-start))

		util.store_model(model, 'model/dependency_rule.model')
		return model

class Tester:
	def __init__(self, fname, **kargs):
		self.file_name = fname
		self.all_relations = util.read_data_utf8(self.file_name)


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
		
		#print features
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


	def test_together_train(self, write_parse_tree = False, write_dependency_rule = False):
		'''
		Phase 1. Preparation
		'''
		# load word pair dict
		self.dict_word_pairs = util.load_dict_word_pairs(config.WORD_PAIRS, config.USE_WORD_PAIR_NO)

		# prepare production rule
		if write_parse_tree:
			util.write_parse_tree_to_file('dev_pdtb.json')

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
		model = pickle.load(open(config.MODEL, 'rb'))
		features = []
		multi_sense_no = 0
		correct_no = 0
		pred_label = []

		start = time.time()
		for index, relation in enumerate(self.all_relations):
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
		pass
		end = time.time()
		print('Extract features cost %1.10fs' % (end-start))

		'''
		Phase 3. Predict
		'''
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
			print(len(write_data))
			file.write('\n'.join((write_data)))

		print( 'Accuracy is: %1.10f' % (correct_no*1.0 / (multi_sense_no + len(features)) ) )
		#util.individual_predict(model, features)

	
if __name__ == '__main__':
	#model = ImplicitDiscourseClassifierModelTrainer('train_pdtb.json').word_pair_train(max_iter=15)
	#model = ImplicitDiscourseClassifierModelTrainer('train_pdtb.json').production_rule_train(max_iter=10)
	#model = ImplicitDiscourseClassifierModelTrainer('train_pdtb.json').dependency_rule_train(max_iter=10)
	model = ImplicitDiscourseClassifierModelTrainer('train_pdtb.json').train_together(max_iter = 30)
	print model
	model.show_most_informative_features()
	#Tester('dev_pdtb.json').test_word_pair_train()
	#Tester('dev_pdtb.json').test_production_rule_train()
	#Tester('dev_pdtb.json').test_dependency_rule_train()
	Tester('dev_pdtb.json').test_together_train()
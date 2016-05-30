# -*- coding: utf-8 -*- 
import config
import json, codecs, pickle, yaml, os, time
from PorterStemmer import PorterStemmer
from nltk.tree import Tree
from byteify import json_loads_byteified, json_load_byteified, byteify

#KEYS = [u'DocID', u'Arg1', u'Arg2', u'Connective', u'Sense', u'Type', u'ID']
"""
DocID is a list of unicode string
Arg1 and Arg2: dict in the format of 
	  {'RawText' : unicode string,
		 'Lemma' : list of unicode string,
		 'Word' : list of unicode string,
		 'NER': list of string, named entity recognizer}
Connective: dict of connective in the format of
		{ 'RawText' : list}
Sense: list
Type: Unicode, can be Explicit, Implicit etc
ID: Unicode

Example:
{"Sense": ["Expansion.Restatement"], 
"Type": "Implicit", 
"DocID": "wsj_2200", 
"Arg2": {
	"Word": ["to", "restrict", "the", "RTC", "to", "Treasury", "borrowings", "only", ",", "unless", "the", "agency", "receives", "specific", "congressional", "authorization"], 
	"POS": ["TO", "VB", "DT", "NNP", "TO", "NNP", "NNS", "RB", ",", "IN", "DT", "NN", "VBZ", "JJ", "JJ", "NN"], 
	"RawText": "to restrict the RTC to Treasury borrowings only, unless the agency receives specific congressional authorization", 
	"NER": ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"], 
	"Lemma": ["to", "restrict", "the", "RTC", "to", "Treasury", "borrowing", "only", ",", "unless", "the", "agency", "receive", "specific", "congressional", "authorization"]}, 
	"Connective": {"RawText": ["specifically"]
	}, 
"Arg1": {
	"Word": ["The", "bill", "would", "prevent", "the", "Resolution", "Trust", "Corp.", "from", "raising", "temporary", "working", "capital", "by", "having", "an", "RTC-owned", "bank", "or", "thrift", "issue", "debt", "that", "would", "n't", "be", "counted", "on", "the", "federal", "budget"], 
	"POS": ["DT", "NN", "MD", "VB", "DT", "NNP", "NNP", "NNP", "IN", "VBG", "JJ", "JJ", "NN", "IN", "VBG", "DT", "JJ", "NN", "CC", "NN", "NN", "NN", "WDT", "MD", "RB", "VB", "VBN", "IN", "DT", "JJ", "NN"], 
	"RawText": "The bill would prevent the Resolution Trust Corp. from raising temporary working capital by having an RTC-owned bank or thrift issue debt that wouldn't be counted on the federal budget", 
	"NER": ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"], 
	"Lemma": ["the", "bill", "would", "prevent", "the", "Resolution", "Trust", "Corp.", "from", "raise", "temporary", "working", "capital", "by", "have", "a", "rtc-owned", "bank", "or", "thrift", "issue", "debt", "that", "would", "not", "be", "count", "on", "the", "federal", "budget"]
	}, 
"ID": "35946"
}
"""


"""
	:type sense: string
	:rtype : int
"""
def map_sense_to_number(sense):
	if sense == 'Expansion.List':
		return 1
	elif sense == 'Expansion.Conjunction':
		return 2
	elif sense == 'Expansion.Instantiation':
		return 3	
	elif sense == 'Contingency.Cause':
		return 4
	elif sense == 'Temporal.Asynchronous':
		return 5
	elif sense == 'Comparison.Contrast':
		return 6
	elif sense == 'Expansion.Restatement':
		return 7
	elif sense == 'Temporal.Synchrony':
		return 8
	elif sense == 'Contingency.Pragmatic cause':
		return 9
	elif sense == 'Comparison.Concession':
		return 10
	elif sense == 'Expansion.Alternative':
		return 11
	else:
		return -1
	# 'EntRel' is excluded, because an implicit connective cannot be inserted between adjacent sentences
	#elif sense == 'EntRel':

	# too few sample, ignored		
	#else sense == 'Contingency.Condition':
	#Contingency.Pragmatic Condition
	#Comparison.Pragmatic Contrast
	#Comparison.Pragmatic Concession
	#Expansion.Exception

def map_number_to_sense(num):
	sense = ['None', 'Expansion.List', 'Expansion.Conjunction', 'Expansion.Instantiation', 'Contingency.Cause', \
	'Temporal.Asynchronous', 'Comparison.Contrast', 'Expansion.Restatement', 'Temporal.Synchrony', \
	'Contingency.Pragmatic cause', 'Comparison.Concession''Expansion.Alternative' ]

	if num >= len(sense):
		return 'Error'
	else:
		return sense[num]

def individual_predict(classifier, gold):
	results = classifier.classify_many([fs for (fs, l) in gold])

	stat = [{}]*8
	for all_type in range(1,8):
		TP = FN = FP = TN = 0
		for index, pred_lable in enumerate(results):
			gold_lable = gold[index][1]
			if gold_lable == all_type: # label is positive
				if pred_lable == gold_lable: # predict positive
					TP += 1
				else:	#predict negative
					FN += 1
			else: # label is negative
				if pred_lable == all_type:
					FP += 1
				else:
					TN += 1
		p = TP * 1.0 / (TP + FP)
		r = TP * 1.0 / (TP + FN)
		print TP, FP, TN, FN
		print 'precision',p
		print 'recall', r

		stat[all_type]['TP'] = TP
		stat[all_type]['TN'] = TN
		stat[all_type]['FP'] = FP
		stat[all_type]['FN'] = FN
		stat[all_type]['Precision'] = p
		stat[all_type]['Recall'] = r
		if r+p == 0:
			stat[all_type]['F1'] = 0
		else:
			stat[all_type]['F1'] = 2 * p * r / ( r + p)

	for t in range(1,8):
		print( 'precision=%s\trecall=%s\tF1=%s' % (stat[t]['Precision'], stat[t]['Recall'], stat['F1']) )


"""
Read json file and return the corresponding json object for further process

	:type file_name : string
	:rtype data : list of dict, each element is a json object(in the format of dict)
"""
def read_data(file_name):
	data = []
	with codecs.open(file_name, 'r', encoding = 'utf-8') as f:
		for line in f:
			# filter only 'Type' : 'Implicit'
			obj = json.loads(line)
			#obj = yaml.safe_load(line)
			#obj = json_loads_byteified(line)
			#obj = byteify(json.loads(line))

			if obj['Type'] == 'Implicit':
				#data.append(convert_unicode_json_object_to_str(obj))
				data.append(obj)

	return data

def read_data_utf8(file_name):
	data = []
	rest_data = []
	with codecs.open(file_name, 'r', encoding = 'utf-8', errors = 'ignore') as f:
		for line in f:
			obj = json.loads(line)
			contain_rest = False
			if obj['Type'] == 'Implicit':
				'''
				for s in obj['Sense']:
					if map_sense_to_number(s) > 7:
						rest_data.append(obj)
						contain_rest = True
						break
				if not contain_rest:
					data.append(obj)
				'''
				data.append(obj)

	return data# + rest_data


def read_all_implicit_data_utf8(file_name):
	data = []
	rest_data = []
	with codecs.open(file_name, 'r', encoding = 'utf-8', errors = 'ignore') as f:
		for line in f:
			obj = json.loads(line)
			contain_rest = False
			if obj['Type'] == 'Implicit':
				for s in obj['Sense']:
					if map_sense_to_number(s) > 7:
						rest_data.append(obj)
						contain_rest = True
						break
				if not contain_rest:
					data.append(obj)
				data.append(obj)

	return data + rest_data


def read_rest_data(file_name):
	data = []
	with codecs.open(file_name, 'r', encoding = 'utf-8', errors = 'ignore') as f:
		for line in f:
			obj = json.loads(line)

			if obj['Type'] == 'Implicit':
				for s in obj['Sense']:
					if map_sense_to_number(s) > 7:
						data.append(obj)
						break

	return data

def load_dict_word_pairs(file_name, length=-1):
	dict_word_pairs = {}
	word_pairs_file = open(file_name)

	for lineno, line in enumerate(word_pairs_file):
		if line == '':
			continue
		if lineno == length:
			break
		dict_word_pairs[line[:-1]] = lineno

	return dict_word_pairs


def write_word_pairs_to_file():
	file_name = config.TRAINSET_PATH
	dict_word_pairs = get_word_pair_from_file_with_count(file_name)
	
	wp_file = codecs.open(config.WORD_PAIRS, 'w', encoding='utf-8')
	
	write_data = [ wp[0] for wp in sorted(dict_word_pairs.items(), key=lambda v:v[1], reverse = True) ] #key is value of dict_word_pairs]

	wp_file.write(u'\n'.join(write_data))

	wp_file.close()


def store_model(model, fname):
	pickle.dump(model, open(fname, 'wb'), -1)#with highest protocol


def get_production_rule_from_file_with_count():
	arg1_parsetree_file = codecs.open('dict/arg1_parsetree.txt')
	arg2_parsetree_file = codecs.open('dict/arg2_parsetree.txt')
	#parsetree_file = codecs.open(config.PARSETREE_DICT, 'w')

	arg1_parsetree = arg1_parsetree_file.read().split('\n')
	arg2_parsetree = arg2_parsetree_file.read().split('\n')

	arg1_production_rule_dict = {}
	arg2_production_rule_dict = {}
	both_production_rule_dict = {}
	for index in range(len(arg1_parsetree)):
		arg1_prule = get_production_rule_by_parse_tree(arg1_parsetree[index])[1:]
		arg2_prule = get_production_rule_by_parse_tree(arg2_parsetree[index])[1:]
		both_prule = list(set(arg1_prule) & set(arg2_prule))
		"""print arg1_prule
								print arg2_prule
								print both_prule"""

		for prule in arg1_prule:
			if prule in arg1_production_rule_dict:
				arg1_production_rule_dict[prule] += 1
			else:
				arg1_production_rule_dict[prule] = 1
		for prule in arg2_prule:
			if prule in arg2_production_rule_dict:
				arg2_production_rule_dict[prule] += 1
			else:
				arg2_production_rule_dict[prule] = 1

		for prule in both_prule:
			if prule in both_production_rule_dict:
				both_production_rule_dict[prule] += 1
			else:
				both_production_rule_dict[prule] = 1

	arg1_production_rule = [ 'Arg1_' + pr[0] for pr in sorted(arg1_production_rule_dict.items(), key = lambda it:it[1], reverse = True) ]
	arg2_production_rule = [ 'Arg2_' + pr[0] for pr in sorted(arg2_production_rule_dict.items(), key = lambda it:it[1], reverse = True) ]
	both_production_rule = [ 'Both_' + pr[0] for pr in sorted(both_production_rule_dict.items(), key = lambda it:it[1], reverse = True)]

	#TODO merge all produciton rules
	#production_rule_dict = arg1_production_rule_dict + arg2_production_rule_dict + both_production_rule_dict
	#prodcution_rule = 

	arg1_production_rule_file = open('dict/arg1_production_rules.txt', 'w')
	arg2_production_rule_file = open('dict/arg2_production_rules.txt', 'w')
	both_production_rule_file = open('dict/both_production_rules.txt', 'w')
	#production_rule_file = open('dict/production_rules.txt', 'w')
	arg1_production_rule_file.write('\n'.join(arg1_production_rule))
	arg2_production_rule_file.write('\n'.join(arg2_production_rule))
	both_production_rule_file.write('\n'.join(both_production_rule))

	#production_rule_file.close()
	arg2_production_rule_file.close()
	arg1_production_rule_file.close()
	arg1_parsetree_file.close()
	arg2_parsetree_file.close()
	#parsetree_file.close()

def get_production_rule_by_parse_tree(parsetree):
	syntax_tree = Tree.fromstring(parsetree)

	convert_str_format = lambda string, strip_char='\'': \
		''.join( [ ch for ch in '->'.join( [ st.strip() for st in string.split('->')] ) if ch not in strip_char ] )

	production_rule = [ convert_str_format(str(pr)) for pr in syntax_tree.productions() ]

	return production_rule


def load_dependency_rule_dict(file_name, length = -1):
	dependency_rule_dict = {}
	with codecs.open(file_name, 'r', encoding = 'utf-8') as file:
		for lineno, line in enumerate(file):
			if lineno == length:
				break
			else:
				dependency_rule_dict[line[:-1]] = lineno
	
	return dependency_rule_dict


def load_production_rule_dict(file_name, length = -1):
	dict_production_rules = {}
	with codecs.open(file_name, 'r', encoding = 'utf-8') as file:
		for lineno, line in enumerate(file):
			if lineno == length:
				break
			else:
				dict_production_rules[line[:-2]] = lineno
	
	return dict_production_rules


def write_parse_tree_to_file(file_name):
	all_relations = read_data_utf8(file_name)
	dict = {}
	arg1_sent = []
	arg2_sent = []
	arg1_sent_file_path = 'tmp/arg1_sentence.txt'
	arg2_sent_file_path = 'tmp/arg2_sentence.txt'
	#arg1_prule_file = 'tmp/arg1_production_rule.txt'
	#arg2_prule_file = 'tmp/arg2_production_rule.txt'
	#both_prule_file = 'tmp/both_'
	for relation in all_relations:
		arg1_sent.append( ' '.join(relation['Arg1']['Lemma']) )
		arg2_sent.append( ' '.join(relation['Arg2']['Lemma']) )

	with codecs.open(arg1_sent_file_path, 'w', encoding = 'utf-8') as file:
		file.write( u'\n'.join(arg1_sent) )

	with codecs.open(arg2_sent_file_path, 'w', encoding = 'utf-8') as file:
		file.write( u'\n'.join(arg2_sent) )

	start = time.time()
	os.system( 'java -jar lib/BerkeleyParser-1.7.jar -gr lib/eng_sm6.gr -inputFile %s -outputFile tmp/arg1_parsetree.txt' 
		% arg1_sent_file_path )
	end = time.time()

	print 'extracting parse tree of all arg1 cost %f' % (end-start)

	start = time.time()
	os.system( 'java -jar lib/BerkeleyParser-1.7.jar -gr lib/eng_sm6.gr -inputFile %s -outputFile tmp/arg2_parsetree.txt'
		% arg2_sent_file_path )
	end = time.time()

	print 'extracting parse tree of all arg2 cost %f' % (end-start)


def __deprecated_get_production_rule_from_file_with_count():
	all_relations = read_data_utf8(config.TRAINSET_PATH)
	#punctuation = ['.', ',', '!', '"', '#', '&', '\'', '*', '+', '-', '...', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\',\
	#	']', '^', '_', '`', '|', '~', '``' + '\'\'']

	dict = {}
	arg1_sent = []
	arg2_sent = []
	arg1_sent_file = 'dict/arg1_sentence.txt'
	arg2_sent_file = 'dict/arg2_sentence.txt'
	arg1_prule_file = config.ARG1_PRODUCTOIN_RULE_PATH
	arg2_prule_file = config.ARG2_PRODUCTOIN_RULE_PATH
	both_prule_file = config.BOTH_PRODUCTOIN_RULE_PATH

	def foo():
		for index, relation in enumerate(all_relations):
			arg1_sent.append( ' '.join(relation['Arg1']['Lemma']) ) 
			arg2_sent.append( ' '.join(relation['Arg2']['Lemma']) )

		with codecs.open(arg1_sent_file, 'w', encoding = 'utf-8') as f1:
			f1.write(u'\n'.join(arg1_sent))
		with codecs.open(arg2_sent_file, 'w', encoding = 'utf-8') as f2:
			f2.write(u'\n'.join(arg2_sent))

		arg1_sentence = codecs.open(arg1_sent_file, 'r', encoding = 'utf-8')
		arg2_sentence = codecs.open(arg2_sent_file, 'r', encoding = 'utf-8')

	arg1_ptree_file_path = 'dict/arg1_parsetree.txt'
	arg2_ptree_file_path = 'dict/arg2_parsetree.txt'
	#os.system( 'java -jar lib/BerkeleyParser-1.7.jar -gr lib/eng_sm6.gr -nThreads 8 -inputFile %s -outputFile %s' \
	#	% (arg1_sent_file, arg1_ptree_file_path) )
	os.system( 'java -jar lib/BerkeleyParser-1.7.jar -gr lib/eng_sm6.gr -nThreads 8 -inputFile %s -outputFile %s' \
		% (arg2_sent_file, arg2_ptree_file_path) )
	

"""
	ignore any word pair with symbols
"""
def get_word_pair_from_file_with_count(fname):
	all_relations = read_data(fname)
	punctuation = ['.', ',', '!', '"', '#', '&', '\'', '*', '+', '-', '...', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\',\
		']', '^', '_', '`', '|', '~', '$', '%', '--', '``', '\'\'']
	dict = {}
	for relation in all_relations:
		for a1 in relation['Arg1']['Lemma']:
			for a2 in relation['Arg2']['Lemma']:
				if a1 in punctuation or a2 in punctuation or a1[0] in '0123456789' or a2[0] in '0123456789':
					pass
				else:
					#pair = '%s|%s' % (stem_string(a1), stem_string(a2))
					pair = '%s|%s' % (a1, a2)
					if pair in dict:
						dict[pair] += 1
					else:
						dict[pair] = 1

	return dict

'''
	deprecated
'''
def get_dependency_rule_from_relation(relation, parser):
	arg1_sent = ' '.join(relation['Arg1']['Lemma']).encode('utf8').replace('\xc2\xa0', '').split('.')
	arg2_sent = ' '.join(relation['Arg2']['Lemma']).encode('utf8').replace('\xc2\xa0', '').split('.')
	
	parse_result = parser.raw_parse_sents(arg1_sent + arg1_sent)

	prule = []
	for result in parse_result:
		for t in result:
			for node in range(len(t.nodes)):
				if t.nodes[node]['word'] == None or t.nodes[node]['deps'].items() == []:
					continue
				else:
					prule.append( '%s<-%s' % (t.nodes[node]['word'], ' '.join( [ key for key, val in t.nodes[node]['deps'].items() ] )))	

	return prule


def write_dependency_rule_sorted():
	with codecs.open('dict/dependency_rule_by_relation.txt', 'r', encoding = 'utf8', errors = 'ignore') as file:
		dependency_rules = file.read().split('\n')
	
	dependency_rule_dict = {}
	for drule_by_relation in dependency_rules:
		rules = drule_by_relation.split('||')
		for rule in rules:
			if rule in dependency_rule_dict:
				dependency_rule_dict[rule] += 1
			else:
				dependency_rule_dict[rule] = 1
	sorted_drule_list = [item[0] for item in sorted(dependency_rule_dict.items(), key = lambda v : v[1], reverse = True) ]

	with codecs.open(config.DEPENDENCY_RULES, 'w', encoding = 'utf8', errors = 'ignore') as file:
		#file.write( '\n'.join([ '%s:%d'%(rule, dependency_rule_dict[rule]) for rule in sorted_drule_list]) )
		file.write( '\n'.join(sorted_drule_list) )

'''
	each line is dependency rules of a relation
'''
def write_dependency_rule_by_line(file_name):
	from nltk.parse.stanford import StanfordDependencyParser
	jar = 'lib/stanford-parser-full-2015-12-09/stanford-parser.jar'
	models_jar = 'lib/stanford-parser-full-2015-12-09/stanford-parser-3.6.0-models.jar'
	dependency_parser = StanfordDependencyParser(path_to_jar = jar, path_to_models_jar = models_jar, java_options='-mx3000m')

	all_relations = read_data_utf8(file_name)

	print( 'len of all relations: %d' % (len(all_relations)) )
	sentences = []
	lineno = 0
	line_interval = []
	for idx, relation in enumerate(all_relations):
		_from = lineno

		lines = []
		sent = []
		if '.' in relation['Arg1']['Lemma']:
			for word in relation['Arg1']['Lemma']:
				if word == '.':
					lines.append(' '.join(sent).encode('utf8').replace('\xc2\xa0', ''))
					sent = []
				else:
					sent.append(word)
			lines.append(' '.join(sent).encode('utf8').replace('\xc2\xa0', ''))
		else:
			lines.append(' '.join(relation['Arg1']['Lemma']).encode('utf8').replace('\xc2\xa0', ''))
		
		_to = _from + len(lines)

		sentences += lines
		lines = []
		sent = []
		if '.' in relation['Arg2']['Lemma']:
			for word in relation['Arg2']['Lemma']:
				if word == '.':
					lines.append(' '.join(sent).encode('utf8').replace('\xc2\xa0', ''))
					sent = []
				else:
					sent.append(word)
			lines.append(' '.join(sent).encode('utf8').replace('\xc2\xa0', ''))
		else:
			lines.append(' '.join(relation['Arg2']['Lemma']).encode('utf8').replace('\xc2\xa0', ''))

		_to += len(lines)
		sentences += lines
		lineno = _to
		line_interval.append( (_from, _to ) )
	pass
	for idx, pair in enumerate(line_interval):
		print( '(%d:%d)' % (pair[0],pair[1]) )
		for i in range(pair[0],pair[1]):
			print( '%d:%s' % (i,sentences[i]) )
	
	print( 'len of sentences: %d' % ( len(sentences) ) )

	line_interval_idx = 0
	count = 0
	'''
		each result is correspoding to a sentence
		a line_interval [from, to)
	'''
	relation_length = len(all_relations)
	all_part = 5
	for part in range(all_part+1):
		_from = part * (relation_length / all_part) # inclusive
		if _from >= relation_length:
			break
		_to = min( (part+1) * (relation_length / all_part) -1, relation_length - 1 ) # inclusive
		print('part %d' % part)
		print('relation %d' % (_to - _from+1))

		to_parse_sentences = sentences[ line_interval[_from][0] : line_interval[_to][1] ]
		print('line of sentences %d' % ( len(to_parse_sentences) ) )

		start = time.time()
		parse_result = dependency_parser.raw_parse_sents(to_parse_sentences)
		end = time.time()
		print( 'cost %f' % (end - start) )

		dep_rule_list = []
		dep_rule_for_one_relation = []
		acutal_result_no = 0
		for result in parse_result:
			acutal_result_no += 1
			for t in result:
				for node in range(len(t.nodes)):
					if t.nodes[node]['word'] == None or t.nodes[node]['deps'].items() == []:
						continue
					else:
						dep_rule_for_one_relation.append( '%s<-%s' % \
							(t.nodes[node]['word'],	' '.join( [ key for key, val in t.nodes[node]['deps'].items() ] )))	
			if count == line_interval[line_interval_idx][1] - 1:
				print '%d: (%d, %d) finished' % (line_interval_idx, line_interval[line_interval_idx][0], line_interval[line_interval_idx][1])
				line_interval_idx += 1
				dep_rule_list.append(dep_rule_for_one_relation)
				dep_rule_for_one_relation = []
			
			count += 1
		print 'actual parse result no : %d' % acutal_result_no
		# last relation
		#print '%d: (%d, %d) finished' % (line_interval_idx, line_interval[line_interval_idx][0], line_interval[line_interval_idx][1])
		#line_interval_idx += 1
		#dep_rule_list.append(dep_rule_for_one_relation)

		write_data = []
		for dep_rules in dep_rule_list:
			write_data.append( '||'.join([rule for rule in dep_rules] ) )

		print('length of  write_data %d' % len(write_data))
		with codecs.open('tmp/dep_rule_%s_part%d.txt'% (file_name, part), 'w', encoding = 'utf-8') as file:
			file.write( u'\n'.join(write_data) )
	pass#for part in range(all_part) end


def stem_string(line):
    if line == "":
        return ""
    p = PorterStemmer()
    word = ""
    output = ""
    for c in line:
        if c.isalpha():
            word += c.lower()
        else:
            if word:
                #output += p.stem(word)
                output += p.stem(word)
                word = ''
            output += c.lower()
    if word:
        output += p.stem(word)

    return str(output)

def stem_list(list):
    return [stem_string(item) for item in list]


def get_set_word_pairs_from_data():
	filter = [',', '.', '\"', '\'\'', ':', '']
	set_word_pairs = set([])
	for data in trainData:
		for a1 in data['Arg1']['Word']:
			for a2 in data['Arg2']['Word']:
				if a1 in fileter or a2 in filter:
					pass
				else:
					word_pair = '%s|%s' % (a1, a2)
					set_word_pairs.add(word_pair)

	return set_word_pairs


def strip_parse_tree():
	arg1_parsetree_file = codecs.open('dict/arg1_parsetree.txt', encoding = 'utf-8')
	arg2_parsetree_file = codecs.open('dict/arg2_parsetree.txt', encoding = 'utf-8')
	parsetree_file = codecs.open(config.PARSETREE_DICT, 'w', encoding = 'utf-8')

	empty_lineno = []
	for lineno, line in enumerate(arg1_parsetree_file):
		if line == '(())\n':
			empty_lineno.append(lineno)

	for lineno, line in enumerate(arg2_parsetree_file):
		if line == '(())\n':
			empty_lineno.append(lineno)

	arg1_parsetree_file.close()
	arg2_parsetree_file.close()

	arg1_parsetree_file = codecs.open('dict/arg1_parsetree.txt', encoding = 'utf-8')
	arg2_parsetree_file = codecs.open('dict/arg2_parsetree.txt', encoding = 'utf-8')

	for lineno, line in enumerate(arg1_parsetree_file):
		if lineno not in empty_lineno:
			parsetree_file.write(line)

	for lineno, line in enumerate(arg2_parsetree_file):
		if lineno not in empty_lineno:
			parsetree_file.write(line)

	print empty_lineno
	parsetree_file.close()


def predict_correct(classifier, gold):
	results = classifier.classify_many([fs for (fs, l) in gold])

	correct = [l == r for ((fs, l), r) in zip(gold, results)]
	return sum(correct)

def analyze_data():
	all_relations = read_data_utf8('train_pdtb.json')
	from collections import defaultdict
	dict = defaultdict(int)
	count = 0
	for relation in all_relations:
		#if len(relation['Sense']) > 1:
			#print [s for s in relation['Sense'] if map_sense_to_number(s)==]

		if len(relation['Sense']) == 1 and map_sense_to_number(relation['Sense'][0])>7:
			count += 1
		if len(relation['Sense']) == 2 and map_sense_to_number(relation['Sense'][1]) > 7 and map_sense_to_number(relation['Sense'][0]) > 7:
			coutn+=1
		for sense in relation['Sense']:
			dict[sense] += 1
	print(count)
	print('total %d' % (sum([l[1] for l in dict.items()]) ) )
	print([(l[0], l[1]*1.0/13351) for l in dict.items()])


def deal_with_rest_data():
	with codecs.open('train_pdtb.json', encoding='utf8',errors='ignore') as file:
		from collections import defaultdict
		stat = defaultdict(int)
		data = []
		for no, line in enumerate(file):
			obj = json.loads(line)
			if obj['Type'] == 'Implicit':
				for s in obj['Sense']:
					if map_sense_to_number(s) > 7:
						data.append(line)
						stat[s] += 1
						break
	print stat
	with codecs.open('rest.json', 'w', encoding='utf8',errors='ignore') as file:
		file.write(''.join(data))

	#write_dependency_rule_by_line('rest.json')
	pass

	
def valid_sense(sense):
	if type(sense) == type('a'):
		num_sense = map_sense_to_number(sense)
	else:
		num_sense = sense

	if num_sense >= 1 and num_sense <= 7:
		return True
	else:
		return False


if __name__ == '__main__':
	analyze_data()
	deal_with_rest_data()
	"""
	relations = read_data('train_pdtb.json')
	sent_len = []
	for index, relation in enumerate(relations):
		leng = len(relation['Arg1']['Lemma'])
		if leng > 200 : print index
		sent_len.append(len(relation['Arg1']['Lemma']))
		leng = len(relation['Arg2']['Lemma'])
		if leng > 200 : print index
		sent_len.append(len(relation['Arg2']['Lemma']))
	"""
	#get_production_rule_from_file_with_count()
	#__deprecated_get_production_rule_from_file_with_count()
	#strip_prod_rule()
	#analyze_data()
	#write_word_pairs_to_file()
	#relations = read_data('dev_pdtb.json')
	#print(relations)
	pass
	
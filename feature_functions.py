# -*- coding: utf-8 -*- 
import util, config
#from syntax_tree import Syntax_tree
import pickle, string, codecs, time

def all_features(relation, parse_dict):
    feature_function_list = [
        word_pairs,
        production_rules, 
        dependency_rules,
        firstlast_first3
        # polarity,
        #modality,
        #verbs,
        #brown_cluster_pair,
        #Inquirer,
        #MPQA_polarity,
    ]

    features = [feature_function(relation, parse_dict) for feature_function in feature_function_list]
    # merge features
    feature = mergeFeatures(features)
    return feature


def get_word_pairs_from_relation(relation):
    punctuation = """!"#&'*+,-..../:;<=>?@[\]^_`|~""" + "``" + "''"
    ret = []
    for a1 in relation['Arg1']['Word']:
        for a2 in relation['Arg2']['Word']:
            if a1 in punctuation or a2 in punctuation:
                pass
            else:
                try:
                    s = '%s|%s' % (util.stem_string(a1), util.stem_string(a2)) 
                    ret.append(s)
                except:
                    pass

    return ret

def word_pairs(relation, dict):
    word_pairs = get_word_pairs_from_relation(relation)

    feat_dict = {}

    #initializatoin
    #for i in range(len(dict)):
    #    feat_dict[ 'wp(%d)' % i ] = 0

    for feat in word_pairs:
        if feat in dict:
            feat_dict['wp(%d)' % dict[feat]] = 1

    return feat_dict


def production_rules(rel_idx, production_rule_dict, parsetree_dict):
    arg1_production_rule_dict, arg2_production_rule_dict, both_production_rule_dict = production_rule_dict
    arg1_parsetree, arg2_parsetree = parsetree_dict

    arg1_prule = util.get_production_rule_by_parse_tree(arg1_parsetree[rel_idx])[1:]
    arg2_prule = util.get_production_rule_by_parse_tree(arg2_parsetree[rel_idx])[1:]
    both_prule = list( set(arg1_prule) & set(arg2_prule))

    arg1_production_rule = {}
    arg2_production_rule = {}
    both_production_rule = {}

    for rule in arg1_prule:
        string = 'Arg1_%s' % rule
        if string in arg1_production_rule_dict:
            arg1_production_rule[ 'Arg1_%d' % arg1_production_rule_dict[string] ] = 1

    for rule in arg2_prule:
        string = 'Arg2_%s' % rule
        if string in arg2_production_rule_dict:
            arg2_production_rule[ 'Arg2_%d' % arg2_production_rule_dict[string] ] = 1

    for rule in both_prule:
        string = 'Both_%s' % rule
        if string in both_production_rule_dict:
            both_production_rule[ 'Both_%d' % both_production_rule_dict[string] ] = 1

    ret = {}
    ret.update(arg1_production_rule)
    ret.update(arg2_production_rule)
    ret.update(both_production_rule)

    return ret


def _train_production_rules(relation_idx, parse_dict, parsetree):
    '''load dict '''
    #dict_production_rules = Non_Explicit_dict().dict_production_rules
    arg1_production_rule_dict = parse_dict[0]
    arg2_production_rule_dict = parse_dict[1]
    both_procution_rule_dict = parse_dict[2] 
    arg1_parsetree = parsetree[0]
    arg2_parsetree = parsetree[1]

    ''' feature '''
    arg1_prule = util.get_production_rule_by_parse_tree( arg1_parsetree[relation_idx] )[1:] #ignore '->S'
    arg2_prule = util.get_production_rule_by_parse_tree( arg2_parsetree[relation_idx] )[1:]

    both_prule = list(set(arg1_prule) & set(arg2_prule))

    arg1_production_rules = {}
    for rule in arg1_prule:
        string = 'Arg1_%s' % rule
        if string in arg1_production_rule_dict:
            arg1_production_rules['Arg1_%d' % arg1_production_rule_dict[string] ] = 1
            
    arg2_production_rules = {}
    for rule in arg2_prule:
        string = 'Arg2_%s' % rule
        if string in arg2_production_rule_dict:
            arg2_production_rules['Arg2_%d' % arg2_production_rule_dict[string] ] = 1

    both_production_rules = {}
    for rule in both_prule:
        string = 'Both_%s' % rule
        if string in arg1_production_rule_dict:
            both_production_rules['Both_%d' % both_production_rule_dict[string] ] = 1

    ret = {}
    ret.update(arg1_production_rules)
    ret.update(arg2_production_rules)
    ret.update(both_production_rules)
    return ret
    #return get_feature_by_feat_list(dict_production_rules, rules)


def dependency_rules(drule_by_relation, drule_dict):
    drule_list = drule_by_relation.split('||')

    feature = {}
    for rule in drule_list:
        if rule in drule_dict:
            feature[ 'dr(%d)' % drule_dict[rule] ] = 1 

    return feature
    

def write_dependency_rule_by_line(file_name):
    from nltk.parse.stanford import StanfordDependencyParser
    jar = config.STANFORD_PARSER_JAR_PATH
    models_jar = config.STANFORD_PARSER_MODEL_PATH
    dependency_parser = StanfordDependencyParser(path_to_jar = jar, path_to_models_jar = models_jar, java_options='-mx3000m')

    all_relations = util.read_data_utf8(file_name)

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

    '''
        each result is correspoding to a sentence
        a line_interval [from, to)
    '''

    start = time.time()
    parse_result = dependency_parser.raw_parse_sents(sentences)
    end = time.time()

    print('extracting dependency rule costs %f s' % (end - start))

    dep_rule_list = [] #list(list(str))
    dep_rule_for_one_relation = [] # list(str)
    line_interval_idx = 0
    count = 0
    for result in parse_result:
        for t in result:
            for node in range(len(t.nodes)):
                if t.nodes[node]['word'] == None or t.nodes[node]['deps'].items() == []:
                    continue
                else:
                    dep_rule_for_one_relation.append( '%s<-%s' % \
                        (t.nodes[node]['word'], ' '.join( [ key for key, val in t.nodes[node]['deps'].items() ] ))) 
        if count == line_interval[line_interval_idx][1] - 1:
            #print '%d: (%d, %d) finished' % (line_interval_idx, line_interval[line_interval_idx][0], line_interval[line_interval_idx][1])
            line_interval_idx += 1
            dep_rule_list.append(dep_rule_for_one_relation)
            dep_rule_for_one_relation = []
        
        count += 1

    write_data = []
    for dep_rules in dep_rule_list:
        write_data.append( '||'.join([rule for rule in dep_rules] ) )

    with codecs.open('tmp/dep_rule_%s.txt'% (file_name), 'w', encoding = 'utf-8') as file:
        file.write( u'\n'.join(write_data) )


"""
def get_Arg_production_rules(relation, Arg, parse_dict):
    #1.  dict[(DocID, sent_index)] = [token_list]
    dict = {}
    DocID = relation["DocID"]
    Arg_TokenList = get_Arg_TokenList(relation, Arg)
    for sent_index, word_index in Arg_TokenList:
        if (DocID, sent_index) not in dict:
            dict[(DocID, sent_index)] = [word_index]
        else:
            dict[(DocID, sent_index)].append(word_index)

    #2.
    Arg_subtrees = []
    for (DocID, sent_index) in dict.keys():
        parse_tree = parse_dict[DocID]["sentences"][sent_index]["parsetree"].strip()
        syntax_tree = Syntax_tree(parse_tree)
        if syntax_tree.tree != None:
            Arg_indices = dict[(DocID, sent_index) ]
            Arg_leaves = set([syntax_tree.get_leaf_node_by_token_index(index) for index in Arg_indices])

            no_need = []
            for node in syntax_tree.tree.traverse(strategy="levelorder"):
                if node not in no_need:
                    if set(node.get_leaves()) <= Arg_leaves:
                        Arg_subtrees.append(node)
                        no_need.extend(node.get_descendants())


    production_rule = []
    for tree in Arg_subtrees:
        for node in tree.traverse(strategy="levelorder"):
            if not node.is_leaf():
                rule = node.name + "-->" + " ".join([child.name for child in node.get_children()])
                production_rule.append(rule)

    return production_rule
"""

def get_production_rules(relation, parse_dict):
    Arg1_production_rules = get_Arg_production_rules(relation, "Arg1", parse_dict)
    Arg2_production_rules = get_Arg_production_rules(relation, "Arg2", parse_dict)
    rules = Arg1_production_rules + Arg2_production_rules

    production_rules = ["Arg1_%s" % rule for rule in rules] + \
                       ["Arg2_%s" % rule for rule in rules] + \
                       ["Both_%s" % rule for rule in rules]

    return production_rules

"""
Utilities for loading data
"""

import json
import networkx as nx
import numpy as np
import pickle
import re
import torch

from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils import degree
from tqdm import tqdm

from metrics import (
	graph_metrics
)


def create_graphs_from_sys_output(torquestra, generated_sys_output, resin, model_name, cache_path, maven):
	"""Converting raw generated output into convenient data structures
	Args:
		torquestra: 			List[Dict]	
		generated_sys_output: 	List[Dict]
		resin: 					List[Dict]
		model_name: 			str, name of sentence transformer model passed as arg, e.g. `all-mpnet-base-v2`
		cache_path: 			str
		maven: 					Dict
	Returns:
		list_objects
	"""

	## torch geometric Data instances to make graphs
	## SentenceTransformer to embed whole texts
	list_objects = []
	model = SentenceTransformer(model_name)
	data_list = torquestra + generated_sys_output + resin
	#data_list =  generated_sys_output + resin

	all_event_types_vector = get_event_type_vector(data_list)
	print(f'Length all event type vector: {len(all_event_types_vector)}')
	print(all_event_types_vector[:20])
	print()

	for data_instance in tqdm(data_list):

		graph_names = ['causal_graph', 'graph', 'resin_graph']

		torque_id = ''
		if 'torque_id' in data_instance:
			torque_id = data_instance['torque_id']

		for graph_name in graph_names:

			origin = ''
			# if graph_name=='causal_graph_detailed':
			# 	origin='schema-original'
			# elif 'origin' in data_instance:
			# 	origin = data_instance['origin']
			# else:
			# 	origin = graph_name.replace('_graph', '')

			if graph_name in data_instance:

				text, thisGraph, event_types, event_type_one_hot, emptyGraph = get_sentence_graph_events(data_instance, 
																graph_name, 
																maven, 
																all_event_types_vector)
				
				if emptyGraph:
					pass
				else:
					# data structures
					head_tail_rel = dict()
					all_elements = set()
					sys_output, graph_stats,  = '', ''

					if graph_name not in ['graph']:
						# for torquestra data and torquestra schemas
						for element in thisGraph:
							if 'rel' in element:
								if element['rel'] == 'ENABLES':
									rel_binary = 1
								else:
									rel_binary = -1
								head_tail_rel[(element['head'], element['tail'])] = rel_binary
								all_elements.add(element['head'])
								all_elements.add(element['tail'])
					
					# for maven sys_output graph
					else:
						nodes = thisGraph['nodes']
						links = thisGraph['links']
						for node in nodes:
							n = node['id'].replace('[EOS]', '')
							all_elements.add(n)
						for link in links:
							if link['rel']== 'enables':
								rel_binary = 1
							else:
								rel_binary = -1

							source = link['source'].replace('[EOS]', '')
							target = link['target'].replace('[EOS]', '')
							head_tail_rel[(source, target)] = rel_binary

						sys_output = data_instance['sys_output']

						graph_stats = graph_metrics(sys_output)
	
					all_elements = list(all_elements)

					## key elements of Data instance
					## 'x' is the row matrix of node embeddings
					## each node = natural language phrase
					edge_attr, edge_index, x = [], [], []

					for node in all_elements:
						phrase = node.replace('Entity::', '')
						phrase_embedding = model.encode(phrase)
						x.append(phrase_embedding)

					x = np.array(x)
					x = torch.tensor(x, dtype=torch.float)

					## embedding the whole text with SentenceTransformer
					## the `text` refers to a Maven Wikipedia article with events appended with event types for encoding
					text_no_events  = re.sub(r'(\S+?::)','', text)
					text_embedding  = model.encode(text_no_events)

					for ht, rel in head_tail_rel.items():
						edge_index.append([all_elements.index(ht[0]), all_elements.index(ht[1])])
						edge_attr.append([rel])

					## graph attributes
					edge_index = torch.tensor(edge_index, dtype=torch.long)
					edge_index = torch.transpose(edge_index, 0, 1)
					edge_attr = torch.tensor(edge_attr, dtype=torch.long)

					data = Data(x=x, 
								num_nodes=len(all_elements),
								edge_index=edge_index, 
								edge_attr=edge_attr, 
								text=text,
								text_no_events=text_no_events,
								event_types=event_types,
								event_types_vector=event_type_one_hot,
								torque_id=torque_id,
								origin=origin,
								graph_stats=graph_stats,
								sys_output=sys_output,
								text_embedding=text_embedding)

					page_rank = get_page_rank(data)

					page_ranks = [] 

					keys = sorted(page_rank.keys())
					for k in keys:
						page_ranks.append(page_rank[k])

					deg = degree(data.edge_index[0], data.num_nodes)
					
					for idx in range(10):
						# masking nodes weighted by pagerank for unsupervised graph embedding learning
						index_to_mask = np.random.choice(a=keys, p=page_ranks)
						mask = torch.zeros(data.num_nodes, dtype=torch.bool)
						mask[index_to_mask] = 1
						fname = f'train_mask_{idx}'
						data[fname] = mask
					
					## test_mask_pr is always node with highest page_rank
					test_mask_pr = torch.zeros(data.num_nodes, dtype=torch.bool)
					test_mask_pr[np.argmax(page_ranks)] = 1
					data.test_mask_pr = test_mask_pr

					## test_mask_rand
					## TODO experimentation: compare random test node with node based on highest pagerank
					test_mask_rand = torch.zeros(data.num_nodes, dtype=torch.bool)
					rand_node_idx = np.random.randint(data.num_nodes)
					test_mask_rand[rand_node_idx] = 1
					data.test_mask_rand = test_mask_rand
					
					data.chains = head_tail_rel
					data.degrees = deg
					data.elements = all_elements
					data.pageranks = page_ranks

					## only test texts (Maven) have assigned topic labels
					thisTopic = ''
					mention2event = dict()
					if graph_name=='graph':
						isMaven=True
						thisTopic = data_instance['maven_topic']
						#mention2event = data_instance['mention2event']
					# check here, 2023-06-23
					elif graph_name=='graph':
						isMaven=True
					else:
						isMaven=False

					## schema chapters from RESIN library
					if graph_name=='resin_graph':
						isResin=True
					else:
						isResin=False
					
					data.isMaven = isMaven
					data.topic = thisTopic
					data.mention2event = mention2event
					data.isResin = isResin

					list_objects.append(data)

	## for convenience (processing of 3K documents takes 20+ mins)
	print(f'Pickling file: {cache_path}')
	print()
	with open(cache_path, 'wb') as fout:
		pickle.dump(list_objects, fout, protocol=pickle.HIGHEST_PROTOCOL)
	
	return list_objects

	
def get_dataloaders(dataset, batch_size=16):
	"""Batching data in convenient containers
	Args:
		dataset: 	List of Data objects
		batch_size: Number of instances in each batch
	Returns:
		Dataloaders for train, validation, test, and resin schemas
	"""
	#torch.manual_seed(12345)
	#train_dataset = [instance for instance in dataset if instance.isMaven==False]
	train_dataset = dataset
	# valid_dataset will include isSchema==True
	valid_dataset = [instance for instance in dataset if instance.isMaven==False and instance.isResin==False]
	test_dataset = [instance for instance in dataset if instance.isMaven==True and instance.isResin==False]
	resin_dataset = [instance for instance in dataset if instance.isResin==True]
	#schema_dataset = [instance for instance in dataset if instance.isSchema==True]

	keys_excluded = ['chains', 
					'degrees', 
					'elements', 
					'pageranks', 
					'text', 
					'event_types', 
					'origin', 
					'torque_id', 
					'graph_stats', 
					'sys_output', 
					'topic', 
					'text_embedding', 
					'text_no_events',
					'mention2event']

	print(f'Number of training graphs: {len(train_dataset)}')
	print(f'Number of validation graphs: {len(valid_dataset)}')
	print(f'`Number of test graphs`: {len(test_dataset)}')
	print(f'Number of resin graphs: {len(resin_dataset)}')
	#print(f'Number of schema graphs: {len(schema_dataset)}')

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, exclude_keys=keys_excluded)
	valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, exclude_keys=keys_excluded)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, exclude_keys=keys_excluded)
	resin_loader = DataLoader(resin_dataset, batch_size=batch_size, shuffle=False, exclude_keys=keys_excluded)
	#schema_loader = DataLoader(schema_dataset, batch_size=batch_size, shuffle=False, exclude_keys=keys_excluded)

	return train_loader, test_loader, valid_loader, resin_loader


def get_torquestra(torquestra_path):
	"""Loads sample torquestra data.
	Args:
		torquestra_path: Path to data sample.
	Returns:
		List of torquestra data dictionaries.
	"""
	with open(torquestra_path, 'r') as fin:
		torquestra = json.load(fin)

	print("torquestra example")
	print(torquestra[0])
	print()

	return torquestra

def get_maven_sys_output(maven_sys_output_path, maven_topics):
	"""Loads sample maven data (3K Wikipedia articles).
	Args:
		maven_sys_output_path: 	Path to generated causal graphs associated with Wikipedia articles.
		maven_topics: 			List of topics (ground truth labels)
	Returns:
		List of maven data dictionaries.
	"""
	with open(maven_sys_output_path, 'r') as fin:
		sys_output_complete = json.load(fin)

	maven_sys_output = list()

	cnt_missing_graphs = 0

	for item in sys_output_complete:

		if 'graph' not in item:
			cnt_missing_graphs+=1
			continue

		else:

			maven_mention2eventType = dict()
			title, topic = '', ''

			text = item['text_with_events'].replace('TEXT: ', '')
			temp_text = ''.join([o if '::' not in o else o.split('::')[-1] for o in text.split()[:20]])
			graph = item['graph']
			pagerank = item['graph_metadata']['pagerank']
			#sys_output = item['sys_output']

			# this is not so elegant way of matching topics and titles to texts
			for obj in maven_topics:
				temp_doc = ''.join([o if '::' not in o else o.split('::')[-1] for o in obj['doc'].split()[:20]])
				
				if temp_text in temp_doc or temp_doc in temp_text:
					title = obj['title']
					topic = obj['topic']
					text_with_events = obj['doc']

					break

			events_list = list()

			for t in text_with_events.split():
				if '::' in t:
					et, mention = t.split('::')
					events_list.append(et)

					maven_mention2eventType[mention]=et

			common_event_types = list(set([i for i in events_list if events_list.count(i)>1]))

			maven_sys_output.append({'title': title, 
									'text': text, 
									'topic': topic, 
									'maven_graph': graph, 
									'pagerank': pagerank, 
									'events': common_event_types,
									#'sys_output': sys_output,
									'mention2event': maven_mention2eventType})

	print("maven example")
	print(maven_sys_output[0])
	print()

	print(f'Number of Maven exs missing graphs:\t{cnt_missing_graphs}')
	print()

	return maven_sys_output


def get_valid_torquestra_sys_output(sys_output_path):
	"""Loads system output using Torquestra validation set
	Args:
		sys_output_path: 	Path to generated causal graphs
	Returns:
		List of data dictionaries.
	"""
	with open(sys_output_path, 'r') as fin:
		sys_output_complete = json.load(fin)

	valid_sys_output = list()

	cnt_missing_graphs = 0

	for item in sys_output_complete:

		if 'graph' not in item['graph_metrics_pred']:
			cnt_missing_graphs+=1
			continue

		else:

			text = item['text'].replace('TEXT: ', '').strip()
			graph = item['graph_metrics_pred']['graph']
			pagerank = item['graph_metrics_pred']['pagerank']
			sys_output = item['sys_output']

			# also include G_SS metrics
		
			valid_sys_output.append({ 
									'text': text, 
									'torquestra_validation_graph': graph, 
									'pagerank': pagerank, 
									'sys_output': sys_output
									})

	print("Torquestra validation example")
	print(valid_sys_output[0])
	print()

	print(f'Number of Torquestra validation exs missing graphs:\t{cnt_missing_graphs}')
	print()

	return valid_sys_output


def get_maven(maven_path):
	"""Loads maven event hierarchy.
	Args:
		maven: Path to maven event hierachy.
	Returns:
		Set of event type tuples.
	"""
	maven = set()

	with open(maven_path, 'r') as file:
		lines = file.readlines()
		lines = [line.rstrip() for line in lines]

	for l in lines:
		events = tuple([item.lower() for item in l.split(',') if len(item)>0])

		while len(events)>0:
			maven.add(events)
			events = events[:-1]
	return maven


def get_resin(resin_path):
	"""Loads RESIN schema and converts to title + graph instances.
	Args:
		resin_path: Path to processed RESIN schema
	Returns:
		List of resin data instances.
	"""
	with open(resin_path, 'r') as fin:
		schemas = json.load(fin)

	resin_list = list()

	for k, v in schemas.items():

		sys_output = []
		for j in v['edges']:
			sys_output.append({'head': j[0], 'rel': 'ENABLES', 'tail': j[1]})
		resin_list.append({'title': k, 'resin_graph': sys_output})

	print("resin example")
	print(resin_list[0])
	print()

	return resin_list


def get_sentence_graph_events(item, graph_name, maven, all_event_types_vector):
	"""Loads graph instance and returns sentence and associated event type information
	Args:
		item: 					Torch Geometric Data object
		graph_name: 			Str source of graph
		maven: 					Event hierarchy to check for existing event types
		all_event_types_vector: Reference vector for one-hot vector of event types
	Returns:
		sentence: 				str
		system output: 			List[dict, dict,...] 
		event types observed: 	List[str, str,...]
		one-hot event encoding: List[int, int,...]
		emptyGraph: 			bool
	"""
	emptyGraph = False

	def sublists(l,i,j):
		# sublists returns all possible combinations of embedded lists
		return [l[m:n+1] for m in range(i,j+1) for n in range(m,j+1)]

	if graph_name == 'causal_graph_dfs':
		sent = item['example_torque']
		causal_graph = item['causal_graph_dfs']
		event_types = item['event_types']

	elif graph_name == 'causal_graph_detailed':
		# make corresponding schema graph
		sent = item['example_torque']
		causal_graph = make_schema_graphs(item['causal_graph_detailed'])
		event_types = item['event_types']

	elif graph_name == 'ester_causal_graph':
		sent = item['example_ester']
		causal_graph = item['ester_causal_graph']
		event_types = item['ester_event_types']

	elif graph_name == 'causal_graph':
		sent=item['text']
		causal_graph = item['causal_graph']
		event_types = 'crime'

	elif graph_name == 'maven_graph':
		sent=item['text']
		causal_graph = item['maven_graph']

		# needed bc generated sys_output includes events only with the MAVEN data
		event_types = item['events']   

		if len(causal_graph['nodes'])==0:
			emptyGraph=True

	elif graph_name == 'torquestra_validation_graph':
		sent=item['text']
		causal_graph = item['torquestra_validation_graph']
		event_types = 'torquestra-valid'

	else:
		sent=item['title']
		causal_graph = item['resin_graph']
		event_types = 'resin' 

	if len(causal_graph)==0:
		emptyGraph=True  	

	if type(event_types)==dict:
		# True for 698 original
		event_types = [' '.join(v.split(';')) for v in event_types.values() if v!='Entity']

	if type(event_types)==str:
		temp_event_types, done = [],[]
		list_events = [e.lower().strip() for e in event_types.split(';')]
		list_sublists = sublists(list_events,0,len(list_events))
		list_sublists.sort(key=len,reverse=True)

		for sublist in list_sublists:
			sublist = tuple(sublist)
			if sublist not in done and sublist in maven:
				temp_event_types.append(' '.join(sublist))
				elements_sublist = sublists(sublist,0,len(sublist))
				for es in elements_sublist:
					done.append(tuple(es))

			elif sublist not in done and len(sublist)==1:
				temp_event_types.append(' '.join(sublist))
				done.append(sublist)

		event_types = temp_event_types

	event_type_count_vector = np.zeros(len(all_event_types_vector))

	single_event_types = set()

	for event_type in event_types:

		event_types_split = event_type.split()
		for thisEventType in event_types_split:
			if thisEventType.lower() in all_event_types_vector:
				event_idx = all_event_types_vector.index(thisEventType.lower())
				event_type_count_vector[event_idx] += 1
				single_event_types.add(thisEventType.lower())


	return sent, causal_graph, list(single_event_types), event_type_count_vector, emptyGraph


def get_page_rank(data):
	"""Calculates page rank for each node in graph
	Args:
		data: 	Torch geometric Data object
	Returns:
		List of tuples [(node, pagerank)...]
	"""
	G = to_networkx(data)
	page_rank = nx.pagerank(G, alpha=0.85)

	return page_rank


def get_event_type_vector(data_list):
	"""Makes an array of all events observed in dataset
	Args:
		data_list: 		List of Data objects
	Returns:
		List[str,...]: 	List of all events observed in dataset
	"""
	all_event_types_temp, event_types, list_event_types = [],[],[]

	for datum in data_list:
		if 'causal_graph_dfs' in datum:
			event_types = datum['event_types']

		elif 'ester_causal_graph' in datum:
			event_types = datum['ester_event_types']

		elif 'causal_graph' in datum:
			event_types = 'crime'

		elif 'events' in datum:
			event_types = datum['events'] 

		if type(event_types)==dict:
			list_event_types = [' '.join(v.split(';')) for v in event_types.values() if v!='Entity']

		elif type(event_types)==str:
			list_event_types = [e.lower().strip() for e in event_types.split(';')]

		else:
			list_event_types = event_types

		for event_type in list_event_types:
			for et in event_type.split():
				all_event_types_temp.append(et.lower())

	all_event_types = [i for i in all_event_types_temp if all_event_types_temp.count(i)>1]

	all_event_types = list(set(all_event_types))

	return all_event_types


def make_schema_graphs(causal_graph):
	"""Converts instance graph into schema graph for original 698
	Args:
		causal_graph:	List[Dict[head,rel,tail]]
	Returns:
		List[Dict[head,rel,tail]]
	"""
	schema_graph = []
	for g in causal_graph:
		head = g['head_event_type'].replace(';', ' ')
		tail = g['tail_event_type'].replace(';', ' ')

		new_g = {'head': head, 'rel': g['rel_short'], 'tail': tail}
		schema_graph.append(new_g)

	return schema_graph

def make_and_save_maven_schema_graphs(list_objects):

	# not implemented, 02/24/2023

	maven_objs = [d for d in list_objects if d.isMaven==True]

	for obj in maven_objs:
		chains = obj.chains
		text = obj.text
		mention2event = obj.mention2event



		


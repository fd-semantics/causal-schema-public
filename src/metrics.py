# metrics.py
# 1/28/2023: add similarity metrics

import numpy as np
import nltk
import networkx as nx
import torch
from collections import Counter
from scipy.spatial import distance
from visualization import make_networkx_digraph_for_viz

def graph_metrics(sys_output: str) -> dict:
	""" Calculates statistics for each generated causal graph
	Args:
		sys_output: str of generated output
	Returns:
		stats_dict: dict
	"""
	G = nx.DiGraph()

	def clean_entity(entity: str) -> str:
		# in rare cases, the entity also includes relation (when overgenerated)
		entity = entity.replace('ENABLES', '').replace('BLOCKS', '')
		return entity.strip()

	def clean_str(text: str) -> str:
		text = text.replace('[GEN]', '').replace('[EOS]', '').replace('Yes: ', '')
		text = text.replace('This is correct:', '')
		text = text.replace('Entity:: ', 'Entity::')
		text = text.lstrip('0123456789.- ')
		return text.strip()

	## search output to merge nodes with nearly same names (measured by edit distance)
	node2aliases, alias2node = {}, {}

	for sd in sys_output.split('\n'):
		if len(sd)>8:
			sd = clean_str(sd)

			if 'ENABLES' not in sd and 'BLOCKS' not in sd:
				continue
			elif 'ENABLES' in sd:
				head, tail = clean_entity(sd.split('ENABLES')[0]), clean_entity(sd.split('ENABLES')[1])
			else:
				head, tail = clean_entity(sd.split('BLOCKS')[0]), clean_entity(sd.split('BLOCKS')[1])

			if len(node2aliases)<1:
				node2aliases[head]=set()

			## simple entity mention matching
			if head not in node2aliases:
				node2aliases[head] = set()
				for k, v in node2aliases.items():  
					if head != k:
						if nltk.edit_distance(head,k)<3:
							node2aliases[k].add(head)
							break

			if tail not in node2aliases:
				node2aliases[tail] = set()
				for k, v in node2aliases.items():
					if tail != k:
						if nltk.edit_distance(tail,k)<3:
							node2aliases[k].add(tail)
							break
							
	## reverse dictionary
	for k, nodes in node2aliases.items():
		for node in nodes:
			alias2node[node]=k
		if k not in alias2node:
			alias2node[k]=k
  
	for sd in sys_output.split('\n'):
		if len(sd)>8:

			sd = clean_str(sd)
			
			if 'ENABLES' not in sd and 'BLOCKS' not in sd:
				pass

			if 'ENABLES' in sd:
				head,tail = alias2node[clean_entity(sd.split('ENABLES')[0])], alias2node[clean_entity(sd.split('ENABLES')[1])]
				if nltk.edit_distance(head,tail)>2:
					G.add_edge(head, tail, rel='enables')

			elif 'BLOCKS' in sd:
				head, tail = alias2node[clean_entity(sd.split('BLOCKS')[0])], alias2node[clean_entity(sd.split('BLOCKS')[1])]
				if nltk.edit_distance(head,tail)>2:
					G.add_edge(head, tail, rel='blocks')

	num_enables,num_blocks = 0,0
	for node1, node2, data in G.edges.data():
		if data['rel'] == 'enables':
			num_enables += 1
		else:
			num_blocks += 1
			
	num_colliders = 0
	for element in list(G.in_degree()):
		if element[1]>1:
			num_colliders += 1
			
	num_splitters = 0
	for element in list(G.out_degree()):
		if element[1]>1:
			num_splitters += 1

	is_connected = False
		
	if not nx.is_empty(G):
		is_connected = nx.is_weakly_connected(G)

	graph_data = nx.node_link_data(G)

	stats_dict = {'in_degree': list(G.in_degree()), 
				  'out_degree': list(G.out_degree()), 
				  'pagerank': {k: round(v, 2) for k, v in sorted(nx.pagerank(G, alpha=0.85).items(), key=lambda item:item[1])},
				 'is_connected': is_connected,
				 'num_enables': num_enables,
				 'num_blocks': num_blocks, 
				 'num_colliders': num_colliders,
				 'num_splitters': num_splitters,
				 'graph': graph_data}
				   
	return stats_dict


def calculate_graph_similarity(data_list, embeddings, k=10, metric='cosine'):
	""" Basic function to determine most and least similar texts given embeddings; currently unused
	Args:
		data_list: 	list of Data objects
		embeddings: embeddings associated with Data objects
	Returns:
		prints out texts
	"""
	every_text = [t.text for t in data_list]
	every_chain = [t.chains for t in data_list]
	every_event_types = [t.event_types for t in data_list]

	embeddings = embeddings.detach()
	embeddings = np.asarray([t.numpy() for t in embeddings])

	print(f'Length data list: {len(data_list)}')
	print(f'Length embeddings: {len(embeddings)}')

	print(data_list[0].text)
	print()

	print(embeddings[0].shape)
	print(embeddings.shape)

	print(f'Metric: {metric}')
	print()
	distances = distance.cdist([embeddings[0]], embeddings, metric)[0]
	min_index = np.argmin(distances)
	print(f'Min index: {min_index}, similarity: {distances[min_index]}')

	## most similar texts
	bottom_k_indices = np.argpartition(distances, k)[:k]
	## least similar texts
	top_k_indices = np.argpartition(distances, -k)[-k:]

	print('Most similar texts')

	for idx in bottom_k_indices:

		thisText = every_text[idx].split('\n')[0]
		theseEvents = every_event_types[idx]
		print(f'Index: {idx}\tSimilarity: {round(distances[idx],3)}')
		print(f'{thisText}')
		print(f'Events: {theseEvents}')
		print(f'Chain: {every_chain[idx]}')
		print()

	# print('Least similar texts')
	# for idx in top_k_indices:
	# 	thisText = every_text[idx].split('\n')[0]
	# 	theseEvents = every_event_types[idx]
	# 	print(f'Index: {idx}\tSimilarity: {round(distances[idx],3)}')
	# 	print(f'{thisText}')
	# 	print(f'Events: {theseEvents}')
	# 	print(f'Chain: {every_chain[idx]}')
	# 	print()


def get_cos_sim_event_vectors(vectors):
	""" Calculates average pairwise cosine similarity for all vectors
	Args:
		vectors: 	List of all vectors
	Returns:
		Mean distance between vectors
	"""
	running_avg = list()
	for vec in vectors:
		running_avg.append(np.mean(distance.cdist([vec], vectors, metric='cosine')))

	return np.mean(running_avg)


def get_graphs_for_viz_and_cluster_metrics(k_indices, every_event_types, every_chains, every_text, graphs_for_viz):
	"""Compiles graphs for visualizations and metrics
	Args:
		k_indices: 			Indices of texts from clusters
		every_event_types: 	Event types associated with texts
		every_chains: 		Causal chains associated with texts
		every_text: 		Texts
	Returns:
		graphs_for_viz: 	List of graphs for visualization
		metrics:			Metrics about graphs in each cluster
	"""
	clustering_coeffs, transitivity, square_clusterings, number_edges, number_nodes = [], [], [], [], []

	degrees, densities = [], []

	for idx in k_indices:
		theseEvents = every_event_types[idx]
		theseChains = every_chains[idx]
		thisDiGraph  = make_networkx_digraph_for_viz(theseChains)
		thisText = every_text[idx]

		coeffs = nx.average_clustering(thisDiGraph)
		clustering_coeffs.append(coeffs)

		transitivity.append(nx.transitivity(thisDiGraph))

		square_cl = np.mean([v for v in nx.square_clustering(thisDiGraph).values()])
		square_clusterings.append(square_cl)

		number_nodes.append(thisDiGraph.number_of_nodes())
		number_edges.append(thisDiGraph.number_of_edges())

		degrees.append(np.mean([j for (i, j) in thisDiGraph.degree()]))
		densities.append(nx.density(thisDiGraph))

		if thisDiGraph not in graphs_for_viz['similar_graphs']:
			graphs_for_viz['similar_graphs'].append(thisDiGraph)
			graphs_for_viz['event_types'].append(theseEvents)
			graphs_for_viz['texts'].append(thisText)

	metrics = {
		'mean_number_nodes': np.mean(number_nodes),
		'mean_number_edges': np.mean(number_edges),
		'mean_degree': np.mean(degrees),
		'mean_clustering_coefficient': np.mean(clustering_coeffs),
		'mean_transitivity': np.mean(transitivity),
		'mean_square_clustering': np.mean(square_clusterings),
		'mean_density': np.mean(densities)
	}

	print(f"Mean number nodes:\t\t {metrics['mean_number_nodes']}")
	print(f"Mean number edges:\t\t {metrics['mean_number_edges']}")
	print(f"Mean degree:\t\t {metrics['mean_degree']:.2f}")
	print(f"Mean clustering coefficient:\t {metrics['mean_clustering_coefficient']:.2f}")
	print(f"Mean transitivity:\t\t {metrics['mean_transitivity']:.2f}")
	print(f"Mean square clustering:\t\t {metrics['mean_square_clustering']:.2f}")
	print(f"Mean density:\t\t {metrics['mean_density']:.2f}")
	print()

	return graphs_for_viz, metrics


def summarize_cluster_metrics(metrics_for_each_cluster, method='gat'):
	"""Prints out cluster metrics
	Args:
		metrics_for_each_cluster: 	Metrics compiled for each cluster
		method: 					Method to make graph or text embeddings
	Returns:
		print out
	"""
	print(f'Metrics across clusters generating graph embeddings using: {method}')
	print()
	num_nodes = np.mean([i['mean_number_nodes'] for i in metrics_for_each_cluster])
	num_edges = np.mean([i['mean_number_edges'] for i in metrics_for_each_cluster])
	degrees = np.mean([i['mean_degree'] for i in metrics_for_each_cluster]) 
	coeffs = np.mean([i['mean_clustering_coefficient'] for i in metrics_for_each_cluster]) 
	transitivities = np.mean([i['mean_transitivity'] for i in metrics_for_each_cluster]) 
	sq_clusters = np.mean([i['mean_square_clustering'] for i in metrics_for_each_cluster]) 
	densities = np.mean([i['mean_density'] for i in metrics_for_each_cluster]) 

	print(f'Cluster mean number nodes:\t\t {num_nodes:.2f}')
	print(f'Cluster mean number edges:\t\t {num_edges:.2f}')
	print(f'Cluster mean degree:\t\t {degrees:.2f}')
	print(f'Cluster mean clustering coefficient:\t {coeffs:.2f}')
	print(f'Cluster mean transitivity:\t\t {transitivities:.2f}')
	print(f'Cluster mean square clustering:\t\t {sq_clusters:.2f}')
	print(f'Cluster mean density:\t\t {densities:.2f}')
	print()


def display_graph_stats(list_graphs):
	""" Prints out attributes of Data object
	Args:
		list_graphs: List[Data,Data,...]
	Returns:
		print out of Data attributes and topics observed
	"""
	topic_counter = Counter()

	for data in list_graphs:

		print('*****'*8)
		print(f'torque_id: {data.torque_id}')
		print(f'origin: {data.origin}')
		print(f'elements: {data.elements}')
		print(f'head tail rel: {data.chains}')
		print(f'Degrees: {data.degrees}')
		print(f'Edge index: {data.edge_index}')
		print(f'Node with max degree: {data.elements[torch.argmax(data.degrees)]}')
		print(f'Node with max pagerank: {data.elements[np.argmax(data.pageranks)]}')
		print(f'Event types: {data.event_types}')
		print(f'Events types vector: {data.event_types_vector[:20]}')
		print(f'Number of event types: {np.sum(data.event_types_vector)}')
		print()
		print(f'isMaven: {data.isMaven}')
		print(f'isResin: {data.isResin}')
		#print(f'isSchema: {data.isSchema}')
		print(f'mention2event: {data.mention2event}')
		print(f'topic: {data.topic}')
		print(f'number nodes: {data.num_nodes}')
		print(f'number edges: {data.num_edges}')
		print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
		print(f'Has isolated nodes: {data.has_isolated_nodes()}')
		print(f'is directed: {data.is_directed()}')
		print(f'has self loops: {data.has_self_loops()}')
		print()
		
		topic_counter[data.topic] +=1

	print('**************'*8)
	print('Topics count')
	for k, v in topic_counter.items():
		print(k, v)
	print('**************'*8)

			
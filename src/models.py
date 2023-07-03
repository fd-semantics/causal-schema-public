## models.py

import numpy as np
import torch

from collections import Counter
from karateclub import FeatherGraph
from scipy.spatial import distance
from sklearn.cluster import KMeans
from torch_geometric.utils.convert import to_networkx

from torch.nn import ELU, Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, global_mean_pool

from visualization import make_networkx_digraph_for_viz
from metrics import (
	get_cos_sim_event_vectors,
	get_graphs_for_viz_and_cluster_metrics
)
from visualization import (
	init_graphs_for_viz,
	make_networkx_digraph_for_viz
)

class GAT(torch.nn.Module):
	""" Graph Attention Network (GAT)
	Documentation: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GATv2Conv
	See papers `Graph Attention Networks`, `How Attentive are Graph Attention Networks?`
	Args:
		hidden_channels: 	presently using 256
		output_channels: 	dimension of graph embedding, presently 768
		num_node_features: 	dimension of input node embedding, presently 768
	Returns:
		x:					graph embedding
		embeddings:			node embeddings after message passing
	"""
	def __init__(self, hidden_channels, output_channels, num_node_features, heads=4):
		super(GAT, self).__init__()
		#torch.manual_seed(12345)
		self.conv1 = GATv2Conv(num_node_features, hidden_channels, heads=heads)
		self.conv2 = GATv2Conv(hidden_channels*heads, hidden_channels, heads=1)
		# self.conv3 = GATv2Conv(hidden_channels, hidden_channels)
		self.lin = Linear(hidden_channels, output_channels)

	def forward(self, x, edge_index, batch, dropout=0.5):
		# 1. Obtain node embeddings
		m = ELU()
		x = F.dropout(x, p=dropout, training=self.training) 
		x = self.conv1(x, edge_index)
		x = m(x)
		x = F.dropout(x, p=dropout, training=self.training)
		x = self.conv2(x, edge_index)
		# x = m(x)
		# x = F.dropout(x, p=dropout, training=self.training)
		# x = self.conv3(x, edge_index)
		embeddings = x
		# 2. Readout layer
		x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

		# 3. Apply a final linear layer
		x = F.dropout(x, p=dropout, training=self.training)
		x = self.lin(x)

		return x, embeddings

class GCN(torch.nn.Module):
	""" Graph Convolutional Network (GCN)
	Documentation: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv
	See paper `Semi-supervised Classification with Graph Convolutional Networks`
	Args:
		hidden_channels: 	presently set at 256
		output_channels: 	dimension of graph embedding, presently 768
		num_node_features: 	dimension of input node embedding, presently 768
	Returns:
		x:					graph embedding
	"""
	def __init__(self, hidden_channels, num_node_features):
		super(GCN, self).__init__()
		torch.manual_seed(12345)
		self.conv1 = GCNConv(num_node_features, hidden_channels)
		self.conv2 = GCNConv(hidden_channels, hidden_channels)
		self.conv3 = GCNConv(hidden_channels, hidden_channels)
		self.lin = Linear(hidden_channels, num_node_features)

	def forward(self, x, edge_index, batch, dropout=0.5):
		# 1. Obtain node embeddings
		x = F.dropout(x, p=dropout, training=self.training) 
		x = self.conv1(x, edge_index)
		x = x.relu()
		x = F.dropout(x, p=dropout, training=self.training)
		x = self.conv2(x, edge_index)
		x = x.relu()
		x = self.conv3(x, edge_index)

		# 2. Readout layer
		x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

		# 3. Apply a final linear transformation
		x = F.dropout(x, p=dropout, training=self.training)
		x = self.lin(x)

		return x


def get_feather_graph_embeddings(list_graphs):
	""" Feather graph embeddings
	Documentation: https://karateclub.readthedocs.io/en/latest/_modules/karateclub/graph_embedding/feathergraph.html
	Karate Club: unsupervised machine learning extension library for NetworkX
	Args:
		list_graphs:		List of graphs to embed
	Returns:
		feather_embeddings:	List of embeddings	
	"""
	networkx_graphs = []
	for graph in list_graphs:
		G = to_networkx(graph)
		networkx_graphs.append(G)

	model = FeatherGraph()
	model.fit(networkx_graphs)
	feather_embeddings = model.get_embedding()	
	print(f'Feather graph embedding shape: {feather_embeddings.shape}')
	return feather_embeddings


def k_means(data_list, embeddings, k=10, num_clusters=6, metric='seuclidean', method='gat'):
	""" K means clustering algorithm
	Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
	Args:
		data_list:		List of data objects
		embeddings: 	Stacked embeddings to cluster
		k:				Number of objects in each cluster
		num_clusters	Number of clusters
		metric:			Distance measure for similarity ('seuclidean', 'cosine', 'euclidean', 'manhattan', etc)
		method:			Name of graph embedding method
	Returns:
		cluster_event_counts:		Count of events in each cluster			
		metrics_for_each_cluster:	Graph metrics for each cluster	
		graphs_for_each_cluster:	Graphs associated with each cluster
		texts_for_each_cluster:		Texts associated with each cluster
	"""
	cluster_event_counts, graphs_for_each_cluster, metrics_for_each_cluster, texts_for_each_cluster = [], [], [], []

	if torch.is_tensor(embeddings):
		X = embeddings.detach().numpy()
	else:
		X = embeddings

	# elements of data_list are Torch Geometric Data objects
	every_text = [t.text_no_events for t in data_list]
	every_event_types = [t.event_types for t in data_list]
	every_event_types_vector = [t.event_types_vector for t in data_list]
	every_topics = [t.topic for t in data_list]
	every_chains = [t.chains for t in data_list]
	every_pageranks = [t.pageranks for t in data_list]
	these_elements = [t.elements for t in data_list]
	
	kmeans = KMeans(n_clusters=num_clusters, random_state=0)
	kmeans.fit(X)

	print(kmeans.cluster_centers_)
	print()

	for clusterIndex, center in enumerate(kmeans.cluster_centers_):

		events_counter = Counter()

		print('************'*8)
		print(f'Center: {clusterIndex}\tMethod: {method}')

		distances = distance.cdist([center], X, metric)[0]
		min_index = np.argmin(distances)
		print(f'Min index: {min_index}, similarity: {distances[min_index]}')

		k_indices = np.argpartition(distances, k)[:k]

		print(f'Distances: {distances[:10]}')
		print(f'Indexed distances: {distances[k_indices]}')
		print()
		print('Most similar texts')

		all_event_type_vectors = []

		for idx in k_indices:

			thisText = every_text[idx].split('\n')[0]
			theseEvents = every_event_types[idx]

			thisEventTypeVector = every_event_types_vector[idx]
			all_event_type_vectors.append(thisEventTypeVector)

			theseChains = every_chains[idx]
			thesePageRanks = every_pageranks[idx]
			thisTopic = every_topics[idx]

			node_with_max_pagerank = these_elements[idx][np.argmax(thesePageRanks)]

			theseEntities = [e for e in these_elements[idx] if 'Entity::' in e]
			theseEntities = list(set(theseEntities))

			## Additional perspective on data: tracking key interactions
			## Presently, written to file at end of main (texts_for_each_cluster)
			key_interactions = []
			for ent in theseEntities:
				for chain, thisRel in theseChains.items():
					if ent in chain and node_with_max_pagerank in chain:
						key_interactions.append(f'{chain[0]}-({thisRel})->{chain[1]}')

			entities = ';'.join(theseEntities)

			interactions = ';'.join(key_interactions)
 
			texts_for_each_cluster.append([clusterIndex, method, distances[idx], node_with_max_pagerank, entities, interactions, thisText])

			## Another perspective on data: schema = bag of frequent event types
			theseEvents = list(set(theseEvents))
			for event in theseEvents:
				events_counter[event.lower()]+=1
			
			print(f'Method: {method}')
			print(f'Index: {idx}\Distance: {round(distances[idx],3)}')
			print(f'{thisText}')
			print()
			print(f'Events: {theseEvents}')
			print()
			print(f'Graph: {theseChains}')
			print()
			## Another perspective: schema = set of blocking relations
			print('BLOCK chains')
			for head_tail, value in theseChains.items():
				if value == -1:
					print(head_tail)
			print()
			print(f'Topic: {thisTopic}')
			print()
			print('*************')

		cluster_event_counts.append(events_counter)

		prototypical_chains = every_chains[min_index]
		prototypical_digraph = make_networkx_digraph_for_viz(prototypical_chains)
		prototypical_text = every_text[min_index]

		## Cluster centroid = prototypical graph = ur-graph
		print(f'Method {method}\tur-graph {clusterIndex}:', prototypical_digraph, prototypical_digraph.nodes())
		print()
		avg_cosine_similarity_event_vectors = get_cos_sim_event_vectors(all_event_type_vectors)
		print(f'Avg event vector similarity: {avg_cosine_similarity_event_vectors:.2f}')

		graphs_for_viz = init_graphs_for_viz(prototypical_digraph, prototypical_text, every_event_types[min_index])

		## Here we select events in the mid-range of frequency; most frequent are top of hierarchy (e.g., Action)
		most_common_events = [item[0] for item in events_counter.most_common(16) if item not in events_counter.most_common(6)]

		print(f'most common events:\t {most_common_events}')
		print()

		print('Count of how many cluster elements contain at least 1 of 10 most common events')

		count_of_elements_with_common_event = []

		## event purity = overlap of observed event types with most frequently observed event types cluster-wide
		for idx in k_indices:
			element_cnt = 0
			theseEvents = every_event_types[idx]
			theseEvents = [j.lower() for j in list(set(theseEvents))]

			for these_e in theseEvents:
				if these_e in most_common_events:
					element_cnt+=1

			count_of_elements_with_common_event.append(element_cnt)

		print(f'Number of elements in cluster:\t {k}')
		print(f'Mean common events in each element:\t {np.mean(count_of_elements_with_common_event):.2f}')
		print(f'Stdev common events in each element:\t {np.std(count_of_elements_with_common_event):.2f}')

		pred_labels = list()
		for idx in k_indices:
			pred_labels.append(every_topics[idx])
		print('***********'*8)
		print("Count of topics in this cluster")
		print(Counter(pred_labels))
		print()

		full_graphs_for_viz, cluster_metrics = get_graphs_for_viz_and_cluster_metrics(k_indices, 
																				every_event_types, 
																				every_chains, 
																				every_text, 
																				graphs_for_viz)

		graphs_for_each_cluster.append(full_graphs_for_viz)
		metrics_for_each_cluster.append(cluster_metrics)

	return cluster_event_counts, metrics_for_each_cluster, graphs_for_each_cluster, texts_for_each_cluster
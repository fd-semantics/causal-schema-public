# visualization.py

import datetime
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def make_networkx_digraph_for_viz(chains_dict):
	""" Makes list of causal chains into networkx digraph
	Args:
		chains_dict: Dict with key: (node0, node1) and value: rel
	Returns:
		networkX DiGraph
	"""
	G = nx.DiGraph()
	for k, v in chains_dict.items():
		G.add_node(k[0])
		G.add_node(k[1])
		G.add_edge(k[0], k[1], rel=v)
	return G


def init_graphs_for_viz(prototypical_digraph, prototypical_text, every_event_type):
	""" Basic object to be passed around for visualization of clusters
	Args:
		prototypical_digraph: 	Causal graph associated with centroid
		prototypical_text: 		Associated text (e.g., from Wikipedia)
		every_event_type: 		Event types observed with this text
	Returns:
		Dict populated with prototypical graph (centroid), initialized for similar graphs
	"""
	graph_for_viz = {
					'method': '',
					'prototypical_graph': prototypical_digraph, 
					'prototypical_text': prototypical_text,
					'prototypical_events': every_event_type, 
					'similar_graphs':[], 
					'event_types':[],
					'texts':[]
					}
	return graph_for_viz


def viz_graph(G, title, color, main_path, write_viz=False):
	""" Basic plotting function for connected graphs; function visualize_graphs outputs subplots also
	Args:
		G: 		Causal graph
		title: 	Graph name for header in viz
		color: 	Node color
	Returns:
		Saves graph image if write_viz = True
	"""
	def nudge(pos, x_shift, y_shift):
		return {n:(x + x_shift, y + y_shift) for n,(x,y) in pos.items()}

	pos = nx.spring_layout(G, k=1.0)
	#pos = nx.spectral_layout(G, scale=2)
	#pos = nx.shell_layout(G, scale=1)
	pos_nodes = nudge(pos, 0, 0.1) 
	
	plt.rcParams['figure.figsize'] = [32,24]
	plt.figure()

	edge_labels = nx.get_edge_attributes(G,'rel')

	nx.draw(
		G, pos=pos, edge_color='black', width=0.5, linewidths=2.0, with_labels=False,
		node_size=600, node_color=color, alpha=1.0, arrowsize=32,
		labels={node: node for node in G.nodes()}
	)
	nx.draw_networkx_labels(G, pos=pos_nodes, font_size=32) 
	nx.draw_networkx_edge_labels(
		G, pos,
		font_color='red',
		font_size=32,
		edge_labels=edge_labels
	)
	plt.axis('off')
	plt.suptitle(title, fontsize=42, y=1.1)
	
	if write_viz:
		image_path = main_path / 'causal-{}.png'.format(title)
		plt.savefig(image_path, format='png')
	## uncomment to view plot
	# else:
	# 	plt.show()


def visualize_graphs(graphs_for_each_cluster, write_data_path, graph_type='gat', task='cluster'):
	""" Creates object for evaluation and visualization of clusters
	Args:
		graphs_for_each_cluster: List of graph objects; see func init_graphs_for_viz
		write_data_path: 		Path where to write
		graph_type: 			For save file name
		task: 					For save file name
	Returns:
		List[dict], each dict a cluster dictionary; to be added to javascript evaluation script
	"""
	output_for_html = []

	now = datetime.datetime.now()

	HOME_PATH = write_data_path

	def nudge(pos, x_shift, y_shift):
		return {n:(x + x_shift, y + y_shift) for n,(x,y) in pos.items()}

	for cluster_num in range(len(graphs_for_each_cluster)):

		proto_G = graphs_for_each_cluster[cluster_num]['prototypical_graph']
		proto_text = graphs_for_each_cluster[cluster_num]['prototypical_text']
		proto_events = graphs_for_each_cluster[cluster_num]['prototypical_events']
		similarGraphs = graphs_for_each_cluster[cluster_num]['similar_graphs'][:5]
		theseEvents = graphs_for_each_cluster[cluster_num]['event_types'][:5]
		similarTexts = graphs_for_each_cluster[cluster_num]['texts'][:5]

		final_similar_graphs, final_these_events, final_similar_texts = [],[],[]

		for g, e, t in zip(similarGraphs, theseEvents, similarTexts):
			if t.strip() != proto_text:
				final_similar_graphs.append(g)
				final_these_events.append(e)
				final_similar_texts.append(t)

		if graph_type not in ['gat', 'feather']:

			cluster_dict = {'cluster_num': cluster_num, 
				'graph_type': graph_type,
				'prototypical_text': proto_text,
				'prototypical_events': proto_events,
				'similar_events': final_these_events,
				'similar_texts': final_similar_texts,
				'prototypical_graph_path': '',
				'similar_graph_path': ''}

		else:
			mid_idx = len(proto_events)//2
			proto_events = f"{' '.join(proto_events[:mid_idx])}\n{' '.join(proto_events[mid_idx:])}"

			plt.figure(figsize=(14,8))
			ax = plt.gca()

			if 'match' in task:
				ax.set_title(f'Schema, {proto_events}')
			else:
				ax.set_title(f'Proto-graph, cluster:{cluster_num}, {proto_events}')
			
			pos = nx.spring_layout(proto_G, k=0.9)
			pos_nodes = nudge(pos, 0, 0.1)
			edge_labels = nx.get_edge_attributes(proto_G,'rel')

			nx.draw(
				proto_G, pos=pos, edge_color='black', width=0.5, linewidths=2.0, with_labels=False,
				node_size=30, node_color='gray', alpha=1.0, arrowsize=16,
				labels={node: node for node in proto_G.nodes()}, 
				ax=ax
			)
			nx.draw_networkx_labels(proto_G, pos=pos_nodes, font_size=16, ax=ax) 
			nx.draw_networkx_edge_labels(
				proto_G, pos,
				font_color='red',
				font_size=10,
				edge_labels=edge_labels,
				ax=ax
			)
			_ = ax.axis('off')

			proto_fname = f'gnn-proto-{task}-{graph_type}-{cluster_num}-{now.month}-{now.day}-{now.hour}-{now.minute}.png'
			fig_name_proto = HOME_PATH  / proto_fname
			print(f'Saving fig of proto graph to: {proto_fname}')
			plt.savefig(fig_name_proto, format='png')

			fig, axs = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
			ax = axs.flatten()

			for ix, sim_G in enumerate(final_similar_graphs):
				## list includes five elements, one of which is sometimes the prototypical graph 
				if ix < 4:

					pos = nx.spring_layout(sim_G, k=1.0)
					pos_nodes = nudge(pos, 0, 0.1)
					edge_labels = nx.get_edge_attributes(sim_G,'rel')

					nx.draw(
						sim_G, pos=pos, edge_color='black', width=0.5, linewidths=2.0, with_labels=False,
						node_size=25, node_color='blue', alpha=1.0, arrowsize=8,
						labels={node: node for node in sim_G.nodes()},
						ax=ax[ix]
					)
					nx.draw_networkx_labels(sim_G, pos=pos_nodes, font_size=8, ax=ax[ix]) 
					nx.draw_networkx_edge_labels(
						sim_G, pos,
						font_color='red',
						font_size=6,
						edge_labels=edge_labels,
						ax=ax[ix]
					)
					## titles of all plots are a list of hierarchical events observed in the text
					## cutting off long lists of events so that title fits into small plot
					final_these_events[ix] = final_these_events[ix][:18]
					mid_idx = len(final_these_events[ix])//2
					titleEvents = f"Similar {ix}\n{' '.join(final_these_events[ix][:mid_idx])}\n{' '.join(final_these_events[ix][mid_idx:])}"

					ax[ix].set_title(titleEvents, fontsize=6)
					ax[ix].set_axis_off()

			plt.axis('off')

			print(now)

			similar_fname = f'gnn-similar-{task}-{graph_type}-{cluster_num}-{now.month}-{now.day}-{now.hour}-{now.minute}.png'

			fig_name_similar = HOME_PATH / similar_fname
			print(f'Saving fig of similar graphs to: {fig_name_similar}')

			plt.savefig(fig_name_similar, format='png')

			cluster_dict = {'cluster_num': cluster_num, 
							'graph_type': graph_type,
							'prototypical_text': proto_text,
							'prototypical_events': proto_events,
							'similar_events': final_these_events,
							'similar_texts': final_similar_texts,
							'prototypical_graph_path': fig_name_proto.name,
							'similar_graph_path': fig_name_similar.name}

		output_for_html.append(cluster_dict)

	return output_for_html

def visualize_count_dictionary(cluster_counts, write_data_path, save_fig=False):
	""" Visualization of counts of event types across clusters; interesting, but not sure how to interpret
	Args:
		cluster_counts: 	List of events observed in each cluster
		write_data_path: 	Path where to write
		save_fig: 			To save or not to save
	Returns:
		Prints dataframe of frequent events (>2 observed) to terminal and saves heatmap to disk
	"""
	print("Visualizing count dictionary")
	observed_events = set()

	for i, cluster in enumerate(cluster_counts):
		for key in cluster.keys():
			observed_events.add(key)
	observed_events = sorted(list(observed_events))

	clusters_for_df = []

	for cluster in cluster_counts:
		complete_count_for_this_cluster = []
		for event in observed_events:
			if event not in cluster:
				complete_count_for_this_cluster.append(0)
			else:
				complete_count_for_this_cluster.append(cluster[event])
		clusters_for_df.append(complete_count_for_this_cluster)

	df = pd.DataFrame(clusters_for_df, columns=observed_events)
	df1 = df[df.columns[df.sum()>2]]
	print(df1)

	now = datetime.datetime.now()
	print(now)

	if save_fig:
		plt.figure(figsize = (60,10))
		plt.xticks(rotation=90) 
		plt.title('Composition of clusters by count of event types', fontsize = 24) # title
		plt.xlabel('Event types', fontsize = 20) # x-axis label
		plt.ylabel('Clusters', fontsize = 20) # y-axis label
		sns.heatmap(df1, annot=True, cmap="Blues")
		fig_name = write_data_path / f'gnn-cluster-event-types-{now.month}-{now.day}-{now.hour}-{now.minute}.png'
		print(f'Saving fig to: {str(fig_name)}')
		plt.savefig(fig_name, format='png', bbox_inches = 'tight')

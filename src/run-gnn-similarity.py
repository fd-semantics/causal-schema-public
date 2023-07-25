# gnn-similarity-v4-maven.py

## 1/2023 update: torquestra to maven matching

import argparse
import csv
import datetime
import json
import numpy as np
import os
import pickle
import random
import torch

from pathlib import Path

from metrics import (
	summarize_cluster_metrics,
	display_graph_stats
)
from models import (
	GAT,
	GCN,
	get_feather_graph_embeddings,
	k_means
)
from utils import (
	create_graphs_from_sys_output,
	get_torquestra, 
	get_maven_sys_output, 
	get_valid_torquestra_sys_output,
	get_maven, 
	get_resin, 
	get_dataloaders,
	make_and_save_maven_schema_graphs
)
from matching import (
	match_maven_to_torq_and_resin_gat, 
	match_resin_to_maven_gat,
	tfidf_schema_matching
)
from visualization import (
	visualize_graphs, 
	visualize_count_dictionary
)



"""

This is the main script for schema matching and clustering using TFIDF, event similarity, and graph neural networks. 
See the README for details on implementation and the paper for a higher-level overview.


"""


def dir_path(path):
	if os.path.isdir(path):
		return path
	else:
		raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")


def run_model(train_loader, 
			test_loader, 
			valid_loader, 
			resin_loader, 
			dropout, 
			hidden_channels, 
			lr, 
			epochs=4,
			num_node_features=768,
			output_channels=768,
			heads=4):

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	#model = GCN(hidden_channels=hidden_channels, num_node_features=num_node_features).to(device)
	
	model = GAT(hidden_channels=hidden_channels, 
				output_channels=output_channels, 
				num_node_features=num_node_features,
				heads=heads
				).to(device)
	
	print(model)

	optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

	def cosine_loss_func(feat1, feat2):
		# minimize average cosine similarity
		cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
		#return F.cosine_similarity(feat1, feat2).mean()
		return cos(feat1, feat2).mean()

	def euclidean_loss_func(feat1, feat2):
		loss = torch.cdist(feat1, feat2, p=2)
		return loss.mean()

	def train(loader):

		model.train()
		train_loss = []

		for data in loader:
			## sampling five of 10 masked graphs for training step
			random_masks_idx = random.sample(range(10), 5)
			for idx in random_masks_idx:
				thisMask = data[f'train_mask_{idx}']
				data = data.to(device)
				target_graph = data.x[thisMask]
				out, gat_embeddings = model(data.x, data.edge_index, data.batch, dropout) 
				loss = abs(cosine_loss_func(out, target_graph) - 1)
				loss.backward()
				optimizer.step() 
				optimizer.zero_grad()
				train_loss.append(loss)
		
		train_loss = [tl.item() for tl in train_loss]
		return train_loss

	def test(loader):
		## validation for this self-supervised learning problem
		## model trains well enough for qualitatively assessed coherent clusters
		model.eval()
		test_loss = []
		for data in loader:  # Iterate in batches over the test set.
			data = data.to(device)
			out, gat_embeddings_test = model(data.x, data.edge_index, data.batch, dropout=dropout) 
			data_with_mask = torch.reshape(data.x[data.test_mask_pr], (out.shape[0], num_node_features))
			loss = abs(cosine_loss_func(out, data_with_mask) - 1)
			test_loss.append(loss) 

		test_loss = [tl.item() for tl in test_loss]
		#return np.mean(test_loss)
		return test_loss


	def get_embeddings_from_final_model(loader):

		print('getting graph embeddings from final model')
		print()
		model.eval()
		stacked_graph_embeddings = torch.empty((0, num_node_features))

		for data in loader:
			data = data.to(device)
			out, _ = model(data.x, data.edge_index, data.batch)
			stacked_graph_embeddings = torch.cat((stacked_graph_embeddings, out.cpu()))

		return stacked_graph_embeddings


	for epoch in range(1, epochs):
		train_loss = train(train_loader)
		## preliminary experiments with unsupervised learning using graph neural networks. e.g., Graph Attention Network (GAT)
		## transforming embedding space so that each node in graph is good predictor of whole graph embedding
		## displaying range of loss: this reduces by half (from 1.0 to 0.5 over 2 epochs of training)
		## this is a lightweight model and trains in a matter of seconds on GPU (or few mins on CPU)
		print(f'train_loss: {train_loss[0]}--{train_loss[-1]}')
		
		if epoch%2==0:
			test_loss = test(test_loader)
			#print(f'test_loss: {test_loss}')
			avg_train_loss = np.mean(train_loss)
			avg_test_loss = np.mean(test_loss)
			print(f'Epoch: {epoch:03d}, Train loss: {avg_train_loss:.4f}, Test loss: {avg_test_loss:.4f}')

	testdata_stacked_graph_embeddings = get_embeddings_from_final_model(test_loader)
	torquestra_stacked_graph_embeddings = get_embeddings_from_final_model(valid_loader)
	resin_stacked_graph_embeddings = get_embeddings_from_final_model(resin_loader)
	# schema_stacked_graph_embeddings = get_embeddings_from_final_model(schema_loader)

	print(f'Number of embeddings in test_loader: {len(testdata_stacked_graph_embeddings)}')
	print(f'Shape each embeddings batch: {testdata_stacked_graph_embeddings[0].shape}')

	print(f'Number of embeddings in valid_loader: {len(torquestra_stacked_graph_embeddings)}')
	print(f'Shape each embeddings batch: {torquestra_stacked_graph_embeddings[0].shape}')

	print(f'Number of embeddings in resin_loader: {len(resin_stacked_graph_embeddings)}')
	print(f'Shape each embeddings batch: {resin_stacked_graph_embeddings[0].shape}')

	# print(f'Number of embeddings in schmea_loader: {len(schema_stacked_graph_embeddings)}')
	# print(f'Shape each embeddings batch: {schema_stacked_graph_embeddings[0].shape}')

	return testdata_stacked_graph_embeddings, torquestra_stacked_graph_embeddings, resin_stacked_graph_embeddings


def main():

	parser = argparse.ArgumentParser(description='gnn similarity metrics.')

	parser.add_argument('--model_name', type=str)
	parser.add_argument('--cache_path')
	parser.add_argument('--torquestra_path', type=str)
	parser.add_argument('--generated_output_path', type=str)
	parser.add_argument('--resin_path', type=str)
	parser.add_argument('--maven_path', type=str)
	parser.add_argument('--topics_path', type=str)
	parser.add_argument('--hidden_channels', type=int)
	parser.add_argument('--batch_size', type=int)
	parser.add_argument('--epochs', type=int)
	parser.add_argument('--transformer_heads', type=int, default=4)
	parser.add_argument('--dropout', type=float)
	parser.add_argument('--learning_rate', type=float)
	parser.add_argument('--num_clusters', type=int)
	parser.add_argument('--k_docs_to_return', type=int)
	parser.add_argument('--metric', type=str)
	parser.add_argument('--data_stats', type=str, required=False)
	parser.add_argument('--save_fig', type=str, required=False)
	parser.add_argument('--save_schema_graphs', type=str, required=False)
	parser.add_argument('--write_data_path', type=str)
	parser.add_argument('--filename', type=str)
	args = parser.parse_args()
	print('******'*8)
	now = datetime.datetime.now()
	print(now)
	print('******'*8)
	print(args)
	print('******'*8)
	print()
	print()

	write_data_path = Path(args.write_data_path)

	if not write_data_path.is_dir():
		print('Making directory for writing: reports/')
		os.makedirs('reports/')
		print()

	cache_path = args.cache_path

	if not os.path.isfile(cache_path):
		torquestra = get_torquestra(args.torquestra_path)

		with open(args.topics_path, 'r') as fin:
			maven_topics = json.load(fin)

		maven_hierarchy = get_maven(args.maven_path)

		if 'maven' in args.generated_output_path:
			print('MAVEN system output!!')
			print()
			generated_sys_output = get_maven_sys_output(args.generated_output_path, maven_topics)

		else:
			print('Torquestra validation system output!!')
			print()
			generated_sys_output = get_valid_torquestra_sys_output(args.generated_output_path)
		
		resin = get_resin(args.resin_path)

		list_graphs = create_graphs_from_sys_output(torquestra, generated_sys_output, resin, args.model_name, cache_path, maven_hierarchy)
	
	else:
		with open(cache_path, 'rb') as fin:
			list_graphs = pickle.load(fin)

	if args.data_stats.lower() not in ['false', 'no', 'n']:
		print('Displaying graph stats')
		print()
		display_graph_stats(list_graphs)
		print()
	
	maven_graphs_for_clustering = [d for d in list_graphs if d.isMaven==True]
	print(f'Length test set: {len(maven_graphs_for_clustering)}')
	print()
	torquestra_graphs_for_clustering = [d for d in list_graphs if d.isMaven==False]

	# train_loader = all examples; test_loader=only maven; valid_loader=only torquestra
	train_loader, test_loader, valid_loader, resin_loader = get_dataloaders(list_graphs, batch_size=args.batch_size)

	testdata_stacked_graph_embeddings, torquestra_stacked_graph_embeddings, resin_stacked_graph_embeddings = run_model(train_loader, 
																					test_loader, 
																					valid_loader,
																					resin_loader,
																					dropout=args.dropout,
																					hidden_channels=args.hidden_channels, 
																					lr=args.learning_rate, 
																					epochs=args.epochs,
																					heads=args.transformer_heads)
	print('******'*8)
	print('GAT clustering Torquestra')
	## in-domain data presently not analyzed
	torqcluster_counts, torqcluster_metrics, torq_graphs_for_each_cluster, torq_texts_for_each_cluster = k_means(torquestra_graphs_for_clustering, 
															torquestra_stacked_graph_embeddings, 
															k=args.k_docs_to_return, 
															num_clusters=args.num_clusters, 
															metric=args.metric,
															method='gat')
	print('******'*8)
	print('GAT clustering generated system output (MAVEN or Torquestra validation)')
	## out-of-domain data for analysis
	gat_cluster_counts, gat_cluster_metrics, gat_graphs_for_each_cluster, gat_texts_for_each_cluster = k_means(maven_graphs_for_clustering, 
															testdata_stacked_graph_embeddings, 
															k=args.k_docs_to_return, 
															num_clusters=args.num_clusters, 
															metric=args.metric,
															method='gat')
	print('******'*8)
	print('Feather clustering generated system output (MAVEN or Torquestra validation)')
	print()

	## limit number of graphs for testing
	#torquestra_graphs = [d for d in list_graphs if d.isMaven==False and len(d.text.split())>150][:20]
	#torquestra_feather_embeddings = get_feather_graph_embeddings(torquestra_graphs)

	maven_feather_embeddings = get_feather_graph_embeddings(maven_graphs_for_clustering)
	
	feather_cluster_counts, feather_cluster_metrics, feather_graphs_for_each_cluster, feather_texts_for_each_cluster = k_means(maven_graphs_for_clustering, 
																		maven_feather_embeddings, 
																		k=args.k_docs_to_return, 
																		num_clusters=args.num_clusters, 
																		metric=args.metric,
																		method='feather')
	save_fig = False
	if args.save_fig.lower() in ['true', 'yes', 'y']:
		save_fig = True

	if 'noEvents' in args.generated_output_path or 'no-events' in args.generated_output_path:
		pass
	else:
		visualize_count_dictionary(gat_cluster_counts, write_data_path, save_fig=save_fig)
		visualize_count_dictionary(feather_cluster_counts, write_data_path, save_fig=save_fig)

	print('******'*8)
	print('GAT schema matching generated system output (MAVEN or Torquestra validation)')
	print()
	gat_schema_matched_graphs_for_viz = match_maven_to_torq_and_resin_gat(list_graphs, 
																			testdata_stacked_graph_embeddings, 
																			torquestra_stacked_graph_embeddings)

	#calculate_graph_similarity(maven_graphs_for_clustering, resin_stacked_graph_embeddings, metric=args.metric)
	print('******'*8)
	print('GAT schema matching RESIN to generated system output (MAVEN or Torquestra validation)')
	print()
	match_resin_to_maven_gat(list_graphs, testdata_stacked_graph_embeddings, resin_stacked_graph_embeddings)

	print('******'*8)
	print('TFIDF schema matching')
	print()
	tfidf_maven, tfidf_matched_graphs_for_viz = tfidf_schema_matching(list_graphs)

	print('***********'*8)
	print('TFIDF clustering generated system output (MAVEN or Torquestra validation)')
	print()
	tfidf_cluster_counts, tfidf_cluster_metrics, tfidf_graphs_for_each_cluster, tfidf_texts_for_each_cluster = k_means(maven_graphs_for_clustering, 
					tfidf_maven, 
					k=args.k_docs_to_return, 
					num_clusters=args.num_clusters, 
					metric=args.metric,
					method='tfidf')
		

	print('***********'*8)
	print('Text embedding clustering generated system output (MAVEN or Torquestra validation)')
	print()
	maven_text_embeddings = [d.text_embedding for d in list_graphs if d.isMaven==True]

	text_emb_cluster_counts, text_emb_cluster_metrics, text_emb_graphs_for_each_cluster, text_embs_texts_for_each_cluster = k_means(maven_graphs_for_clustering, 
				maven_text_embeddings, 
				k=args.k_docs_to_return, 
				num_clusters=args.num_clusters, 
				metric=args.metric,
				method='text-embedding')

	summarize_cluster_metrics(gat_cluster_metrics, method='gat')
	summarize_cluster_metrics(feather_cluster_metrics, method='feather')
	summarize_cluster_metrics(tfidf_cluster_metrics, method='tfidf')
	summarize_cluster_metrics(text_emb_cluster_metrics, method='text-embedding')


	if args.save_schema_graphs.lower() in ['true', 'yes', 'y']:
		# not yet implemented, 2023-02-16
		make_and_save_maven_schema_graphs(list_graphs)

	if save_fig:

		## Clustering output for visualization
		gat_output = visualize_graphs(gat_graphs_for_each_cluster, write_data_path, graph_type='gat')
		feather_output = visualize_graphs(feather_graphs_for_each_cluster, write_data_path, graph_type='feather')
		tfidf_output = visualize_graphs(tfidf_graphs_for_each_cluster, write_data_path, graph_type='tfidf')
		emb_output = visualize_graphs(text_emb_graphs_for_each_cluster, write_data_path, graph_type='text-embedding')

		output_for_html = gat_output + feather_output + tfidf_output + emb_output

		output_dictionary_path = write_data_path / f'gnn_clusters_compiled-{args.filename}-{now.month}-{now.day}-{now.hour}-{now.minute}.json'
		print(f'Saving dictionary of html paths to:  {output_dictionary_path}')
		with open(output_dictionary_path, 'w') as fout:
			json.dump(output_for_html, fout)

		## Schema matching output for visualization

		gat_matched_output = visualize_graphs(gat_schema_matched_graphs_for_viz, 
											write_data_path, 
											graph_type='gat',
											task='matching')

		tfidf_matched_output = visualize_graphs(tfidf_matched_graphs_for_viz, 
											write_data_path, 
											graph_type='tfidf',
											task='matching')

		matched_output_for_html = gat_matched_output + tfidf_matched_output

		output_matching_path = write_data_path / f'gnn_matching_compiled-{args.filename}-{now.month}-{now.day}-{now.hour}-{now.minute}.json'
		print(f'Saving dictionary of matched html paths to:  {output_matching_path}')
		with open(output_matching_path, 'w') as fileout:
			json.dump(matched_output_for_html, fileout)


		## Writing final cluster results
		## Each element in the lists is a tuple (cluster, method, text)
		final_cluster_text_list = gat_texts_for_each_cluster + feather_texts_for_each_cluster
		text_path = write_data_path / f'gnn_cluster_texts_{now.month}-{now.day}-{now.hour}-{now.minute}.csv'
		print(f'Saving cluster texts: {text_path}')

		with open(text_path, 'w') as csvfile:
			columns = ['cluster', 'method', 'distance', 'node_with_max_page_rank', 'entities', 'interactions', 'text']
			writer = csv.writer(csvfile)
			writer.writerow(columns)
			for item in final_cluster_text_list:
				writer.writerow(item)


if __name__ == "__main__":
	main()
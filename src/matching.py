# schema-matching.py
import random

from sentence_transformers import util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from nltk.corpus import stopwords

from visualization import (
	init_graphs_for_viz,
	make_networkx_digraph_for_viz
)


def match_maven_to_torq_and_resin_gat(list_graphs, maven_graph_embeddings, torquestra_graph_embeddings, k=10, metric='cosine', method='gat'):

	schema_matched_graphs_for_viz = []

	print('********'*8)
	print('Schema matching: 25 sampled MAVEN to Torquestra')
	print(f'Metric: \t{metric}')
	print(f'Method: \t{method}')
	print()

	## torquestra_graphs consists of all short TORQUESTRA graphs + RESIN graphs + Schema graphs (isMaven==False)
	## first try: limit schema library to shorter texts < 200 chars
	#torquestra_graphs = [d for d in list_graphs if d.isMaven==False and len(d.text.split())<250]
	torquestra_graphs = [d for d in list_graphs if d.isMaven==False]
	maven_graphs = [d for d in list_graphs if d.isMaven==True]

	every_torq_text = [t.text for t in torquestra_graphs]
	every_torq_chain = [t.chains for t in torquestra_graphs]
	every_torq_event_type = [t.event_types for t in torquestra_graphs]
	#every_torq_topic = [t.topic for t in torquestra_graphs]

	print(f'Torquestra embeddings shape for {method}: {torquestra_graph_embeddings.shape}')
	print(f'Maven embeddings shape for {method}: {maven_graph_embeddings.shape}')
	print()

	cnt = 0

	for mav_embedding, mav_graph in zip(maven_graph_embeddings, maven_graphs):
		r = random.random()
		if cnt < 25 and r<0.02:
			cnt+=1
			print('************'*8)
			print(f'Schema: {mav_graph.text}')

			prototypical_digraph = make_networkx_digraph_for_viz(mav_graph.chains)
			graphs_for_viz = init_graphs_for_viz(prototypical_digraph, mav_graph.text, mav_graph.event_types)
			graphs_for_viz['method']=method
			
			cosine_scores = util.cos_sim(mav_embedding, torquestra_graph_embeddings)

			## Calculate highest cosine similarity scores
			values,indices = cosine_scores.topk(5)
			## Some texts appear twice in corpus but we only want one as a candidate
			seen_texts = []

			for val, ind in zip(values[0], indices[0]):
				thisIx = ind.item()
				thisText = every_torq_text[thisIx]
				if thisText not in seen_texts:
					seen_texts.append(thisText)
					thisDiGraph = make_networkx_digraph_for_viz(every_torq_chain[thisIx])

					
					graphs_for_viz['similar_graphs'].append(thisDiGraph)
					graphs_for_viz['texts'].append(thisText)
					
					## Titles in each visualization are associated with event_types or RESIN schema name
					if len(every_torq_event_type[thisIx])>0:
						graphs_for_viz['event_types'].append(every_torq_event_type[thisIx])
					else:
						graphs_for_viz['event_types'].append(thisText)


				print("Score: {:.3f}\t\t{}".format(val.item(), every_torq_text[thisIx]))
				print()

			schema_matched_graphs_for_viz.append(graphs_for_viz)

	return schema_matched_graphs_for_viz


def match_resin_to_maven_gat(list_graphs, maven_graph_embeddings, resin_graph_embeddings, k=10, metric='cosine'):

	resin_graphs = [d for d in list_graphs if d.isResin==True]
	maven_graphs = [d for d in list_graphs if d.isMaven==True]

	every_text = [t.text for t in maven_graphs]
	every_topic = [t.topic for t in maven_graphs]

	print(f'RESIN embeddings shape for gat: {resin_graph_embeddings.shape}')
	print(f'Maven embeddings shape for gat: {maven_graph_embeddings.shape}')
	print()

	experiment_titles = ["General Attack, Criminal investigation.",
						"civil unrest, Response. Reaction.",
						" election, Voting.",
						"Investment news story.",
						"terrorist attacks, Crime.",
						"kidnapping, Committing crime. Unlawful act forbidden and punishable by criminal law. Crime.",
						"international conflict, War. Organised and prolonged violent conflict between states.",
						"mass shooting, Attack.",
						"Manmade disaster. A disaster caused by humans.",
						"natural disaster and rescue, Natural disaster. A disaster caused by humans.",
						"Preparations for the nuclear attack.",
						"violent crime, Crime. Central elements of criminal event."]

	for resin_embedding, resin_graph in zip(resin_graph_embeddings, resin_graphs):

		if resin_graph.text in experiment_titles:

			print('************'*8)
			print(f'Resin schema: {resin_graph.text}')

			cosine_scores = util.cos_sim(resin_embedding, maven_graph_embeddings)

			#Find highest cosine similarity scores
			values,indices = cosine_scores.topk(5)

			for val, ind in zip(values[0], indices[0]):
				print("Score: {:.4f} \t\t Topic: {} \t\t{}".format(val.item(), every_topic[ind.item()], every_text[ind.item()]))
				print()



def tfidf_schema_matching(list_graphs):

	tfidf_matched_graphs_for_viz = []

	eng_stopwords = set(stopwords.words('english'))
    
	vectorizer = TfidfVectorizer(stop_words=eng_stopwords)

	#torquestra_texts = [d.text for d in list_graphs if d.isMaven==False and d.isResin==False]
	torquestra_texts = [d.text for d in list_graphs if d.isMaven==False]
	torquestra_events = [d.event_types for d in list_graphs if d.isMaven==False]
	maven_texts = [d.text_no_events for d in list_graphs if d.isMaven==True]
	maven_topics = [d.topic for d in list_graphs if d.isMaven==True]

	X = vectorizer.fit_transform(torquestra_texts+maven_texts)

	X_torquestra = X[:len(torquestra_texts)]
	X_maven = X[len(torquestra_texts):]

	cnt_done = 0

	for mav_text, X_mav_tfidf, mav_topic in zip(maven_texts, X_maven, maven_topics):
	#for torq_text, torq_tfidf in zip(torquestra_texts, X_torquestra):
		# cosine_similarities = linear_kernel(torq_tfidf, X_maven).flatten()
		# related_docs_indices = cosine_similarities.argsort()[:-6:-1]

		if cnt_done < 25:

			graphs_for_viz = init_graphs_for_viz('', mav_text, mav_topic)
			graphs_for_viz['method']='tfidf'
			done_texts = []

			cnt_done+=1
			cosine_similarities = linear_kernel(X_mav_tfidf, X_torquestra).flatten()
			related_docs_indices = cosine_similarities.argsort()[:-7:-1]
			print('**************')
			print(mav_text)
			print()
			print(f'Topic: {mav_topic}')
			print()
			for ix in related_docs_indices:
				if len(graphs_for_viz['texts'])<5:
					if torquestra_texts[ix][:50] not in mav_text and torquestra_texts[ix][:50] not in done_texts:
						done_texts.append(torquestra_texts[ix][:50])
						print(torquestra_texts[ix])
						graphs_for_viz['similar_graphs'].append('')
						graphs_for_viz['texts'].append(torquestra_texts[ix])
						graphs_for_viz['event_types'].append(torquestra_events[ix])
						print()
			print()
			tfidf_matched_graphs_for_viz.append(graphs_for_viz)

	return X_maven.toarray(), tfidf_matched_graphs_for_viz
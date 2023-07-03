# causal-schema-public

This is a demo of causal schema induction, with data and results for schema clustering and schema matching experiments available.

[See website for more information!](https://fd-semantics.github.io/ "Causal schema public website")

### Update 2/23/2023: sample TORQUESTRA data available

### Update 6/2023: code and examples

Paper: https://arxiv.org/abs/2303.15381


## Resources

The following resources are available:
* Sample data from the TORQUESTRA dataset (see paper)
* Generated causal graphs for 3K Wikipedia articles using GPT2-XL distill-high (West et al, 2022)
* Sample processed schema from RESIN-11
* Code for clustering experiments using graph and text similarity methods
* Code for schema matching experiments
* Code for causal graph inference with trained model

## Environment 

#### Processing will be faster using a GPU but can also be done on CPU. Modify pytorch installation to meet your system's requirements.

Create anaconda environment
```bash
conda create --name torch-gnn python=3.10 && conda activate torch-gnn
conda install -y python=3.10 tqdm numpy scipy jupyter pandas matplotlib
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

# This sometimes resolves problems with numpy ndarray size
pip uninstall numpy
pip install numpy==1.22.4

pip install -r requirements.txt
```

To try out a sample cluster with the 3000 MAVEN Wikipedia articles, use the following. The first time this will take ~15+ minutes as embeddings for all the graph nodes need to be computed, which are then cached (update the cache path as necessary). To view data stats or to save sample figures for the clusters, change those flags to True.

```bash
python run-gnn-similarity.py \
--model_name all-mpnet-base-v2 \
--cache_path data/gnn_cache/maven-similarity-16-epochs-high-700-3k-maven.pickle \
--torquestra_path data/torquestra-human-2023-02-23.json \
--generated_output_path data/torquestra-auto-gpt2xl-high-maven-adversarial.json \
--resin_path data/resin11-schema-library.json \
--maven_path data/maven-hierarchy-complete.csv \
--topics_path data/maven-for-gen-with-topics.json \
--hidden_channels 256 \
--batch_size 32 \
--epochs 3 \
--dropout 0.5 \
--num_clusters 6 \
--learning_rate 0.01 \
--k_docs_to_return 25 \
--metric cosine \
--data_stats False \
--write_data_path reports/ \
--save_fig False \
--save_schema_graphs False \
--filename 16-epochs-250bs-high-maven2torq
```

## Checkpoints

We fine-tuned a knowledge distilled version of GPT2-XL on the TORQUESTRA dataset, 16 epochs, monitoring graph structural metrics. You can use these to generate causal graphs for your own data, but are not needed for the clustering and matching experiments, as we provide sample generated causal graphs in ```data/```).


## Data
### Sample data instance from TORQUESTRA


### Sample generated causal graph (Wikipedia)

{'title': '2006 Pangandaran earthquake and tsunami', 
'text': "The 2006 Pangandaran earthquake and tsunami occurred on July 17 at along a subduction zone off the coast of west and central Java , a large and densely populated island in the Indonesian archipelago . The shock had a moment magnitude of 7.7 ... [+300-600 tokens]", 
'topic': 'natural-disaster', 
'maven_graph': {'directed': True, 
                'multigraph': False, 
                'links': [{'rel': 'enables', 'source': 'Entity::Japanese Meteorological Center', 'target': 'tsunami watch'}, 
                    {'rel': 'enables', 'source': 'tsunami', 'target': 'deaths'}, {'rel': 'enables', 'source': 'tsunami', 'target': 'damage to homes'}, {'rel': 'enables', 'source': 'tsunami', 'target': 'high runups of tsunami'}, {'rel': 'enables', 'source': 'deaths', 'target': 'awareness of tsunami'}, {'rel': 'enables', 'source': 'Entity::people at the coast', 'target': 'people notice tsunami'}, {'rel': 'enables', 'source': 'people notice tsunami', 'target': 'awareness of tsunami'}, {'rel': 'enables', 'source': 'Entity::American tsunami warning center', 'target': 'tsunami watch posted'}, {'rel': 'enables', 'source': 'awareness of tsunami', 'target': 'people flee homes'}, {'rel': 'enables', 'source': 'damage to homes', 'target': 'people flee homes'}, {'rel': 'enables', 'source': 'Entity::Tungkang', 'target': 'Entity::Electricity of Vietnam'}, {'rel': 'enables', 'source': 'Entity::Electricity Company of Vietnam', 'target': 'Entity::Pangandaran'}, {'rel': 'enables', 'source': 'Entity::Panganda', 'target': 'Entity::Indonesia'}, {'rel': 'enables', 'source': 'Indian Ocean earthquake', 'target': 'tsunami'}, {'rel': 'enables', 'source': 'high runups of tsunamismic wave', 'target': 'damage to Indonesia'}]}, 
                'pagerank': {'Entity::Japanese Meteorological Center': 0.03, 'Entity::people at the coast': 0.03, 'Entity::American tsunami   
                      warning center': 0.03, 'Entity::Tungkang': 0.03, 'Entity::Electricity Company of Vietnam': 0.03, 'Entity::Panganda': 0.03, 'Indian Ocean earthquake': 0.03, 'high runups of tsunamismic wave': 0.03, 'deaths': 0.04, 'damage to homes': 0.04, 'high runups of tsunami': 0.04, 'tsunami watch': 0.05, 'tsunami': 0.05, 'people notice tsunami': 0.05, 'tsunami watch posted': 0.05, 'Entity::Electricity of Vietnam': 0.05, 'Entity::Pangandaran': 0.05, 'Entity::Indonesia': 0.05, 'damage to Indonesia': 0.05, 'awareness of tsunami': 0.11, 'people flee homes': 0.15}, 
                'events': ['Causation', 'Know', 'Destroying', 'Death', 'Damaging', 'Catastrophe', 'Sending', 'Entity', 'Presence']'}



### Citation

```
@misc{regan-etal-2023-causal-schema,
    title = "Causal schema induction for knowledge discovery",
    author = "Michael Regan and Jena Hwang and Keisuke Sakaguchi and James Pustejovsky",
    month = mar,
    year = "2023",
    url = "https://arxiv.org/abs/2303.15381",
}
```
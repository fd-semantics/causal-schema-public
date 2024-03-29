## TORQUESTRA data

This folder contains machine-generated (`torquestra-auto.json`) and human-made (`torquestra-human.json`) causal graphs with associated data (texts, topics, event types, temporal questions, event structure questions, etc.). We also include data resources from other work: hierarchical event types (`maven-hierarchy-complete.csv`) and the RESIN schema library (`resin11-schema-library.json`), formatted specifically for the demo code. See paper for citations.

### torquestra-auto.json 

A sample of 3k causal noisy graphs generated using GPT2-XL_distill (West et al, 2022). These data instances are the same as the examples for the matching experiments available in this repo, which we use as a proxy measure for the overall quality of generation.

The data for `maven_topic` and the event types in `text_with_events` are from the MAVEN dataset (Wang et al, 2020).

Data instance has form:

```
{
    'wikipedia_title': str,
    'maven_topic': str,
    'text_with_events': str,
    'graph': Dict,
    'graph_metadata: Dict
}
```


### torquestra-human.json 

This is the manually created dataset. The Data are assigned to two splits: `train` (698 exs.) and `dev` (180 exs.)

Splitting the data this way corresponds to a single variant of how we can evaluate causal generative models: one of *growing* a causal graph from a seed graph instantiation. That is, each train `causal_graph` is a 'mini' version of a corresponding `dev` graph. Graphs corresponding to one another share the same `torque_id`.

Note the following similarities and differences of the two splits:

* Each train `text` is shorter than a corresponding dev `text` that contains it (i.e. dev texts are longer contexts)

* Train examples include *temporal* `questions` and `answers` from TORQUE. In contrast, dev examples have  *event structure* `questions` and `answers` from ESTER (see paper for details).

* To make schemas, use `event_types` instead of natural language nodes in `causal_graph`!

* In train, `event_types` are assigned at node level, while in the dev split `event_types` are at the graph level (as a single concatenated string, semi-colon separated).

* We include `noncausal_event_types` with the train examples; these represent text mentions judged to not contribute to the overall causal story

* `Causal_graph` represents the latent causal structure of the event sequence described in the `text`. We make the graphs populating the nodes with text mentions *or* simple event descriptions in natural language

* Train causal graphs include both short and full causal relations (`rel` and `full_rel`) (see paper for definitions of all causal relations)

* Dev causal graphs include annotations for `saliency` (most important paths in the causal graph)


Summarizing the above, an instance in each split takes these forms:

train

```
{
    'split': 'train',
    'source': 'torque',
    'torque_id': str,
    'text': str,
    'questions': List[str],
    'answers': List[List[str]],
    'event_types: Dict(@node str: @event_type str),
    'noncausal_event_types': Dict(@mention-text str: @event_type str)
    'causal_graph': List[Dict('head': str, 'tail': str, 'rel': str, 'rel_full': str)],
}
```

dev

```
{
    'split': 'dev',
    'source': 'ester',
    'torque_id': str,
    'text': str,
    'questions': List[str],
    'answers': List[str],
    'schema_graph_event_types: str,
    'causal_graph': List[Dict('head': str, 'tail': str, 'rel': str, 'saliency': bool)],
}
```

Here are two full examples for comparison side to side of the `train` and `dev` splits. Note that both examples share the same `torque_id` (the dev text contains the train text).

```json
[
    {
        "split": "train",
        "source": "torque",
        "torque_id": "docid_PRI19980115.2000.0186_sentid_6",
        "text": "And on that basis Keating was released from prison before he was eligible for parole. Now the ninth US circuit court of appeals has ruled that the original appeal was flawed since it brought up issues that had not been raised before.",
        "questions": [
            "What event has already finished?",
            "What event has begun but has not finished?",
            "What will happen in the future?",
            "What happened after Keating was released?",
            "What did not happen before Keating was released?",
            "What happened before the court ruled?",
            "What did not happen after Keating was released?",
            "What happened after the court ruled?"
        ],
        "answers": [
            [
                "released",
                "ruled",
                "brought",
                "flawed"
            ],
            [],
            [],
            [
                "ruled",
                "flawed",
                "brought",
                "raised"
            ],
            [
                "was",
                "parole"
            ],
            [
                "flawed",
                "brought",
                "raised",
                "released"
            ],
            [],
            []
        ],
        "event_types": {
            "basis": "Sentiment;Suspicion",
            "Keating was released": "Action;Legality;Legal_rulings;Releasing",
            "not eligible for parole": "Action;Legality;Legal_rulings",
            "issues that had not been raised before": "Scenario",
            "ruled that the original appeal was flawed": "Action;Legality;Legal_rulings"
        },
        "noncausal_event_types": {
            "brought": "Action;Communication;Reporting"
        },
        "causal_graph": [
            {
                "head": "ruled that the original appeal was flawed",
                "tail": "Keating was released",
                "rel": "ENABLES",
                "rel_full": "ENABLES::BEGINS"
            },
            {
                "head": "not eligible for parole",
                "tail": "Keating was released",
                "rel": "BLOCKS",
                "rel_full": "BLOCKS::WITHOUT EFFECT"
            },
            {
                "head": "issues that had not been raised before",
                "tail": "ruled that the original appeal was flawed",
                "rel": "ENABLES",
                "rel_full": "ENABLES::BEGINS"
            }
        ]
    },
    {
        "split": "dev",
        "source": "ester",
        "torque_id": "docid_PRI19980115.2000.0186_sentid_6",
        "text": "\nFormer savings and loan chief, Charles Keating, is facing more legal troubles in California. A federal appeals court has reinstated his state convictions for securities fraud. NPR's Elaine Corey has more from San Francisco.\nIn nineteen ninety-one Charles Keating was convicted in state court of helping to defraud thousands of investors who bought high risk junk bonds sold by Keating's employees at Lincoln savings and loan. The bonds became worthless when the bankrupt thrift was seized by government regulators. Keating's convictions were thrown out in nineteen ninety-six on a technicality. And on that basis Keating was released from prison before he was eligible for parole. Now the ninth US circuit court of appeals has ruled that the original appeal was flawed since it brought up issues that had not been raised before. That means the convictions stand, a ruling likely to send Keating's lawyers back to state court where they must start over with a new appeal. Elaine Corey, NPR news, San Francisco.\n\n",
        "questions": [
            "Why was Mr. Keating convicted?",
            "Why was Mr. Keating released from prison?",
            "What might happen as a result of the convictions being ruled to stand after the flawed appeal?"
        ],
        "answers": [
            "defraud thousands of investors",
            "convictions were thrown out in nineteen ninety-six on a technicality",
            "send Keating's lawyers back to state court"
        ],
        "event_types": "banking;crime;action;legality;legal_rulings",
        "causal_graph": [
            {
                "head": "Entity::Charles Keating",
                "rel": "ENABLES",
                "tail": "Keating faces legal troubles",
                "saliency": 0
            },
            {
                "head": "Entity::Charles Keating",
                "rel": "ENABLES",
                "tail": "security fraud",
                "saliency": 0
            },
            {
                "head": "security fraud",
                "rel": "ENABLES",
                "tail": "Keating is convicted of security fraud",
                "saliency": 1
            },
            {
                "head": "Keating's convictions dismissed",
                "rel": "BLOCKS",
                "tail": "Keating is convicted of security fraud",
                "saliency": 1
            },
            {
                "head": "Keating's convictions dismissed",
                "rel": "ENABLES",
                "tail": "Keating was released from prison",
                "saliency": 1
            },
            {
                "head": "technicality",
                "rel": "ENABLES",
                "tail": "Keating's convictions dismissed",
                "saliency": 0
            },
            {
                "head": "bonds became worthless",
                "rel": "ENABLES",
                "tail": "security fraud",
                "saliency": 0
            },
            {
                "head": "bankrupt thrift was seized",
                "rel": "ENABLES",
                "tail": "bonds became worthless",
                "saliency": 0
            },
            {
                "head": "Entity::government regulators",
                "rel": "ENABLES",
                "tail": "bankrupt thrift was seized",
                "saliency": 0
            },
            {
                "head": "Entity::Charles Keating",
                "rel": "ENABLES",
                "tail": "Entity::Lincoln savings and loan",
                "saliency": 0
            },
            {
                "head": "Entity::Lincoln savings and loan",
                "rel": "ENABLES",
                "tail": "investors bought high risk junk bonds sold by Keating",
                "saliency": 0
            },
            {
                "head": "Entity::thousands of investors",
                "rel": "ENABLES",
                "tail": "investors bought high risk junk bonds sold by Keating",
                "saliency": 0
            },
            {
                "head": "Entity::ninth US circuit court of appeals",
                "rel": "ENABLES",
                "tail": "court reinstated Keating's state convictions for securities fraud",
                "saliency": 1
            },
            {
                "head": "court reinstated Keating's state convictions for securities fraud",
                "rel": "ENABLES",
                "tail": "Keating faces legal troubles",
                "saliency": 1
            },
            {
                "head": "Keating's convictions dismissed",
                "rel": "ENABLES",
                "tail": "court reinstated Keating's state convictions for securities fraud",
                "saliency": 1
            }
        ]
    }
]
```
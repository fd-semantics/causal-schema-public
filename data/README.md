Data in two splits: 'train' (698 exs.) and 'dev' (180 exs.)

This split corresponds to one way we imagine we can evaluate causal generative models. The way the data is here presented, train graphs are 'mini' versions of corresponding dev graphs. Graphs corresponding to one another share the samee `torque_id`. Viewing the data in this way makes it easy to view the task as one of 'growing' a causal graph from a seed graph instantiation. 

The two splits have the following similarities and differences:

*Train texts are shorter than dev texts. Each dev text is longer context than a corresponding train text. A causal graph represents the latent causal structure of the event sequence described in the text.

*Train examples include corresponding *temporal* questions and answers from TORQUE. In contrast, dev examples have corresonding *event structure* questions and answers from ESTER (see paper for details).

*To make schemas, use event_types instead of natural language nodes in causal_graph!

*In train, event_types are assigned at node level, while in the dev split event_types are at the graph level (as single concatenated string).

*We include `noncausal_event_types` with the train examples; these represent text mentions that judged to not contribute to the overall causal story

*Train and dev causal graphs have natural language descriptions of events from the texts. As dev causal graphs are based on longer texts, dev graphs have more nodes than train graphs.

*Train causal graphs include both short and full causal relations (see paper).

*Dev causal graphs include annotations for 'saliency' (most important paths in the causal graph)


Summarizing the above, an instance in each split takes these forms:

train

{'split': 'train',
'source': str 'torque',
'torque_id': str,
'text': str,
'questions': List[str],
'answers': List[List[str]],
'event_types: Dict(@node str: @event_type str),
'noncausal_event_types': Dict(@mention-text str: @event_type str)
'causal_graph': List[Dict('head': str, 'tail': str, 'rel': str, 'rel_full': str)],
}

dev

{'split': 'dev',
'source': 'ester',
'torque_id': str,
'text': str,
'questions': List[str],
'answers': List[str],
'schema_graph_event_types: str,
'causal_graph': List[Dict('head': str, 'tail': str, 'rel': str, 'saliency': bool 0|1)],
}


Here are two full example for comparison side to side. Note that the train example's `torque_id` is the same as the dev example (the dev text contains the train text).


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
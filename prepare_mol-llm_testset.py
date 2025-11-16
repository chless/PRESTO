from presto.chemistry_tools import smiles
from presto.chemistry_tools.reaction import multicomponent_smiles_to_list, list_to_multicomponent_smiles
from presto.chemistry_tools.smiles import convert_to_canonical_smiles
from presto.constants import ROLE_ASSISTANT, ROLE_USER, ROLE_SYSTEM
import selfies as sf
from typing import List
import random
import datasets
import re


MOLECULE_TOKEN = "<molecule_2d>"

testset_path = '/data/text-mol/data/Mol-LLM-v7.1/mol_llm_testset_general'
testset = datasets.load_from_disk(testset_path)
original_testset_size = len(testset)

# from the testset, check whether field "smiles" gives canonical smiles, and if not, exclude those samples
def is_canonical_smiles(smiles: str, task) -> bool:
    if task in ['chebi-20-text2mol', 'smol-molecule_generation']:
        return True
    else:
        canonical_smiles = convert_to_canonical_smiles(smiles)
        flag = canonical_smiles is not None 
        return flag
    
def is_processable(instance):
    try:
        process_instance(instance)
        return True
    except:
        return False

num_procs = 10

from presto.chemistry_tools.reaction import multicomponent_smiles_to_list, list_to_multicomponent_smiles
from presto.chemistry_tools.smiles import convert_to_canonical_smiles
from presto.constants import ROLE_ASSISTANT, ROLE_USER, ROLE_SYSTEM
import selfies as sf
from typing import List
import random

MOLECULE_TOKEN = "<molecule_2d>"

def process_reaction_equation(reaction, format = "smiles", token=True)->List[str]:
    smiles_list = multicomponent_smiles_to_list(reaction)
    smiles_list = [convert_to_canonical_smiles(smi) for smi in smiles_list]
    selfies_list = [sf.encoder(smi) for smi in smiles_list]
    if token:
        molecules = ".".join([MOLECULE_TOKEN for _ in range(len(smiles_list))])
    elif format == "smiles":
        molecules = ".".join(smiles_list)
    elif format == "selfies":
        molecules = ".".join(selfies_list)
    else:
        raise ValueError(f"Unsupported molecule format: {format}")
    
    return selfies_list, smiles_list, molecules

def conversation_test_forward_reaction_prediction(instance, format = "smiles", token=True):
    assert instance['input_mol_string'] in instance['prompt_text']
    input = instance['smiles']
    output = sf.decoder(instance['target'])
    task = instance['task']
    
    SYSTEM_PROMPT = """You are a chemist. Now you are given a reaction equation. Please predict the product of the reaction. The reaction equation has the following format:
    ```
    reactant1.reactant2. ... .reactantN>>product
    ```
    Your task is to predict the <REP_1> representation of the product molecule. We provide the <REP_2> of the reactants."""

    PROMPT_TEMPLATES = [
        {
            "input": "<MOLECULE> Based on the reactants and reagents given above, suggest a possible product.",
            "output": "A possible product can be <OUTPUT> .",
        },
        {
            "input": "Based on the given reactants and reagents: <MOLECULE>, what product could potentially be produced?",
            "output": "The product can be <OUTPUT> .",
        },
        {
            "input": "Given the following reactants and reagents, please provide a possible product. <MOLECULE>",
            "output": "<OUTPUT> .",
        },
        {
            "input": "<MOLECULE> Given the above reactants and reagents, what could be a probable product of their reaction?",
            "output": "A probable product could be <OUTPUT> .",
        },
        {
            "input": "Please provide a feasible product that could be formed using these reactants and reagents: <MOLECULE> .",
            "output": "<OUTPUT> .",
        },
        {
            "input": "Consider that for a chemical reaction, if <MOLECULE> is/are the reactants and reagents, what can be the product?",
            "output": "<OUTPUT> .",
        },
        {
            "input": "Propose a potential product given these reactants and reagents. <MOLECULE>",
            "output": "<OUTPUT> .",
        },
        {
            "input": "Predict the product of a chemical reaction with <MOLECULE> as the reactants and reagents.",
            "output": "<OUTPUT> .",
        },
        {
            "input": "Can you tell me the potential product of a chemical reaction that uses <MOLECULE> as the reactants and reagents?",
            "output": "Sure. A potential product: <OUTPUT> .",
        },
        {
            "input": "Using <MOLECULE> as the reactants and reagents, tell me the potential product.",
            "output": "<OUTPUT> .",
        },
        {
            "input": "Predict a possible product from the listed reactants and reagents. <MOLECULE>",
            "output": "<OUTPUT> .",
        },
        {
            "input": "<MOLECULE> Considering the given starting materials, what might be the resulting product in a chemical reaction?",
            "output": "<OUTPUT> .",
        },
        {
            "input": "A chemical reaction has started with the substance(s) <MOLECULE> as the reactants and reagents, what could be a probable product?",
            "output": "A probable product: <OUTPUT> .",
        }
    ]

    selfies, smiles, molecules = process_reaction_equation(input, format, token)
    _, _, output = process_reaction_equation(output, format, False)
    prompt_template = random.choice(PROMPT_TEMPLATES)
    input_template = prompt_template["input"].replace("<MOLECULE>", molecules)
    system_prompt = SYSTEM_PROMPT.replace("<REP_1>", "structure" if token else format.upper()).replace("<REP_2>", format.upper())
    
    content = input_template
    assert len(smiles), f"{task} processing returned empty smiles list.\nInstance: {instance}"
    return {
        "smiles": smiles,
        "messages": [
            {
                "role": ROLE_SYSTEM,
                "content": system_prompt
            },
            {
                "role": ROLE_USER,
                "content": content
            }
        ]
    }

def conversation_test_molecule_generation(instance):
    prompt_text = instance['prompt_text']
    output = instance['target']
    task = instance['task']


    SYSTEM_PROMPT = """You are a chemist. Now you are given a description of a molecule. Please generate a molecule that meets the description."""

    GENERATION_PHRASES = [
        'Based on the given information, generate a molecule that meets the desired specifications: <CAPTION>',
        'Give me a molecule that satisfies the conditions outlined in the description: <CAPTION>',
        'Generate a molecule based on this description: <CAPTION>',
        'Can you create a molecule that matches the given characteristics? <CAPTION>',
        'I need a molecule that meets the following conditions: <CAPTION> Please represent the molecule in SMILES.',
        'Suppose there is a molecule that meets the following description: <CAPTION> Please write the SMILES representation of it.',
        '<CAPTION> Use the above information to create a molecule.',
        'Build a molecule that meets the requirement: <CAPTION>',
        'Generate a molecule that fulfills the requirement: <CAPTION>',
        'Conceptualize a molecule that meets the specified attribute(s): <CAPTION>',
        'Come up with a molecule based on the description: <CAPTION>',
        'Could you please return a molecule that adheres to this description? <CAPTION>',
        'I give you a description of a molecule, and you need to return one molecule in SMILES that meets the description. The description: <CAPTION>', 
    ]

    caption_pattern = r'<DESCRIPTION>.*?</DESCRIPTION>'
    caption = re.search(caption_pattern, prompt_text).group(0)
    caption = caption.replace('<DESCRIPTION>', '').replace('</DESCRIPTION>', '')
    smiles = ["None"]
    assert len(smiles), f"{task    } processing returned empty smiles list.\nInstance: {instance}"
    return {
        "smiles": ["None"],        
        "messages": [
            {
                "role": ROLE_SYSTEM,
                "content": SYSTEM_PROMPT
            },
            {
                "role": ROLE_USER,
                "content": random.choice(GENERATION_PHRASES).replace("<CAPTION>", caption)
            }
        ]}


def conversation_test_molecule_captioning(instance, format = "smiles", token=True):
    assert instance['input_mol_string'] in instance['prompt_text']
    output = instance['target']
    input = instance['smiles']
    task = instance['task']

    SYSTEM_PROMPT = """You are a chemist. Now you are given a representation of a molecule. Please help me to understand the molecule."""

    CAPTION_PHRASES = [
    f'Could you give me a brief overview of this molecule {MOLECULE_TOKEN} ?',
    f'Could you provide a description of this molecule {MOLECULE_TOKEN} ?',
    f'Describe this molecule: {MOLECULE_TOKEN} .',
    f'Please give me some details about this molecule {MOLECULE_TOKEN} .',
    f'Provide a brief overview of this molecule {MOLECULE_TOKEN} .',
    f'Provide a description of this molecule {MOLECULE_TOKEN} .',
    f'Given the molecule {MOLECULE_TOKEN}, what can you tell me about it?',
    f'Tell me something about this molecule: {MOLECULE_TOKEN}',
    f'{MOLECULE_TOKEN} What do you know about the molecule?',
    f"I'd like a short overview about this molecule {MOLECULE_TOKEN}. Can you do that?",
    f"{MOLECULE_TOKEN} The above is a compound. Could you please tell me something about it?"
    ]
    selfies, smiles, molecules = process_reaction_equation(input, format, token)
    assert len(smiles), f"{task    } processing returned empty smiles list.\nInstance: {instance}"
    return {
        "smiles": smiles,
                "messages": [
                    {
                        "role": ROLE_SYSTEM,
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": ROLE_USER,
                        "content": random.choice(CAPTION_PHRASES)
                    }
                ]
            }

def conversation_test_property_prediction(instance, format = "smiles", token=True):
    assert instance['input_mol_string'] in instance['prompt_text']
    input = instance['smiles']
    output = instance['target']
    task = instance['task']


    SYSTEM_PROMPT = """You are a chemist. Now you are given a representation of a molecule.  Please predict a molecular property asked by a instruction."""

    instruction = instance['instruction'].replace('<INPUT>', MOLECULE_TOKEN)
    selfies, smiles, molecules = process_reaction_equation(input, format, token)
    assert len(smiles), f"{task} processing returned empty smiles list.\nInstance: {instance}"
    return {
                "smiles": smiles,
                "messages": [
                    {
                        "role": ROLE_SYSTEM,
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": ROLE_USER,
                        "content": instruction
                    }
                ]
            }

def conversation_test_reagent_prediction(instance, format = "smiles", token=True):
    prompt_text = instance['prompt_text']
    selfies_pattern = r'<SELFIES>.*?</SELFIES>'
    reactants, products = re.findall(selfies_pattern, prompt_text)
    reactants = reactants.replace('<SELFIES>', '').replace('</SELFIES>', '').strip()
    reactants = sf.decoder(reactants)
    products = products.replace('<SELFIES>', '').replace('</SELFIES>', '').strip()
    products = sf.decoder(products)
    output = sf.decoder(instance['target'])
    task = instance['task']

    SYSTEM_PROMPT = """You are a chemist. Now you are given a reaction equation. Please predict the possible reagents of the reaction. The reaction equation has the following format:
    ```
    reactant1.reactant2. ... .reactantN>>product
    ```
    Your task is to predict the <REP_1> representation of the reagents molecule. We provide the <REP_2> of the reactions."""

    PROMPT_TEMPLATES = [
        {
            "input": "<MOLECULE> Based on the given chemical reaction, can you propose some likely reagents that might have been utilized?",
            "output": "A possible reagents can be <OUTPUT> .",
        },
        {
            "input": "Based on the given chemical reaction <MOLECULE>, suggest some possible reagents.",
            "output": "The reagents can be <OUTPUT> .",
        },
        {
            "input": "Can you provide potential reagents for the following chemical reaction? <MOLECULE>",
            "output": "<OUTPUT> .",
        },
        {
            "input": "Can you suggest some reagents that might have been used in the given chemical reaction? <MOLECULE>",
            "output": "A probable reagents could be <OUTPUT> .",
        },
        {
            "input": "<MOLECULE> From the provided chemical reaction, propose some possible reagents that could have been used.",
            "output": "<OUTPUT> .",
        },
        {
            "input": "Given the following chemical reaction <MOLECULE>, what are some potential reagents that could have been employed?",
            "output": "<OUTPUT> .",
        },
        {
            "input": "Given the following reaction <MOLECULE>, what are some possible reagents that could have been utilized?",
            "output": "<OUTPUT> .",
        },
        {
            "input": "Given this chemical reaction <MOLECULE>, what are some reagents that could have been used?",
            "output": "<OUTPUT> .",
        },
        {
            "input": "<MOLECULE> Please propose potential reagents that might have been utilized in the provided chemical reaction.",
            "output": "Sure. A potential answer could be: <OUTPUT> .",
        },
        {
            "input": "Please provide possible reagents based on the following chemical reaction <MOLECULE>.",
            "output": "<OUTPUT> .",
        },
        {
            "input": "Please suggest some possible reagents that could have been used in the following chemical reaction <MOLECULE>.",
            "output": "<OUTPUT> .",
        },
        {
            "input": "What reagents could have been utilized in the following chemical reaction? <MOLECULE>",
            "output": "<OUTPUT> .",
        },
    ]
    react_selfies_list, react_smiles_list, react_molecules = process_reaction_equation(reactants, format, token)
    prod_selfies_list, prod_smiles_list, prod_molecules = process_reaction_equation(products, format, token)
    selfies_list = react_selfies_list + prod_selfies_list
    smiles_list = react_smiles_list + prod_smiles_list
    _, _, output = process_reaction_equation(output, format, False)
    prompt_template = random.choice(PROMPT_TEMPLATES)
    input_template = prompt_template["input"].replace("<MOLECULE>", react_molecules+">>"+prod_molecules)
    output_template = prompt_template["output"].replace("<OUTPUT>", output)
    system_prompt = SYSTEM_PROMPT.replace("<REP_1>", "structure" if token else format.upper()).replace("<REP_2>", format.upper())
    
    content = input_template

    smiles = smiles_list
    assert len(smiles), f"{task    } processing returned empty smiles list.\nInstance: {instance}"
    return {
        "smiles": smiles,
        "messages": [
            {
                "role": ROLE_SYSTEM,
                "content": system_prompt
            },
            {
                "role": ROLE_USER,
                "content": content
            }
        ],
    }

def conversation_test_retrosynthesis(instance, format = "smiles", token=True):
    assert instance['input_mol_string'] in instance['prompt_text']
    input = instance['smiles']
    output = instance['target']
    task = instance['task']

    SYSTEM_PROMPT = """You are a chemist. Now you are given a product molecule. Please predict the the reactant molecules of the reaction.
    The reaction equation has the following format:
    ```
    reactant1.reactant2. ... .reactantN>>product
    ```
    Your task is to predict the <REP_1> representation of the reactant molecule. We provide the <REP_2> of the reactants."""

    PROMPT_TEMPLATES = [
        {
            "input": "Based on the given product, provide some plausible reactants that might have been utilized to prepare it. <INPUT>",
            "output": "<OUTPUT>"
        },
        {
            "input": "Can you identify the reactant(s) that might result in the given product <INPUT> ?",
            "output": "<OUTPUT>"
        },
        {
            "input": "Given the following product, please provide possible reactants. <INPUT>",
            "output": "Possible reactant(s): <OUTPUT> ."
        },
        {
            "input": "Do retrosynthesis with the product <INPUT> .",
            "output": "OK. The reactants may be <OUTPUT> ."
        },
        {
            "input": "<INPUT> Given the product provided, propose some possible reactants that could have been employed in its formation.",
            "output": "Here are possible reactants: <OUTPUT> ."
        },
        {
            "input": "To synthesis <INPUT>, what are the possible reactants? Write in the SMILES representation.",
            "output": "<OUTPUT>"
        },
        {
            "input": "Provide the potential reactants that may be used to produce the product <INPUT> .",
            "output": "The potential reactants: <OUTPUT> ."
        },
        {
            "input": "What reactants could lead to the production of the following product? <INPUT>",
            "output": "<OUTPUT>"
        },
        {
            "input": "With the given product <INPUT>, suggest some likely reactants that were used in its synthesis.",
            "output": "<OUTPUT>"
        },
        {
            "input": "Identify possible reactants that could have been used to create the specified product. <INPUT>",
            "output": "<OUTPUT>"
        },
        {
            "input": "Could you tell which reactants might have been used to generate the following product? <INPUT>",
            "output": "<OUTPUT>"
        },
        {
            "input": "Suggest possible substances that may have been involved in the synthesis of the presented compound. <INPUT>",
            "output": "<OUTPUT>"
        },
        {
            "input": "Can you list the reactants that might result in the chemical product <INPUT> ?",
            "output": "<OUTPUT>"
        }
    ]

    selfies, smiles, molecules = process_reaction_equation(input, format, token)
    prompt_template = random.choice(PROMPT_TEMPLATES)
    input_template = prompt_template["input"].replace("<INPUT>", molecules)
    system_prompt = SYSTEM_PROMPT.replace("<REP_1>", "structure" if token else format.upper()).replace("<REP_2>", format.upper())
    
    content = input_template
    try:
        len(smiles)
    except:
        print(f"{task} processing returned empty smiles list.\n smiles: {smiles}")
    return {
        "smiles": smiles,
        "messages": [
            {
                "role": ROLE_SYSTEM,
                "content": system_prompt
            },
            {
                "role": ROLE_USER,
                "content": content
            }
        ],
    }

unique_tasks = list(set(testset['task']))


example_tasks = [
 'forward_reaction_prediction',
 'retrosynthesis',
 'reagent_prediction',
 'smol-property_prediction-esol',
'smol-property_prediction-bbbp',
 'chebi-20-mol2text',
 'chebi-20-text2mol',
 ]
"""
task_examples = {}
for task in example_tasks:
    example = testset.filter(lambda x: x['task']==task, num_proc=num_procs)[0]
    task_examples[task] = example

fs = conversation_test_forward_reaction_prediction(task_examples['forward_reaction_prediction'])
rs = conversation_test_retrosynthesis(task_examples['retrosynthesis'])
rp = conversation_test_reagent_prediction(task_examples['reagent_prediction'])
pp = conversation_test_property_prediction(task_examples['smol-property_prediction-esol'])
mc = conversation_test_molecule_captioning(task_examples['chebi-20-mol2text'])
mg = conversation_test_molecule_generation(task_examples['chebi-20-text2mol'])
print('Task: ', fs['messages'][1]['content'], '\n')
print('Task: ', rs['messages'][1]['content'], '\n')
print('Task: ', rp['messages'][1]['content'], '\n')
print('Task: ', pp['messages'][1]['content'], '\n')
print('Task: ', mc['messages'][1]['content'], '\n')
print('Task: ', mg['messages'][1]['content'], '\n')
"""

def process_instance(instance):
    try:
        task = instance['task']
        if task in [
            'forward_reaction_prediction',
            'smol-forward_synthesis',
            'presto-forward_reaction_prediction',
            'orderly-forward_reaction_prediction',
        ]:
            task_specific_processor = conversation_test_forward_reaction_prediction
        elif task in [
            'retrosynthesis',
            'smol-retrosynthesis',
            'orderly-retrosynthesis',
            'presto-retrosynthesis',
        ]:
            task_specific_processor = conversation_test_retrosynthesis
        elif task in ['reagent_prediction']:
            task_specific_processor = conversation_test_reagent_prediction
        elif task in [
            'smol-property_prediction-esol',
            'aqsol-logS',
            'smol-property_prediction-lipo',
            'qm9_homo',
            'qm9_lumo',
            'qm9_homo_lumo_gap',
            'bace',
            'smol-property_prediction-bbbp',
            'smol-property_prediction-hiv',
            'smol-property_prediction-sider',
            'smol-property_prediction-clintox',
        ]:
            task_specific_processor = conversation_test_property_prediction
        elif task in ['chebi-20-mol2text', 'smol-molecule_captioning']:
            task_specific_processor = conversation_test_molecule_captioning
        elif task in ['chebi-20-text2mol', 'smol-molecule_generation']:
            task_specific_processor = conversation_test_molecule_generation
        else:
            raise ValueError(f"Unknown task: {task}")

        instance['ground_truth'] = instance.get('target')
        out_dict = task_specific_processor(instance)

        instance['messages'] = out_dict['messages']
        instance['molecules'] = {
            "smiles": out_dict['smiles']
        }

        for k, v in instance.items():
            assert v is not None, f"Task-{task} has unprocessed field: {k}.\n{instance}"

        return instance

    except Exception as e:
        print("=" * 80)
        print(f"‼️ Error while processing instance with task {instance.get('task')}")
        # Pretty-print the instance so you can inspect it
        import json
        try:
            print(json.dumps(instance, indent=2, ensure_ascii=False))
        except Exception:
            print("Could not JSON-encode instance:", instance)
        print("Exception type:", type(e).__name__)
        print("Exception message:", str(e))
        print("=" * 80)
        raise  # re-raise so the map() stops and shows traceback

"""
task_specific_testsets = {}
processed_testsets = {}
for task in unique_tasks:
    task_specific_testsets[task] = testset.filter(lambda x: x['task']==task, num_proc=10)

for task in unique_tasks:
    print(f"Processing task: {task} with {len(task_specific_testsets[task])} instances.")
    processed_testsets[task] = task_specific_testsets[task].map(process_instance, num_proc=10)
"""

filtered_testset = testset.filter(lambda x: is_processable(x), num_proc=num_procs)
print("filtered ratio: ", len(filtered_testset) / len(testset))
presto_testset = filtered_testset.map(process_instance, num_proc=num_procs)
# save the processed testset
presto_testset.save_to_disk('/data/text-mol/data/Mol-LLM-v7.1/mol_llm_testset_presto')

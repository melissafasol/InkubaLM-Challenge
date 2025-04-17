'''Script to data augmentation to the sentiment datasets. '''

import pandas as pd
import random

synonyms_xnli = {
    "hau": {
        "nace": ["na fada", "na bayyana"],
        "ba": ["babu", "baya"],
        "ya": ["shi", "mutumin"],
    },
    "swa": {
        "kanuni": ["sheria", "mwongozo"],
        "maduka": ["duka", "bohari"],
        "habari": ["taarifa", "maelezo"],
    }
}

synonyms_sent = {
    "hausa": {
        "kyakkyawa": ["mai kyau", "nagari"],
        "rahama": ["jinƙai", "tausayawa"],
        "korau": ["mummuna", "banza"],
        "jama'a": ["mutane", "al'umma"],
    },
    "swahili": {
        "zuri": ["safi", "ya kupendeza"],
        "mbaya": ["haifai", "bovu"],
        "hakuna": ["hamna", "siyo"],
        "chanya": ["nzuri", "ya kufurahisha"],
    }
}

lang_map_xnli = {'hau': 'hau', 'swa': 'swa'}
lang_map_sent = {'hausa': 'hausa', 'swahili': 'swahili'}

def synonym_replace(text, lang, syn_dict):
    words = text.split()
    new_words = words[:]
    for i, word in enumerate(words):
        if word in syn_dict.get(lang, {}):
            new_words[i] = random.choice(syn_dict[lang][word])
            break
    return ' '.join(new_words)

def random_swap(text):
    words = text.split()
    if len(words) < 2:
        return text
    i, j = random.sample(range(len(words)), 2)
    words[i], words[j] = words[j], words[i]
    return ' '.join(words)

def random_deletion(text, p=0.1):
    words = text.split()
    if len(words) == 1:
        return text
    new_words = [w for w in words if random.random() > p]
    return ' '.join(new_words) if new_words else random.choice(words)



def apply_augmentation(row, synonym_dict, lang_key_map=None):
    """
    Generic text augmentation function for any task.
    
    Args:
        row (dict): One row of your dataset
        synonym_dict (dict): Dictionary of synonyms by language
        lang_key_map (dict, optional): Mapping from language codes to synonym dict keys
    
    Returns:
        dict or None: Augmented row or None if no change
    """
    text = row["inputs"]
    lang = row["langs"].strip().lower()
    
    # Map language code if needed
    if lang_key_map:
        lang = lang_key_map.get(lang, lang)
    
    methods = [synonym_replace, random_swap, random_deletion]
    method = random.choice(methods)
    
    new_text = method(text, lang, synonym_dict) if method == synonym_replace else method(text)
    
    if new_text.strip() == text.strip():
        return None  # No real change
    
    new_row = row.copy()
    new_row["inputs"] = new_text
    new_row["is_augmented"] = True
    return new_row

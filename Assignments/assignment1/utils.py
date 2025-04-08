from nltk.corpus import wordnet as wn
import nltk

def setup_wordnet():
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

def get_synset_name(synset_id):
    if not synset_id.startswith('n'):
        return None
    
    offset = int(synset_id[1:])
    
    synset = wn.synset_from_pos_and_offset('n', offset)
    
    if synset:
        name = synset.name().split('.')[0]
        definition = synset.definition()
        return {
            'name': name,
            'definition': definition
        }
    return None

if __name__ == '__main__':
    import os
    import json
    from datetime import datetime
    
    setup_wordnet()
    
    mini_imagenet_path = os.path.join('data', 'mini-imagenet')
    output_file = os.path.join('data', 'index.txt')
    
    class_folders = [f for f in os.listdir(mini_imagenet_path) 
                    if os.path.isdir(os.path.join(mini_imagenet_path, f))]
    
    class_info = []
    for folder in sorted(class_folders):
        result = get_synset_name(folder)
        if result:
            info = {
                'synset_id': folder,
                'name': result['name'],
                'definition': result['definition'],
                'image_count': len(os.listdir(os.path.join(mini_imagenet_path, folder)))
            }
            class_info.append(info)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Mini-ImageNet Dataset Class Information\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        for info in class_info:
            f.write(f"Class ID: {info['synset_id']}\n")
            f.write(f"Name: {info['name']}\n")
            f.write(f"Definition: {info['definition']}\n")
            f.write(f"Image Count: {info['image_count']}\n")
            f.write("-" * 80 + "\n\n")
        
        f.write(f"\nSummary:\n")
        f.write(f"Total Classes: {len(class_info)}\n")
        f.write(f"Total Images: {sum(info['image_count'] for info in class_info)}\n")
    
    print(f"类别信息已保存到: {output_file}")
import os
import numpy as np
from nltk.corpus import wordnet as wn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils import setup_wordnet, get_synset_name

def get_similarity_matrix(synset_ids):
    """计算类别间的相似度矩阵"""
    n = len(synset_ids)
    sim_matrix = np.zeros((n, n))
    
    for i, id1 in enumerate(synset_ids):
        synset1 = wn.synset_from_pos_and_offset('n', int(id1[1:]))
        for j, id2 in enumerate(synset_ids):
            if i <= j:
                synset2 = wn.synset_from_pos_and_offset('n', int(id2[1:]))
                # 使用路径相似度度量
                similarity = synset1.path_similarity(synset2)
                sim_matrix[i, j] = similarity if similarity else 0
                sim_matrix[j, i] = sim_matrix[i, j]
    
    return sim_matrix

def visualize_categories(sim_matrix, synset_ids, names):
    """使用t-SNE可视化类别分布"""
    tsne = TSNE(n_components=2, random_state=42)
    coords = tsne.fit_transform(sim_matrix)
    
    plt.figure(figsize=(15, 10))
    plt.scatter(coords[:, 0], coords[:, 1], c='blue', alpha=0.5)
    
    # 添加类别标签
    for i, (x, y) in enumerate(coords):
        plt.annotate(
            names[i],
            (x, y),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            alpha=0.8
        )
    
    plt.title('Similarity between categories')
    plt.xlabel('t-SNEdimension1')
    plt.ylabel('t-SNEdimension2')
    
    # 保存结果
    plt.savefig('data/category_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_categories():
    """分析并可视化类别分布"""
    # 确保WordNet数据已下载
    setup_wordnet()
    
    # 获取所有类别
    mini_imagenet_path = os.path.join('data', 'mini-imagenet')
    class_folders = sorted([f for f in os.listdir(mini_imagenet_path) 
                          if os.path.isdir(os.path.join(mini_imagenet_path, f))])
    
    # 获取类别信息
    synset_ids = []
    names = []
    for folder in class_folders:
        result = get_synset_name(folder)
        if result:
            synset_ids.append(folder)
            names.append(result['name'])
    
    # 计算相似度矩阵
    sim_matrix = get_similarity_matrix(synset_ids)
    
    # 可视化结果
    visualize_categories(sim_matrix, synset_ids, names)
    
    # 保存相似度矩阵
    np.save('data/similarity_matrix.npy', sim_matrix)
    
    return sim_matrix, synset_ids, names

def suggest_categories(sim_matrix, synset_ids, names, n_fine=5, n_coarse=5):
    """推荐细粒度和粗粒度分类的类别组合"""
    # 计算每个类别与其他类别的平均相似度
    avg_similarities = np.mean(sim_matrix, axis=1)
    
    # 找出相似度最高的类别组(细粒度分类)
    fine_grained_indices = np.argsort(avg_similarities)[-n_fine:]
    
    # 找出相似度最低的类别组(粗粒度分类)
    coarse_grained_indices = np.argsort(avg_similarities)[:n_coarse]
    
    # 保存推荐结果
    with open('data/category_recommendations.txt', 'w', encoding='utf-8') as f:
        f.write("推荐的类别组合:\n\n")
        
        f.write("细粒度分类类别(相似度高):\n")
        for idx in fine_grained_indices:
            f.write(f"{synset_ids[idx]}: {names[idx]}\n")
        
        f.write("\n粗粒度分类类别(相似度低):\n")
        for idx in coarse_grained_indices:
            f.write(f"{synset_ids[idx]}: {names[idx]}\n")

if __name__ == '__main__':
    # 运行分析
    sim_matrix, synset_ids, names = analyze_categories()
    
    # 获取推荐的类别组合
    suggest_categories(sim_matrix, synset_ids, names)
    
    print("分析完成!结果已保存到data目录下:")
    print("- category_visualization.png: 类别分布可视化")
    print("- similarity_matrix.npy: 相似度矩阵")
    print("- category_recommendations.txt: 推荐的类别组合")
"""文档话题聚类。

纯 Python 的稀疏 TF-IDF + 球面 k-means 实现，面向中文/英文混合语料：

* 文档向量使用 L2 归一化的 TF-IDF 稀疏向量，距离用余弦距离；
* 话题数可固定，也可通过轮廓信息自动估计；
* 初始化采用确定性的“最远质心”策略，同一批文档多次运行结果稳定；
* 关键词直接来自簇质心中的真实词项，标签由关键词生成，不引入幻觉名称。

增量更新由上层 :class:`~web` 中使用的 TopicModelService 维护，本模块只提供
无状态、易测试的聚类算法。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Optional

from .lexicon import STOPWORDS
from .segmenter import Segmenter


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-@][a-z0-9]+)*")

# 关键词/标签提取时额外过滤的泛化词
TOPIC_STOPWORDS = {
    "一个", "一种", "这个", "那个", "这些", "那些", "进行", "需要", "可以",
    "应该", "可能", "表示", "认为", "相关", "内容", "问题", "方式", "方面",
    "通过", "使用", "提供", "支持", "包括", "当前", "以及", "没有", "我们",
}


def tokenize_text(text: str, segmenter: Optional[Segmenter] = None) -> list[str]:
    """把文档切成适合话题模型的实义词。"""
    seg = segmenter or Segmenter()
    words = seg.cut(text)
    tokens: list[str] = []
    for word in words:
        word = word.strip().lower()
        if not word or word in STOPWORDS or word in TOPIC_STOPWORDS:
            continue
        if _TOKEN_RE.fullmatch(word):
            if len(word) >= 2 and not word.isdigit():
                tokens.append(word)
            elif len(word) >= 3:
                tokens.append(word)
            continue
        if any("一" <= ch <= "鿿" for ch in word):
            if len(word) >= 2:
                tokens.append(word)
            continue
        # 标点和其它符号丢弃
    return tokens


def sparse_dot(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(value * b.get(term, 0.0) for term, value in a.items())


def sparse_norm(vec: dict[str, float]) -> float:
    return math.sqrt(sum(v * v for v in vec.values()))


def normalize_sparse(vec: dict[str, float]) -> dict[str, float]:
    norm = sparse_norm(vec)
    if norm <= 1e-12:
        return {}
    return {term: value / norm for term, value in vec.items()}


def cosine_sparse(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    return sparse_dot(a, b) / (sparse_norm(a) * sparse_norm(b))


def average_vectors(vectors: list[dict[str, float]],
                    normalize: bool = True) -> dict[str, float]:
    """求稀疏向量均值；normalize=True 时返回球面质心。"""
    center: dict[str, float] = {}
    for vec in vectors:
        for term, value in vec.items():
            center[term] = center.get(term, 0.0) + value
    if vectors:
        inv = 1.0 / len(vectors)
        center = {term: value * inv for term, value in center.items()}
    return normalize_sparse(center) if normalize else center


class TfidfModel:
    """根据已缓存的词频与文档频次增量构造 TF-IDF 向量。"""

    def __init__(self, document_frequency: dict[str, int], document_count: int):
        self.df = dict(document_frequency)
        self.n = max(0, int(document_count))

    def idf(self, term: str) -> float:
        # 与 compute_tfidf 保持同一平滑形式
        return math.log((self.n + 1) / (self.df.get(term, 0) + 1)) + 1.0

    def vectorize(self, term_counts: dict[str, int]) -> dict[str, float]:
        total = sum(term_counts.values())
        if total <= 0:
            return {}
        vec = {
            term: (count / total) * self.idf(term)
            for term, count in term_counts.items()
            if count > 0
        }
        return normalize_sparse(vec)


def estimate_topic_count(vectors: list[dict[str, float]],
                         max_clusters: int = 8) -> int:
    """自动估计话题数。

    使用文档到质心的近似轮廓系数：不需要 O(n²) 两两比较，适合分片存储中的
    较大语料。再对单文档簇和极小簇做惩罚，避免单纯追求轮廓值导致碎簇。
    """
    n = len(vectors)
    if n <= 1:
        return 1
    max_clusters = max(1, min(int(max_clusters), n))
    if max_clusters == 1:
        return 1

    best_k = 1
    best_score = -2.0
    for k in range(1, max_clusters + 1):
        labels, centers = spherical_kmeans(vectors, k)
        if k == 1:
            # 单簇时用紧致度估计；显著紧致才保留单话题
            score = sum(cosine_sparse(vectors[i], centers[0])
                        for i in range(n)) / n - 0.12
        else:
            sizes = Counter(labels)
            values = []
            for i, vec in enumerate(vectors):
                own = labels[i]
                own_sim = cosine_sparse(vec, centers[own])
                other_sim = max(
                    cosine_sparse(vec, centers[c])
                    for c in range(k) if c != own and centers[c]
                ) if k > 1 else 0.0
                own_dist = max(0.0, 1.0 - own_sim)
                other_dist = max(0.0, 1.0 - other_sim)
                values.append((other_dist - own_dist) /
                              max(other_dist, own_dist, 1e-9))
            score = sum(values) / n
            tiny = sum(1 for size in sizes.values() if size < 2)
            score -= 0.08 * tiny / n
        if score > best_score:
            best_score = score
            best_k = k
    return best_k


def spherical_kmeans(vectors: list[dict[str, float]], k: int,
                     max_iter: int = 60,
                     initial_centers: Optional[list[dict[str, float]]] = None,
                     ) -> tuple[list[int], list[dict[str, float]]]:
    """确定性球面 k-means。

    返回 ``(每个向量的簇编号, 归一化质心)``。空簇会把距当前质心最远的文档
    迁移过去，避免退化；相同输入始终得到相同输出。
    """
    n = len(vectors)
    k = max(1, min(int(k), n))
    if n == 0:
        return [], []
    if n == 1:
        return [0], [normalize_sparse(vectors[0])]

    if initial_centers is not None and len(initial_centers) >= k:
        centers = [normalize_sparse(dict(c)) for c in initial_centers[:k]]
    else:
        centers = _initial_centers(vectors, k)

    labels = [0] * n
    for _ in range(max_iter):
        new_labels = [_nearest_center(vec, centers) for vec in vectors]
        changed = sum(a != b for a, b in zip(labels, new_labels))
        labels = new_labels
        new_centers = [
            average_vectors([vectors[i] for i, label in enumerate(labels)
                             if label == c])
            for c in range(k)
        ]
        _repair_empty_clusters(vectors, labels, new_centers)
        centers = new_centers
        if changed == 0:
            break
    return labels, centers


def _initial_centers(vectors: list[dict[str, float]], k: int) -> list[dict[str, float]]:
    """确定性 k-means++ 风格初始化。

    第一个质心取非空词项最多的文档，后续每轮取到已有质心最远的文档。按向量
    在列表中的下标打破平局，不使用随机数。
    """
    chosen: list[int] = []
    candidates = [i for i, vec in enumerate(vectors) if vec]
    if not candidates:
        return [{} for _ in range(k)]
    first = max(candidates, key=lambda i: (len(vectors[i]), -i))
    chosen.append(first)
    while len(chosen) < k:
        best_idx = -1
        best_distance = -2.0
        for i, vec in enumerate(vectors):
            if i in chosen or not vec:
                continue
            nearest = max(
                (cosine_sparse(vec, vectors[c]) for c in chosen),
                default=0.0,
            )
            distance = 1.0 - nearest
            if distance > best_distance + 1e-12 or (
                abs(distance - best_distance) <= 1e-12 and
                    (best_idx < 0 or i < best_idx)):
                best_distance = distance
                best_idx = i
        if best_idx < 0:
            break
        chosen.append(best_idx)
    centers = [normalize_sparse(vectors[i]) for i in chosen]
    while len(centers) < k:
        # 文档数少于非空有效向量时，用零质心占满请求的簇数。
        centers.append({})
    return centers


def _nearest_center(vec: dict[str, float], centers: list[dict[str, float]]) -> int:
    best = 0
    best_sim = -2.0
    for c, center in enumerate(centers):
        sim = cosine_sparse(vec, center) if center else -1.0
        if sim > best_sim + 1e-12:
            best_sim = sim
            best = c
    return best


def _repair_empty_clusters(vectors: list[dict[str, float]], labels: list[int],
                           centers: list[dict[str, float]]) -> None:
    k = len(centers)
    while True:
        empty = [c for c in range(k) if not centers[c]]
        if not empty:
            return
        target = empty[0]
        # 找当前离自己质心最远的文档作为新质心，降低 SSE。
        donor_label = 0
        donor_idx = -1
        donor_dist = -1.0
        for i, vec in enumerate(vectors):
            label = labels[i]
            members = sum(1 for x in labels if x == label)
            if members <= 1:
                continue
            dist = 1.0 - cosine_sparse(vec, centers[label])
            if dist > donor_dist:
                donor_dist = dist
                donor_idx = i
                donor_label = label
        if donor_idx < 0:
            # 所有簇都只有一个文档，无法修复，剩余空簇保留。
            return
        labels[donor_idx] = target
        centers[target] = normalize_sparse(vectors[donor_idx])
        old_members = [vectors[i] for i, label in enumerate(labels)
                       if label == donor_label]
        centers[donor_label] = average_vectors(old_members) if old_members else {}


def two_means(vectors: list[dict[str, float]]) -> tuple[list[int], list[dict[str, float]]]:
    """把一簇稳定地拆成两簇。"""
    if len(vectors) < 2:
        return [0] * len(vectors), [average_vectors(vectors)] if vectors else [{}]
    return spherical_kmeans(vectors, 2)


def topic_terms(center: dict[str, float],
                document_frequency: dict[str, int],
                document_count: int,
                top_k: int = 10) -> list[dict]:
    """从簇质心抽取关键词，分数和词项均来自成员文档。"""
    if not center:
        return []
    terms = []
    for term, weight in center.items():
        df = document_frequency.get(term, 0)
        discrimination = math.log((document_count + 1) / (df + 1)) + 1.0
        terms.append({
            "term": term,
            "score": round(weight * discrimination, 6),
            "df": df,
        })
    terms.sort(key=lambda x: (-x["score"], x["term"]))
    return terms[:top_k]


def make_label(keywords: list[dict]) -> str:
    terms = [item["term"] for item in keywords[:3]]
    if not terms:
        return "未命名话题"
    return " / ".join(terms)


def multidimensional_scaling(distances: list[list[float]],
                             dimensions: int = 2) -> list[list[float]]:
    """经典 MDS 的小规模实现，用于绘制簇间远近。"""
    n = len(distances)
    if n == 0:
        return []
    if n == 1:
        return [[0.0, 0.0]]
    # 双中心化 D2 = -0.5 * D^2
    d2 = [[distances[i][j] ** 2 for j in range(n)] for i in range(n)]
    row_mean = [sum(row) / n for row in d2]
    total_mean = sum(row_mean) / n
    matrix = [[-0.5 * (d2[i][j] - row_mean[i] - row_mean[j] + total_mean)
               for j in range(n)] for i in range(n)]

    # 小规模对称矩阵幂迭代，取前两个特征向量。
    points = [[0.0] * dimensions for _ in range(n)]
    b = [row[:] for row in matrix]
    for dim in range(dimensions):
        vector = [1.0 / math.sqrt(n)] * n
        eigenvalue = 1.0
        for _ in range(80):
            new_vector = [sum(b[i][j] * vector[j] for j in range(n))
                          for i in range(n)]
            norm = math.sqrt(sum(x * x for x in new_vector)) or 1.0
            new_vector = [x / norm for x in new_vector]
            if sum(abs(a - c) for a, c in zip(new_vector, vector)) < 1e-7:
                vector = new_vector
                break
            vector = new_vector
        ray = [sum(matrix[i][j] * vector[j] for j in range(n)) for i in range(n)]
        eigenvalue = sparse_dot({i: vector[i] for i in range(n)},
                                {i: ray[i] for i in range(n)})
        scale = math.sqrt(max(0.0, eigenvalue))
        for i in range(n):
            points[i][dim] = vector[i] * scale
        for i in range(n):
            for j in range(n):
                b[i][j] -= eigenvalue * vector[i] * vector[j]
    return points

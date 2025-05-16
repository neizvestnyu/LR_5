# Дані з умов прикладу
data = [
    ['Sunny',    'High',   'Weak',   'No'],
    ['Sunny',    'High',   'Strong', 'No'],
    ['Overcast', 'High',   'Weak',   'Yes'],
    ['Rain',     'High',   'Weak',   'Yes'],
    ['Rain',     'Normal', 'Weak',   'Yes'],
    ['Rain',     'Normal', 'Strong', 'No'],
    ['Overcast', 'Normal', 'Strong', 'Yes'],
    ['Sunny',    'High',   'Weak',   'No'],
    ['Sunny',    'Normal', 'Weak',   'Yes'],
    ['Rain',     'High',   'Weak',   'Yes'],
    ['Sunny',    'Normal', 'Strong', 'Yes'],
    ['Overcast', 'High',   'Strong', 'Yes'],
    ['Overcast', 'Normal', 'Weak',   'Yes'],
    ['Rain',     'High',   'Strong', 'No']
]

from collections import Counter
from fractions import Fraction

# Функція для обчислення апостеріорної ймовірності
def naive_bayes_predict(query, data):
    total = len(data)
    labels = [row[-1] for row in data]
    label_counts = Counter(labels)
    
    results = {}
    for label in label_counts:
        prob = Fraction(label_counts[label], total)
        for i in range(len(query)):
            attr_val = query[i]
            matching = [row for row in data if row[i] == attr_val and row[-1] == label]
            count = len(matching)
            label_total = label_counts[label]
            prob *= Fraction(count, label_total)
        results[label] = prob

    # Нормалізація
    total_prob = sum(results.values())
    normalized = {label: float(p / total_prob) for label, p in results.items()}
    return normalized

variants = [
    ['Overcast', 'High',   'Weak'],    
    ['Overcast', 'High',   'Strong'],  
    ['Sunny',    'High',   'Weak'],    
    ['Sunny',    'Normal', 'Strong'],  
    ['Rain',     'High',   'Strong']  
]

# Обрахунок і вивід результатів
for i, variant in enumerate(variants, 1):
    probs = naive_bayes_predict(variant, data)
    decision = max(probs, key=probs.get)
    print(f"Варіант {i}: {variant} =>")
    print(f"  Ймовірність Yes: {probs.get('Yes'):.4f}")
    print(f"  Ймовірність No:  {probs.get('No'):.4f}")
    print(f"  Рішення: Матч {'відбудеться' if decision == 'Yes' else 'не відбудеться'}\n")

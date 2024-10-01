import clip
import numpy as np

# vec = clip.tokenize("This is a photo of a white horse")
# vec = clip.tokenize("This is a photo of a emerald green cucumber")
# vec = clip.tokenize("This is a photo of skydiving")
# vec = clip.tokenize("This is a photo of a switch")
# vec = clip.tokenize("This is a photo of pomegranate pulp")
# vec = clip.tokenize("This is a photo of a black swan")
vec = clip.tokenize("This is a photo of a dog sitting on a green lawn")
# vec = clip.tokenize("This is a photo of a golden dog sitting on a green lawn")
# vec = clip.tokenize("This is a photo of a girl in blue clothes holding an umbrella")

tensor_str_with_comma = ','.join(map(str, vec.tolist()))
print(tensor_str_with_comma, end='')  # 同样使用end=''来防止换行

# vec_np = np.array(vec)
# print(vec)


import clip
import numpy as np

vec = clip.tokenize("This is a photo of a dog sitting on a green lawn")

tensor_str_with_comma = ','.join(map(str, vec.tolist()))
print(tensor_str_with_comma, end='')  # 同样使用end=''来防止换行



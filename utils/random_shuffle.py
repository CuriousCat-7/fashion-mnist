import random
from typing import List


def random_choice(m: int, layers: int) -> List[int]:
    choice = []
    for i in range(layers):
        choice.append(random.randint(0, m-1))
    return choice


def random_shuffle(m: int, layers: int) -> List[List[int]]:
    choice_list = [list(range(m)) for i in range(layers)]
    for l in choice_list:
        random.shuffle(l)
    choice_list = list(zip(*choice_list))
    for i in range(len(choice_list)):
        choice_list[i] = list(choice_list[i])
    return choice_list

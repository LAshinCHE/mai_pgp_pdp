#! /usr/bin/env python3
import sys 
import random


RANDOM_MIN = -1e6
RANDOM_MAX = 1e6

def generate_two_random_vector(n : int)-> tuple[list[float], list[float]]: 
    def generate_random_vector() -> list[float]:
        return [random.randint(RANDOM_MIN, RANDOM_MAX) * random.random() for _ in range(n)]
    return generate_random_vector(), generate_random_vector()

def get_answear(l1 : list[float], l2 : list[float]):
    return [min(l1[i], l2[i]) for i in range(len(l1))]


def main():
    count_element = [10, 1000, 100000]
    count_of_test = 0
    for elements in count_element:
            l1, l2 = generate_two_random_vector(elements)
            ans = get_answear(l1, l2)
            with open(f"./test/test{count_of_test}.t", "w+") as f:
                f.write(f"{elements}\n{' '.join(map(str,l1))}\n{' '.join(map(str,l1))}\n")

            with open(f"./ans/ans{count_of_test}.a", "w+") as f:
                f.writelines(" ".join(map(str, ans)))
            count_of_test += 1

if __name__ == '__main__':
    main()
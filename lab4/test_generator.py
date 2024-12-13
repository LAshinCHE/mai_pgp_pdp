import random

def main():
    filename = input("filename: ")
    n = int(input("n: "))
    with open(filename, "w") as file:
        file.write(f"{n}\n")
        for i in range(n):
            for j in range(n):
                file.write(f"{random.randint(-1000, 1000)*random.random()} ")
            file.write("\n")


main()
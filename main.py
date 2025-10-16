from database.from_bd import get_info_from_bd

def main():
    X,y = get_info_from_bd()
    for i in range(min(5, len(X))):
        print(f"=== {i + 1} ===")
        print(f"Title (y): {y[i]}")
        print(f"Synopsis (X): {X[i][:300]}...\n")  # ограничим длину вывода



if __name__ == "__main__":
    main()


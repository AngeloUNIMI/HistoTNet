def print_pers(strP, fileName):
    fileH = open(fileName, "a")
    print(strP)
    fileH.write(strP + "\n")
    fileH.close()


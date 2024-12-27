import os
import shutil
import time

BASE_DIR_TRAIN = "./trainset"
BASE_DIR_TEST = "./testset"
dirs = {
        
        "Square":f"{BASE_DIR_TRAIN}/square/",
        "Pentagon":f"{BASE_DIR_TRAIN}/pentagon/",
        "Hexagon":f"{BASE_DIR_TRAIN}/hexagon/",
        "Heptagon":f"{BASE_DIR_TRAIN}/heptagon/",
        "Octagon":f"{BASE_DIR_TRAIN}/octagon/",
        "Nonagon":f"{BASE_DIR_TRAIN}/nonagon/",
        "Circle":f"{BASE_DIR_TRAIN}/circle/",
        "Star":f"{BASE_DIR_TRAIN}/star/",
    }

dirs_test = {
        
        "Square":f"{BASE_DIR_TEST}/square/",
        "Pentagon":f"{BASE_DIR_TEST}/pentagon/",
        "Hexagon":f"{BASE_DIR_TEST}/hexagon/",
        "Heptagon":f"{BASE_DIR_TEST}/heptagon/",
        "Octagon":f"{BASE_DIR_TEST}/octagon/",
        "Nonagon":f"{BASE_DIR_TEST}/nonagon/",
        "Circle":f"{BASE_DIR_TEST}/circle/",
        "Star":f"{BASE_DIR_TEST}/star/",
    }

def sendImageToTrain(file: str, source: str, type_: str):
    destination_file = os.path.join(dirs.get(type_), file)
    os.makedirs(os.path.dirname(destination_file), exist_ok=True)
    shutil.move(source, destination_file)
    print(f"File: {file} moved to trainset")
    
def sendImageToTest(file: str, source: str, type_: str):
    destination_file = os.path.join(dirs_test.get(type_), file)
    os.makedirs(os.path.dirname(destination_file), exist_ok=True)
    shutil.move(source, destination_file)
    print(f"File: {file} mooved to testset")

def sendImages():
    all_files = [f for f in os.listdir("./output/") if os.path.isfile(os.path.join("./output/", f))]
    
    squares = [f for f in all_files if "Square" in f]
    pentagons = [f for f in all_files if "Pentagon" in f]
    hexagons = [f for f in all_files if "Hexagon" in f]
    heptagons = [f for f in all_files if "Heptagon" in f]
    octagons = [f for f in all_files if "Octagon" in f]
    nonagons = [f for f in all_files if "Nonagon" in f]
    circles = [f for f in all_files if "Circle" in f]
    stars = [f for f in all_files if "Star" in f]
    files = [squares, pentagons, hexagons, heptagons, octagons, nonagons, circles, stars]
    for files_type in files:
        #files_type = List of all files of a type
        i = 0
        for file in files_type:
            image_type = file[0:file.find("_")]
            print(i)
            if i < 2000:
                sendImageToTrain(file, os.path.join("./output/", file), image_type)
                i+=1
            elif i >= 2000 and i<=3000:
                sendImageToTest(file, os.path.join("./output/", file), image_type)
                i+=1
            else:
                break
            
start_time = time.time()
sendImages()
end_time = time.time()
exec_time = end_time - start_time
print(f"Execution time: {exec_time} seconds")
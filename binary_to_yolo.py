import numpy as np
import os                          
import math
import cv2
import random
import yaml
from tqdm import tqdm


#The nice thins about cracks is that they only have two levels. It should be relatively easy to figure out boundaries.
def externals(mask,h,w):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    segmentations = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        contour_norm = contour/[w,h]
        polyline = np.squeeze(contour_norm)
        segmentation = polyline.ravel().tolist()
        segmentations.append(segmentation)
    return segmentations

#Given two paths, return paths to properlly paired (image, mask)
#result [(img_path1, mask_path1), ....]
def find_proper_set(image_path, mask_path, root):
    result = []
    image_path = os.path.join(root, image_path)
    mask_path = os.path.join(root, mask_path)
    image_list, mask_list = os.listdir(image_path), os.listdir(mask_path)
    image_names, mask_names = set([os.path.splitext(image)[0] for image in image_list]),  \
                              set([os.path.splitext(mask)[0] for mask in mask_list])
    image_extention, mask_extension = os.path.splitext(image_list[0])[1], os.path.splitext(mask_list[0])[1]
    proper_set = image_names.intersection(mask_names)
    for root_name in proper_set:
        result.append(
            (root_name+image_extention, root_name+mask_extension)
        )
    return result
#implement all preprocessing here. Nothing for now
def preprocessing(image):
    return image

def randomShuffleSplit(input, prob, size):
    split_size = [math.floor(p*size) for p in prob]
    random.shuffle(input)
    train, val, test = input[:split_size[0]], input[split_size[0]:split_size[0]+split_size[1]], input[split_size[0]+split_size[1]:]
    return train, val, test

def create_annotations(split, names, source_folders, target_folders, prefix):
    src_image_folder, src_mask_folder, src_folder = source_folders
    src_image_folder, src_mask_folder = os.path.join(src_folder,src_image_folder), os.path.join(src_folder,src_mask_folder)
    tgt_image_folder, tgt_mask_folder = target_folders
    for image_name, mask_name in tqdm(names):
        #print(image_name)
        id = os.path.splitext(image_name)[0]
        image = cv2.imread(os.path.join(src_image_folder,image_name))
        mask = cv2.imread(os.path.join(src_mask_folder, mask_name),0)
        image = cv2.resize(image,(640,640))
        mask = cv2.resize(mask, (640,640))
        h,w = mask.shape
        bounding_polygons = externals(mask,h,w)
        annotations = [[0]+bounding_polygon for bounding_polygon in bounding_polygons]
        annotation_strings = [" ".join(map(str,annotation))+"\n" for annotation in annotations]
        cv2.imwrite(os.path.join(tgt_image_folder, prefix+'_'+image_name), preprocessing(image))
        with open(os.path.join(tgt_mask_folder,prefix+'_'+id+".txt"), 'w') as annotation_file:
            for annotation_sting in annotation_strings:
                annotation_file.write(annotation_sting)
    print(split + " is done!")

def create_yaml(root):
    if os.path.isfile(os.path.join(root, 'data.yaml')):
        print('Config file already exists')
    else:
        yaml_dict = {
        'path': root,
        'names': ['crack'],
        'nc': 1,
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images'
        }
        yaml_content = yaml.dump(yaml_dict)
        with open(os.path.join(root, 'data.yaml'),'w')as f:
            f.write(yaml_content)
        print("config file specified")
    

def create_folders(root):
    train_images = os.path.join(root,'train','images')
    train_labels = os.path.join(root,'train','labels')
    val_images = os.path.join(root,'valid','images')
    val_labels = os.path.join(root,'valid','labels')
    test_images = os.path.join(root,'test','images')
    test_labels = os.path.join(root,'test','labels')
    try:
        os.makedirs(root, exist_ok=True)
        os.makedirs(train_images, exist_ok=True)
        os.makedirs(train_labels,exist_ok=True)
        os.makedirs(val_images,exist_ok=True)
        os.makedirs(val_labels,exist_ok=True)
        os.makedirs(test_images,exist_ok=True)
        os.makedirs(test_labels,exist_ok=True)
        return (train_images, train_labels), (val_images,val_labels), (test_images,test_labels)
    except:
        print("error creating some folder.")
        exit()



#extend this code to be called on multiple datasets
def createBatch(gid, data_folder, image_folder, mask_folder, target_folder):
    print("Adding %s to the dataset." %(gid))
    data_paths = find_proper_set(image_folder, mask_folder, data_folder)
    data_size = len(data_paths)
    split_distribution = [0.6,0.2,0.2]
    train,val,test = randomShuffleSplit(data_paths,split_distribution,data_size)
    train_folders, val_folders, test_folders = create_folders(target_folder)
    create_annotations(split = "train", names = train, source_folders=(image_folder,mask_folder,data_folder), target_folders = train_folders, prefix = gid)
    create_annotations(split = "valid", names = val, source_folders=(image_folder,mask_folder,data_folder), target_folders = val_folders, prefix = gid)
    create_annotations(split = "test", names = test, source_folders=(image_folder,mask_folder,data_folder), target_folders = test_folders, prefix = gid)
    create_yaml(target_folder)

if __name__ == "__main__":
    createBatch('CrackDataset', "./public_datas/cocos", "Original_Image", "Labels", "E:/SRI_Road_Monitor/datasets")
    createBatch('EdmCrack600', "./public_datas/dataset-EdmCrack600", "images", "annotations", "E:/SRI_Road_Monitor/datasets")
    createBatch('Conglomerate_Concrete_tr', './public_datas/Conglomerate Concrete Crack Detection\Train',"images", "masks","E:/SRI_Road_Monitor/datasets")
    createBatch('Conglomerate_Concrete_te', './public_datas/Conglomerate Concrete Crack Detection\Test',"images", "masks","E:/SRI_Road_Monitor/datasets")

        


    
    



    











import glob
import csv
import cv2


csv_files = glob.glob('./outputs/*.csv')
img_files = glob.glob('./inputs/*g')
with open('./outputs/raw_predicted_results.csv', 'w') as wfile:

    writer = csv.writer(wfile)
    writer.writerow(["image", "width", "height", "username", "x", "y", "timestamp"])
    username = 'test'
    for csv_file in csv_files:
        
        print(csv_file)
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[0] != 'x':

                    image_name = csv_file.split('/')[-1][:-4] + '.png'
                    path = './inputs/{}'.format(image_name)
                    if path not in img_files:
                        image_name = image_name[:-4] + '.jpg'
                        path = path[:-4] + '.jpg'
                    if path not in img_files:
                        image_name = image_name[:-4] + '.jpg'
                        path = path[:-4] + '.jpeg'
                    if path not in img_files:
                        print('ERROR: no image found')
                        exit()
                    
                    image = cv2.imread(path)
                    width = image.shape[1]
                    height = image.shape[0]
                    
                    writer.writerow([image_name, width, height, username, 
                        float(row[0]) * width, 
                        float(row[1]) * height, row[3]])#, row[8]])
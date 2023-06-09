import csv

with open('./outputs/final_predicted_results.csv', 'w') as wfile:

    writer = csv.writer(wfile)
    writer.writerow(["image", "width", "height", "username", "x", "y", "timestamp"])

    with open('./outputs/raw_predicted_results.csv', 'r') as f:
        first_row = True
        prev_img_name = None
        i = 0
        for line in f:
            line = line[:-1]
            if first_row:
                first_row = False
                continue
            row = line.split(',')
            img_name = row[0]
            print(img_name)
            if img_name != prev_img_name:
                prev_img_name = img_name
                i = 0
            if i % 2 == 1 and i < 30:
                print(i)
                writer.writerow(row)
            i += 1

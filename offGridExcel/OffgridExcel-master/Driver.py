from DigitRecognizer import extract_digit
from roll_table_segmentation import roll_extracter
from score_table_segmentation import score_extracter
from openpyxl import load_workbook
from os import listdir
from os.path import isfile, join


wb = load_workbook('result.xlsx')
sheet = wb.worksheets[0]

list_of_roll_images = [f for f in listdir('Roll_output') if isfile(join('Roll_output', f))]
list_of_marksheet_images = [f for f in listdir('Marks_output') if isfile(join('Marks_output', f))]

for index in range(len(list_of_marksheet_images)):
    row = []
    max_roll = roll_extracter('Roll_output/'+list_of_roll_images[index])
    max_score = score_extracter('Marks_output/'+list_of_marksheet_images[index])

    print('Writing data to excel sheet....')
    roll_number = ''
    for count in range(1, max_roll+1):
        roll_number += extract_digit('Roll_output/output/Image_new'+str(count)+'.jpg')
    row.append(int(roll_number))

    mat = []
    for i in range(6):
        mat.append([0]*5)

    r, c = 5, 4
    grand_total = 0

    for count in range(max_score):
        if count == 0:
            grand_total = extract_digit('Marks_output/output/Image_new'+str(count)+'.jpg')
        else:
            temp = extract_digit('Marks_output/output/Image_new'+str(count)+'.jpg')
            if temp != '':
                mat[r][c] = temp
            r -= 1
            if r < 0:
                r = 5
                c -= 1

    # feeding marks in row variable
    for i in mat:
        for j in i:
            row.append(int(j))

    # storing grand total
    row.append(int(grand_total))

    sheet.append(row)
    wb.save('result.xlsx')
    print('Marks for roll number',roll_number, 'stored successfully!')

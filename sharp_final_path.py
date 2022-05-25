from traceback import print_tb

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

import os 
from PIL import Image
import cv2
from array import array
from IPython.display import display
import os
import cv2
import numpy as np
from PIL import Image
import sys
import pandas as pd
import time
import glob
import argparse


def cnvrt_to_jpeg(path_,name):

    if path_ is not None:
        img = Image.open(path_)
        out = img.convert("RGB")
        outfile = str(name).split('.')[0] + '.jpg'
        out.save('images/'+outfile, "JPEG", quality=500)

def save_images(img_path,coordinates,save_path):
    for i in os.listdir(img_path):
        img = Image.open(os.path.join(img_path,i))
        img = img.resize((700,900))
        [x, y, w, h] = coordinates
        img = img.crop((x, y, x+w, y+h))
        img.save(save_path+'/'+i)

def save_cropped_images():
    
    img_path = 'images'
    coordinates = [[16,761,195,115],[199,759,230,119],[419,755,264,136],[16, 592, 665, 177]]
    save_path = ['bottom_first_box_images_1','bottom_second_box_images_1','bottom_third_box_images','bottom_fourth_box_images']#,'bottom_third_box_images_1']

    for i,j in zip(coordinates,save_path):
        list(map(save_images,[img_path],[i],[j]))

def IOU(l1,l2):
    # [x1,y1,w1,h1] = [55, 71, 213, 82]
    # [x2,y2,w2,h2 ]= [54, 69, 161, 81]
    [x1,y1,w1,h1] = l1
    [x2,y2,w2,h2 ]= l2

    x_left1 = x1
    y_top1 = y1
    x_right1 = w1
    y_bottom1 = h1

    x_left2 = x2
    y_top2 = y2
    x_right2 = w2
    y_bottom2 = h2

    boxA = [x_left1,y_top1,x_right1,y_bottom1]
    boxB = [x_left2,y_top2,x_right2,y_bottom2] 

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # '''compute the area of intersection rectangle'''
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # '''
    # compute the area of both the prediction and ground-truth
    # rectangles
    # '''
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # ''' 
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area'''
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def extract(img_path,computervision_client):
    text_lst = []
    bound_lst = []
    read_response = computervision_client.read_in_stream(open(img_path,"rb"),language='en',  raw=True)
    read_operation_location = read_response.headers["Operation-Location"]
    operation_id = read_operation_location.split("/")[-1]
    while True:
        read_result = computervision_client.get_read_result(operation_id)
        if read_result.status not in ['notStarted', 'running']:
            break
        time.sleep(1)
    if read_result.status == OperationStatusCodes.succeeded:
            for text_result in read_result.analyze_result.read_results:
                for line in text_result.lines:
                    text_lst.append(line.text)
                    bound_lst.append(line.bounding_box)

    return text_lst,bound_lst

def y_coordinate_based(text_lst,bound_lst):
    y_coordinates = []
    for coordinates in bound_lst:
        if coordinates[0] in range(10-10,10+10):
            idx = bound_lst.index(coordinates)
            y_coordinates.append(coordinates[1])

    field = []
    data = []
    for y_coordinate in y_coordinates:
        field = []
        for bound_coordinate in bound_lst:
            if y_coordinate in range(int(bound_coordinate[1]-5),int(bound_coordinate[1]+10)):
                idx = bound_lst.index(bound_coordinate)
                field.append(text_lst[idx])
        data.append(field)
    table_data = pd.DataFrame(data)
    return table_data

def table_1(ref_data,computervision_client):

    ref_boxes = ref_data.iloc[:,1:].values
    ref_field = ref_data.iloc[:,0].values
    for path in os.listdir('images'):
        img_path = os.path.join('images',path)
        text_lst,bound_lst =  extract(img_path,computervision_client)
        time.sleep(1)

        info = []
        iou_score_lst = []
        iou_column_score=[]

        for i in ref_boxes:
            iou_score_lst = []
            for j in bound_lst:
                [x,y,w,h]=[j[0]-5,j[1]-5,j[2]+10,j[7]+10]
                iou_score = IOU(i,[x,y,w,h])
                iou_score_lst.append(iou_score)
            if max(iou_score_lst)>0:
                idx= np.argmax(iou_score_lst)
                info.append(text_lst[idx])
            else:
                info.append('Null')
            iou_column_score.append(max(iou_score_lst))

        dct = {'fields':ref_field,'info':info}#,'IOU_score':iou_column_score}
        df1=pd.DataFrame(dct)
        po_box_lst = df1.iloc[0:2,1].values
        po_box_lst = [i for i in po_box_lst if 'PO BOX' in i ]
        try:
            po_box_lst = po_box_lst[0].split(' ')[2]
            df1 = df1.drop(labels=0)
            df1.iloc[0,1] = po_box_lst
        except:
            df1.iloc[0,1] = 'null'
        df1.to_csv('csv_files/1/{}.csv'.format(path[:-4]),index=False)

def table_2(computervision_client):


    for path in os.listdir('bottom_first_box_images_1'):
        img_path = os.path.join('bottom_first_box_images_1',path)
        text_lst,bound_lst =  extract(img_path,computervision_client)
        time.sleep(1)
        new_lst = [i for i in text_lst if i!='x' and i!='X' and len(i)>3]
        try:
            dct = {'fields':['Federal Tax id and SSN/EIN','Physician Name and Signature '],'info':[new_lst[0],new_lst[1]]}
        except:
            dct = {'fields':['null','null'],'info':['null','null']}
            pass
        df2=pd.DataFrame(dct)
        df2.to_csv('csv_files/2/{}.csv'.format(path[:-4]),index=False)

def table_3(computervision_client):

        for path in os.listdir('bottom_second_box_images_1'):
            img_path = os.path.join('bottom_second_box_images_1',path)
            text_lst,bound_lst =  extract(img_path,computervision_client)
            time.sleep(1)
            info = []
            iou_score_lst = []
            iou_column_score=[]
            i=[9.0, 17.0, 86.0, 28.0]
            for j in bound_lst:
                    [x,y,w,h]=[j[0]-5,j[1]-5,j[2]+10,j[7]+10]
                    iou_score = IOU(i,[x,y,w,h])
                    iou_score_lst.append(iou_score)
            if max(iou_score_lst)>0:
                idx= np.argmax(iou_score_lst)
                info.append(text_lst[idx])
            else:
                info.append('Null')
                iou_column_score.append(max(iou_score_lst))

            new_lst = [i for i in text_lst if i!='x' and i!='X' and i!='x' and len(i)>3]
            idx = new_lst.index(info[0])
            try:
                dct = {'fields':['Patients Account No','Facility Address(m1)','m2','m3','Facility NPIID'],'info':[new_lst[0],new_lst[1],new_lst[2],new_lst[3],new_lst[4]]}
            except:
                pass

            df3=pd.DataFrame(dct)
            df3.to_csv('csv_files/3/{}.csv'.format(path[:-4]),index=False)
def table_4(image_path_4,computervision_client):
    for i in os.listdir(image_path_4):
        path = os.path.join(image_path_4,i)
        text_lst,bound_lst =  extract(path,computervision_client)
        table = y_coordinate_based(text_lst,bound_lst)
        data = pd.DataFrame(table)
        return data

def concat_tables(computervision_client):
    a=[i for i in os.listdir('csv_files/1')]
    # '''
    # the variable 'a' contains all file names, by using this we can fetch csv data which are in different folders using the common name
    # '''
    for i in a:
        df_lst =[]
        for j in range(1,4):
            data = pd.read_csv('csv_files/'+str(j)+'/'+i)
            df_lst.append(data)
        final_data = pd.concat(df_lst,axis=0)
        final_data = final_data.reset_index(drop=True)

        final_data.to_csv('csv_files/final_csv_files/'+i[:-4]+'.csv',index=False)

        img_path_4 = 'bottom_fourth_box_images'
        table_4_data = table_4(img_path_4,computervision_client)

        files = glob.glob('final_csv/*')
        [os.remove(f) for f in files]

        writer = pd.ExcelWriter('final_csv/'+i[:-4]+'.xlsx', engine = 'xlsxwriter')
        final_data.to_excel(writer, sheet_name = 'data')
        table_4_data.to_excel(writer, sheet_name = 'table_data')
        writer.save()
        writer.close()
        return '/fina_csv'
    # with open('final_csv/'+i[:-4]+'.xlsx', 'rb') as f:
    #     st.download_button('Download File', f, file_name='final_csv/'+i[:-4]+'.xlsx') 




def main():
    subscription_key = "4d2ccba6557b4eb88e28e49260a473d2"
    endpoint = "https://allianzeformautomation.cognitiveservices.azure.com/"
    computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))
    # html_temp = """
    # <div style="background-color:tomato;padding:10px">
    # <h2 style="color:white;text-align:center;">Data Extraction </h2>
    # </div>
    # """
    # html_temp1 = """
    # <div style="background-color:DarkViolet;padding:0px">
    # <h2 style="color:white;text-align:center;">Upload Image </h2>
    # </div>
    # """


    # st.header(html_temp,'Data Extraction')

    # st.subheader("Upload Image")
    # image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg","tif"])
    image_path = '22012400229.tif'             
    name = image_path.split('/')[-1] 

    if image_path:
        # print(str(image_file.name).split('.')[0])
        # st.write(image_file.name)
        cnvrt_to_jpeg(image_path,name)
        # '''
        # convert tif to jpeg
        # '''
    
        save_cropped_images()
        # '''
        # saving cropped images in to the seperate folders
        # '''

        ref_data = pd.read_csv('ref_coordinates.csv')
        table_1(ref_data,computervision_client)
        # ''' 
        # creating table 1 which contains fields:
        # PO box,PO box,insured’s id number,patients name,patients birth date,insured’s name,
        # patients address,insured’s address,city,state,state,insured’s city,insured’s state,insured’s state,pincode,telephone number,
        # insured’s pincode,insured’s telephone number,other insured’s name,other insured’s policy number,insured’s date of birth,program name ,insurance plan name,
        # Patient's or authorized persons signature,Patients Signature Date,Insured's or authorized persons signature,Date of current illness,injury, or Pregnancy (LMP),
        # Date of current illness,injury, or Pregnancy (LMP),Referring PhysicianName Qual and Referring PhysicianName or other name,Referring NPIID,
        # Hospitalization Date,Diagnosis Codes,Prior authorization number
        # '''

        table_2(computervision_client)
        # '''
        # creating table 2 , fields : Federal Tax id and SSN/EIN,Physician Name and Signature 
        # '''

        table_3(computervision_client)
        # '''
        # creating table 3 , fields : Federal Tax id and SSN/EIN, Physician Name and Signature 
        # '''

        # '''
        # table data
        # '''
        img_path_4 = 'bottom_fourth_box_images'
        table_4(img_path_4,computervision_client)

        final_path = concat_tables(computervision_client)
        print(final_path)
        # '''
        # here we concat all tables which have been saved in different folders
        # '''


if __name__ == "__main__":

    	main()

		

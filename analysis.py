import math, cv2, csv, pathlib, datetime
import matplotlib.pyplot as plt
import numpy as np
path=str(pathlib.Path(__file__).parent.resolve())+"\data"
photos=['img_0.jpeg', 'img_1.jpeg', 'img_2.jpeg', 'img_3.jpeg', 'img_4.jpeg', 'img_5.jpeg', 'img_6.jpeg', 'img_7.jpeg', 'img_8.jpeg', 'img_9.jpeg', 'img_10.jpeg', 'img_11.jpeg', 'img_12.jpeg', 'img_13.jpeg', 'img_14.jpeg', 'img_15.jpeg', 'img_16.jpeg', 'img_17.jpeg', 'img_18.jpeg', 'img_19.jpeg', 'img_20.jpeg', 'img_21.jpeg', 'img_22.jpeg', 'img_23.jpeg', 'img_24.jpeg', 'img_25.jpeg', 'img_26.jpeg', 'img_27.jpeg', 'img_28.jpeg', 'img_29.jpeg', 'img_30.jpeg', 'img_31.jpeg', 'img_32.jpeg', 'img_33.jpeg', 'img_34.jpeg', 'img_35.jpeg', 'img_36.jpeg', 'img_37.jpeg', 'img_38.jpeg', 'img_39.jpeg', 'img_40.jpeg', 'img_41.jpeg', 'img_42.jpeg', 'img_43.jpeg', 'img_44.jpeg', 'img_45.jpeg', 'img_46.jpeg', 'img_47.jpeg', 'img_48.jpeg', 'img_49.jpeg', 'img_50.jpeg', 'img_51.jpeg', 'img_52.jpeg', 'img_53.jpeg', 'img_54.jpeg', 'img_55.jpeg', 'img_56.jpeg', 'img_57.jpeg', 'img_58.jpeg', 'img_59.jpeg', 'img_60.jpeg', 'img_61.jpeg', 'img_109.jpeg', 'img_110.jpeg', 'img_111.jpeg', 'img_112.jpeg', 'img_113.jpeg', 'img_114.jpeg', 'img_115.jpeg', 'img_116.jpeg', 'img_117.jpeg', 'img_118.jpeg', 'img_119.jpeg', 'img_120.jpeg', 'img_121.jpeg', 'img_122.jpeg', 'img_123.jpeg', 'img_124.jpeg', 'img_125.jpeg', 'img_126.jpeg', 'img_127.jpeg', 'img_128.jpeg', 'img_129.jpeg', 'img_130.jpeg', 'img_131.jpeg', 'img_132.jpeg', 'img_133.jpeg', 'img_134.jpeg', 'img_135.jpeg', 'img_136.jpeg', 'img_137.jpeg', 'img_138.jpeg', 'img_139.jpeg', 'img_140.jpeg', 'img_141.jpeg', 'img_142.jpeg', 'img_143.jpeg', 'img_144.jpeg', 'img_145.jpeg']
file=path+"\data.csv"
acc_f=path+"\\acceleration.csv"
data=list()
acc=list()
with open(file, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        data.append(row)
data=data[1::]
with open(acc_f, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        acc.append(row)
acc=acc[284::]
final=path+"\\final.csv"
bf=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
orb=cv2.ORB_create()
rt=6378
lx=math.atan(0.6287)
ly=math.atan(0.471525)
def save_data(file, mode, *args):
    with open(file, mode, buffering=1, newline="") as f:
        file=csv.writer(f)
        file.writerow(args)
def rot_yaw(photo, yaw):
    photo=cv2.imread(photo)
    w, h=int(photo.shape[1]), int(photo.shape[0])
    photo=cv2.resize(photo, (w,h), interpolation = cv2.INTER_AREA)
    m=cv2.getRotationMatrix2D((w/2, h/2), yaw, 1)
    photo=cv2.warpAffine(photo, m, (w,h))
    return photo
def find_matches(ph1, ph2, i):
    kp1, d1=orb.detectAndCompute(ph1, None)
    kp2, d2=orb.detectAndCompute(ph2, None)
    matches=bf.knnMatch(d1, d2, 2)
    good_matches = []
    for m1, m2 in matches:
        if m1.distance < 0.7 * m2.distance:
            good_matches.append(m1)
    #match_img=cv2.drawMatches(ph1, kp1, ph2, kp2, good_matches, None)
    #resize = cv2.resize(match_img, (1600,600), interpolation = cv2.INTER_AREA)
    #cv2.imshow('matches', resize)
    #cv2.waitKey(0)
    c1=[]
    c2=[]
    for match in good_matches:
        image_1_idx=match.queryIdx
        image_2_idx=match.trainIdx
        (x1,y1)=kp1[image_1_idx].pt
        (x2,y2)=kp2[image_2_idx].pt
        c1.append((x1-1296,y1-972))
        c2.append((x2-1296,y2-972))
    return c1, c2
def calc_gsc(coor, r):
    h=rt+float(r[7])-0.0225
    dx=1296/(math.tan(lx))
    dy=972/(math.tan(ly))
    points=[]
    for co in coor:
        tx=co[0]/dx
        ty=co[1]/dy
        a=tx**2+ty**2+1
        b=-2*h
        c=h**2-rt**2
        k1=(-b+(b**2-4*a*c)**.5)/(2*a)
        k2=(-b-(b**2-4*a*c)**.5)/(2*a)
        if k1>h:
            p=(k2*tx, k2*ty, h-k2)
        else:
            p=(k1*tx, k1*ty, h-k1)
        points.append(p)
    return(points)
def calc_doe(p1, p2):
    distance=0
    alt=0
    for i in range(len(p1)):
        d=((p1[i][0]-p2[i][0])**2+(p1[i][1]-p2[i][1])**2+(p1[i][2]-p2[i][2])**2)**.5
        angle=math.asin(d/(2*rt))
        d=2*angle*rt
        al=math.tanh((p1[i][1]-p2[i][1])/(p1[i][0]-p2[i][0]))
        distance+=d
        alt+=al
    return distance/len(p1), alt/len(p1)
def distance_coor(r1, r2):
    angle=math.acos(math.sin(math.radians(float(r1[6]))) * math.sin(math.radians(float(r2[6]))) + math.cos(math.radians(float(r1[6]))) * math.cos(math.radians(float(r2[6]))) * math.cos(math.radians(float(r2[5])) - math.radians(float(r2[5]))))
    distance=angle*rt
    return distance
def sup_vel(v1, angle, acc):
    v1x=v1*math.cos(angle)
    v1y=v1*math.sin(angle)
    pred=[0]
    for row in range(len(acc)-1):
        time=datetime.datetime.strptime(acc[row+1][0], "%Y-%m-%d %H:%M:%S.%f")-datetime.datetime.strptime(acc[row][0], "%Y-%m-%d %H:%M:%S.%f")
        time=time.seconds+time.microseconds/1000000
        v1x=v1x+float(acc[row][1])*0.00981*time
        v1y=v1y+float(acc[row][2])*0.00981*time
        vf=(v1x**2+v1y**2)**.5
        if acc[row][4]=="yes":
            pred.append(vf)
    return(pred)
l_v_img=[]
l_v_coor=[]
l_pred=[]
for i in range(len(photos)-1):
    ph1, ph2=photos[i], photos[i+1]
    for i in range(len(data)-1):
        r1, r2=data[i], data[i+1]
        if r1[1]=="/home/sandbox/msl/Pistein/data"+"/"+ph1 and r2[1]=="/home/sandbox/msl/Pistein/data"+"/"+ph2:
            ph1=rot_yaw(path+"\\"+ph1, float(r1[4]))
            ph2=rot_yaw(path+"\\"+ph2, float(r2[4]))
            c1, c2=find_matches(ph1, ph2, i)
            p1, p2=calc_gsc(c1, r1), calc_gsc(c2, r2)
            time=datetime.datetime.strptime(r2[0], "%Y-%m-%d %H:%M:%S.%f")-datetime.datetime.strptime(r1[0], "%Y-%m-%d %H:%M:%S.%f")
            time=time.seconds+time.microseconds/1000000
            distance_img, angle=calc_doe(p1, p2)
            velocity_img=distance_img/time
            velocity_coor=distance_coor(r1, r2)/time
            if i==0:
                pred=sup_vel(velocity_img, angle, acc)
            save_data(final, "a", i, velocity_img, velocity_coor, pred[i])
            l_v_img.append(velocity_img)
            l_v_coor.append(velocity_coor)
            l_pred.append(pred[i])
            break
#v_img=np.array(l_v_img[0:61])
#v_coor=np.array(l_v_coor[0:61])
#v_pred=np.array(l_pred[0:61])
#plt.subplot(2,2,1)
#plt.plot(v_img)
#plt.plot(v_coor)
#plt.plot(v_pred)
#plt.title('1st Series of Images (Yucatan)')
#plt.ylabel('Velocity')
#v_img=np.array(l_v_img[61::])
#v_coor=np.array(l_v_coor[61::])
#v_pred=np.array(l_pred[61::])
#plt.subplot(2,2,2)
#plt.plot(v_img)
#plt.plot(v_coor)
#plt.plot(v_pred)
#plt.title('2nd Series of Images (Sakhalin Oblast)')
#plt.ylabel('Velocity')
#plt.show()
print("code finished")

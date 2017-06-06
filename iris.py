import numpy as np
import matplotlib.pyplot as plt
import sys
import numpy.linalg as lin
from sympy import Symbol, solve



def Boundary(C_inv,iris_class,x):
    d=np.dot(np.dot(np.transpose(x),(-1/2)*C_inv),x)+np.dot(np.dot(C_inv,iris_class),x)-(1/2)*(np.dot(np.dot(np.transpose(iris_class),C_inv),iris_class))-(1/2)*np.log(lin.det(lin.inv(C_inv)))

    return d
def Count_data(data,type):
    count=0
    for i in range (0,len(data)):
        if data[i][4]==type :
            count+=1
    return count

def iris_mean(data,type,number):
    count=0
    sepal_length=0
    sepal_width=0
    setal_length=0
    setal_width=0

    number=len(data)
    for i in range(0,number) :


        if data[i][4]==type :
            count+=1
            sepal_length+=data[i][0]
            sepal_width+=data[i][1]
            setal_length+=data[i][2]
            setal_width+=data[i][3]


    return (sepal_length/count,sepal_width/count,setal_length/count,setal_width/count)


def covariance(a, b):

    if len(a) != len(b):
        return

    a_mean = np.mean(a)
    b_mean = np.mean(b)


    sum = 0

    for i in range(0, len(a)):
        sum += ((a[i] - a_mean) * (b[i] - b_mean))

    return sum/(len(a)-1)


def MakeConfusionMatrix(data,max):
    con_matrix=np.array([[0,0,0],[0,0,0],[0,0,0]])
    for i in range(0,max):

        if data[i][4]==1:
            x=np.array([[data[i][0]],[data[i][1]],[data[i][2]],[data[i][3]]])
            if Boundary(c1_cov_inv,iris_class1,x)-Boundary(c2_cov_inv,iris_class2,x)>0 :
                if Boundary(c3_cov_inv,iris_class3,x)-Boundary(c1_cov_inv,iris_class1,x)<0 :
                    con_matrix[0][0]=con_matrix[0][0]+1
                else:
                    con_matrix[0][2]=con_matrix[0][2]+1
            elif Boundary(c1_cov_inv,iris_class1,x)-Boundary(c2_cov_inv,iris_class2,x)<0 :
                if Boundary(c2_cov_inv,iris_class2,x)-Boundary(c3_cov_inv,iris_class3,x)>0 :
                    con_matrix[0][1]=con_matrix[0][1]+1
                else:
                    con_matrix[0][2]=con_matrix[0][2]+1
        if data[i][4]==2:
            x=np.array([[data[i][0]],[data[i][1]],[data[i][2]],[data[i][3]]])
            if Boundary(c1_cov_inv,iris_class1,x)-Boundary(c2_cov_inv,iris_class2,x)>0 :
                if Boundary(c3_cov_inv,iris_class3,x)-Boundary(c1_cov_inv,iris_class1,x)<0 :
                    con_matrix[1][0]=con_matrix[1][0]+1
                else:
                    con_matrix[1][2]=con_matrix[1][2]+1
            elif Boundary(c1_cov_inv,iris_class1,x)-Boundary(c2_cov_inv,iris_class2,x)<0 :
                if Boundary(c2_cov_inv,iris_class2,x)-Boundary(c3_cov_inv,iris_class3,x)>0 :
                    con_matrix[1][1]=con_matrix[1][1]+1
                else:
                    con_matrix[1][2]=con_matrix[1][2]+1
        if data[i][4]==3:
            x=np.array([[data[i][0]],[data[i][1]],[data[i][2]],[data[i][3]]])
            if Boundary(c1_cov_inv,iris_class1,x)-Boundary(c2_cov_inv,iris_class2,x)>0 :
                if Boundary(c3_cov_inv,iris_class3,x)-Boundary(c1_cov_inv,iris_class1,x)<0 :
                    con_matrix[2][0]=con_matrix[2][0]+1
                else:
                    con_matrix[2][2]=con_matrix[2][2]+1
            elif Boundary(c1_cov_inv,iris_class1,x)-Boundary(c2_cov_inv,iris_class2,x)<0 :
                if Boundary(c2_cov_inv,iris_class2,x)-Boundary(c3_cov_inv,iris_class3,x)>0 :
                    con_matrix[2][1]=con_matrix[2][1]+1
                else:
                    con_matrix[2][2]=con_matrix[2][2]+1
        #print(con_matrix)
      #  print('-------------',i)
    return  con_matrix


if __name__ == "__main__":
    data=np.loadtxt('Iris_train.txt')
    n=len(data)
    x=[[Symbol('x1')],[Symbol('x2')],[Symbol('x3')],[Symbol('x4')]]

    c1_count=Count_data(data,1)
    c2_count=c1_count+Count_data(data,2)
    c3_count=c2_count+Count_data(data,3)


#같은 특징끼리 묶기
    c1_x1=[data[i][0] for i in range(0,c1_count)]
    c1_x2=[data[i][1] for i in range(0,c1_count)]
    c1_x3=[data[i][2] for i in range(0,c1_count)]
    c1_x4=[data[i][3] for i in range(0,c1_count)]

    c2_x1=[data[c1_count][0] for c1_count in range(c1_count,c2_count)]
    c2_x2=[data[c1_count][1] for c1_count in range(c1_count,c2_count)]
    c2_x3=[data[c1_count][2] for c1_count in range(c1_count,c2_count)]
    c2_x4=[data[c1_count][3] for c1_count in range(c1_count,c2_count)]

    c3_x1=[data[c2_count][0] for c2_count in range(c2_count,c3_count)]
    c3_x2=[data[c2_count][1] for c2_count in range(c2_count,c3_count)]
    c3_x3=[data[c2_count][2] for c2_count in range(c2_count,c3_count)]
    c3_x4=[data[c2_count][3] for c2_count in range(c2_count,c3_count)]



    iris_class1=iris_mean(data,1,n)
    iris_class2=iris_mean(data,2,n)
    iris_class3=iris_mean(data,3,n)

    #공분산 4X4 (함수로 만들기)
    c1_cov=np.array([[covariance(c1_x1,c1_x1),covariance(c1_x1,c1_x2),covariance(c1_x1,c1_x3),covariance(c1_x1,c1_x4)],
            [covariance(c1_x2,c1_x1),covariance(c1_x2,c1_x2),covariance(c1_x2,c1_x3),covariance(c1_x2,c1_x4)],
            [covariance(c1_x3,c1_x1),covariance(c1_x3,c1_x2),covariance(c1_x3,c1_x3),covariance(c1_x3,c1_x4)],
            [covariance(c1_x4,c1_x1),covariance(c1_x4,c1_x2),covariance(c1_x4,c1_x3),covariance(c1_x4,c1_x4)]])
    c2_cov=np.array([[covariance(c2_x1,c2_x1),covariance(c2_x1,c2_x2),covariance(c2_x1,c2_x3),covariance(c2_x1,c2_x4)],
            [covariance(c2_x2,c2_x1),covariance(c2_x2,c2_x2),covariance(c2_x2,c2_x3),covariance(c2_x2,c2_x4)],
            [covariance(c2_x3,c2_x1),covariance(c2_x3,c2_x2),covariance(c2_x3,c2_x3),covariance(c2_x3,c2_x4)],
            [covariance(c2_x4,c2_x1),covariance(c2_x4,c2_x2),covariance(c2_x4,c2_x3),covariance(c2_x4,c2_x4)]])
    c3_cov=np.array([[covariance(c3_x1,c3_x1),covariance(c3_x1,c3_x2),covariance(c3_x1,c3_x3),covariance(c3_x1,c3_x4)],
            [covariance(c3_x2,c3_x1),covariance(c3_x2,c3_x2),covariance(c3_x2,c3_x3),covariance(c3_x2,c3_x4)],
            [covariance(c3_x3,c3_x1),covariance(c3_x3,c3_x2),covariance(c3_x3,c3_x3),covariance(c3_x3,c3_x4)],
            [covariance(c3_x4,c3_x1),covariance(c3_x4,c3_x2),covariance(c3_x4,c3_x3),covariance(c3_x4,c3_x4)]])
#c_cov역행렬

    c1_cov_inv=lin.inv(c1_cov)
    c2_cov_inv=lin.inv(c2_cov)
    c3_cov_inv=lin.inv(c3_cov)

    # d1=Boundary(c1_cov_inv,iris_class1,x)
    # d2=Boundary(c2_cov_inv,iris_class2,x)
    # d3=Boundary(c3_cov_inv,iris_class3,x)
    #
    # #print("class1 mean vector : ",iris_class1)
    # #print("class1 covariance: ",c1_cov)
    # print("g1(x)=",d1)
    # #print("class2 mean vector : ",iris_class2)
    # #print("class2 covariance: ",c2_cov)
    # print("g2(x)=",d2)
    # #print("class3 mean vector : ",iris_class3)
    # #print("class3 covariance: ",c3_cov)
    # print("g1(x)=",d3)
    #
    # boundary1_2=d1-d2
    # boundary3_1=d3-d1
    # boundary2_3=d2-d3
    #
    # #print(boundary1_2)
    # #print(boundary3_1)
    # #print(boundary2_3)
    #
    #

    # print("g1-g2: ",Boundary(c1_cov,iris_class1,x)-Boundary(c2_cov,iris_class2,x))
    # print("g3-g1: ",Boundary(c3_cov,iris_class3,x)-Boundary(c1_cov,iris_class1,x))
    # print("g2-g3: ",Boundary(c2_cov,iris_class2,x)-Boundary(c3_cov,iris_class3,x))


    data=np.loadtxt('iris_test.txt')

    a=len(data)
    print(MakeConfusionMatrix(data,a))
    data=np.loadtxt("Iris_train.txt")
    a=len(data)
    print(MakeConfusionMatrix(data,a))

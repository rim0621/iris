import numpy as np
import matplotlib.pyplot as plt
import sys
import numpy.linalg as lin
import sympy as syp
from scipy.optimize import fsolve
from sympy import Symbol, solve
from sympy import *
from sympy import sympify
from sympy import MatrixSymbol

def iris_mean(data,type,number):
    count=0
    sepal_length=0
    sepal_width=0


    number=len(data)
    for i in range(0,number) :


        if data[i][4]==type :
            count+=1
            sepal_length+=data[i][0]
            sepal_width+=data[i][1]


    return (sepal_length/count,sepal_width/count)


def Count_data(data,type):
    count=0
    for i in range (0,len(data)):
        if data[i][4]==type :
            count+=1
    return count

def findIntersection(fun1, fun2, x0):
    return fsolve(lambda x: fun1(x) - fun2(x), x0)

def covariance(a, b):

    if len(a) != len(b):
        return

    a_mean = np.mean(a)
    b_mean = np.mean(b)


    sum = 0

    for i in range(0, len(a)):
        sum += ((a[i] - a_mean) * (b[i] - b_mean))

    return sum/(len(a)-1)


def Boundary(C_inv,iris_class,x):

    d=np.dot(np.dot(np.transpose(x),(-1/2)*C_inv),x)+np.dot(np.dot(C_inv,iris_class),x)-(1/2)*(np.dot(np.dot(np.transpose(iris_class),C_inv),iris_class))-(1/2)*np.log(lin.det(lin.inv(C_inv)))
    return d

def MakeConfusionMatrix(data,max):

    con_matrix=np.array([[0,0,0],[0,0,0],[0,0,0]])
    for i in range(0,max):

        if data[i][4]==1:
            x=np.array([[data[i][0]],[data[i][1]]])
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
            x=np.array([[data[i][0]],[data[i][1]]])
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
            x=np.array([[data[i][0]],[data[i][1]]])
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

    return  con_matrix

def mahal(C_inv,iris_class,x):

    x1, x2=symbols('x1, x2')

    MD=lambda x1,x2 :np.dot(np.dot(np.transpose([x1-iris_class[0],x2-iris_class[1]]),C_inv),[x1-iris_class[0],x2-iris_class[1]])-4

    print("mahal= ",MD(x1,x2))
    sol=solve(MD(x1,x2))

    return sol

if __name__ == "__main__":
    data=np.loadtxt('Iris_train.txt')
    n=len(data)
    x=[[Symbol('x1')],[Symbol('x2')]]

    #print(type(x2))

    c1_count=Count_data(data,1)
    c2_count=c1_count+Count_data(data,2)
    c3_count=c2_count+Count_data(data,3)

#같은 특징끼리 묶기
    c1_x1=[data[i][0] for i in range(0,c1_count)]
    c1_x2=[data[i][1] for i in range(0,c1_count)]

    c2_x1=[data[c1_count][0] for c1_count in range(c1_count,c2_count)]
    c2_x2=[data[c1_count][1] for c1_count in range(c1_count,c2_count)]

    c3_x1=[data[c2_count][0] for c2_count in range(c2_count,c3_count)]
    c3_x2=[data[c2_count][1] for c2_count in range(c2_count,c3_count)]

    plt.scatter(c1_x2,c1_x1,c='blue',marker='s')
    plt.scatter(c2_x2,c2_x1,c='green',marker='*')
    plt.scatter(c3_x2,c3_x1,c='red',marker='x')


    #평균벡터
    iris_class1=iris_mean(data,1,n)
    iris_class2=iris_mean(data,2,n)
    iris_class3=iris_mean(data,3,n)




    #공분산
    c1_cov=np.array([[covariance(c1_x1,c1_x1),covariance(c1_x1,c1_x2)],
            [covariance(c1_x2,c1_x1),covariance(c1_x2,c1_x2)]])
    c2_cov=np.array([[covariance(c2_x1,c2_x1),covariance(c2_x1,c2_x2)],
            [covariance(c2_x2,c2_x1),covariance(c2_x2,c2_x2)]])
    c3_cov=np.array([[covariance(c3_x1,c3_x1),covariance(c3_x1,c3_x2)],
            [covariance(c3_x2,c3_x1),covariance(c3_x2,c3_x2)]])


    c1_cov_inv=lin.inv(c1_cov)
    c2_cov_inv=lin.inv(c2_cov)
    c3_cov_inv=lin.inv(c3_cov)

    x1, x2=symbols('x1, x2')
    x2=np.linspace(2.0,4.5,10000)
    c1_mahal=mahal(c1_cov_inv,iris_class1,x)
    c2_mahal=mahal(c2_cov_inv,iris_class2,x)
    c3_mahal=mahal(c3_cov_inv,iris_class3,x)

    #print(c1_mahal[0])
    #print(c2_mahal[0])
    #print(c3_mahal[0])

    c1_mahala_1=0.685178374402354*x2 - 1.53695740326855e-17*np.sqrt(-1.65909418611427e+33*pow(x2,2) + 1.13399087620911e+34*x2 - 1.83360513067514e+34) + 2.65590290547996
    c1_mahala_2=0.685178374402354*x2 + 1.53695740326855e-17*np.sqrt(-1.65909418611427e+33*pow(x2,2) + 1.13399087620911e+34*x2 - 1.83360513067514e+34) + 2.65590290547996
    c2_mahala_1=0.86400351667674*x2 - 4.83307831791264e-17*np.sqrt(-7.09436014984561e+32*pow(x2,2) + 3.94091706323924e+33*x2 - 5.14189662191442e+33) + 3.59023023243035
    c2_mahala_2=0.86400351667674*x2 + 4.83307831791264e-17*np.sqrt(-7.09436014984561e+32*pow(x2,2) + 3.94091706323924e+33*x2 - 5.14189662191442e+33) + 3.59023023243035
    c3_mahala_1=0.899568034557236*x2 - 1.82546657805837e-17*np.sqrt(-1.15358536759251e+34*pow(x2,2) + 6.85229708349953e+34*x2 - 9.73741704575963e+34) + 3.93828293736502
    c3_mahala_2=0.899568034557236*x2 + 1.82546657805837e-17*np.sqrt(-1.15358536759251e+34*pow(x2,2) + 6.85229708349953e+34*x2 - 9.73741704575963e+34) + 3.93828293736502
    #평균벡터 그래프
    plt.scatter(iris_class1[1],iris_class1[0],edgecolors='m')
    plt.scatter(iris_class2[1],iris_class2[0],edgecolors='m')
    plt.scatter(iris_class3[1],iris_class3[0],edgecolors='m')

    #mahalanobis distanceon graph
    plt.plot(x2,c1_mahala_1,c='b')
    plt.plot(x2,c1_mahala_2,c='b')
    plt.plot(x2,c2_mahala_1,c='green')
    plt.plot(x2,c2_mahala_2,c='green')
    plt.plot(x2,c3_mahala_1,c='red')
    plt.plot(x2,c3_mahala_2,c='red')
    #Boundary on graph
    b1=Boundary(c1_cov_inv,iris_class1,x)
    b2=Boundary(c2_cov_inv,iris_class2,x)
    b3=Boundary(c3_cov_inv,iris_class3,x)

    x1, x2=symbols('x1, x2')
    b12=b1-b2
    b23=b2-b1
    b31=b3-b1

    x2=np.linspace(2.0,4.5,100)

    b12_solve= 0.60179350065938*x2 + 1.80290364186342e-15*np.sqrt(6.7656216363665e+28*pow(x2,2) - 4.18768179856701e+28*x2 - 2.37289842932876e+28) + 2.22023281344768
    b23_solve= 0.823976523083762*x2 + 4.10903582439455e-15*np.sqrt(4.78122295994377e+28*pow(x2,2) - 3.59747847024408e+29*x2 + 6.78603325781983e+29) + 3.19850543983648
    b31_solve=0.641767112874829*x2 + 1.47853785434607e-15*np.sqrt(1.45605013543265e+29*pow(x2,2) - 5.80295177041999e+29*x2 + 8.49433916112419e+29) + 2.39623676812231
    plt.plot(x2,b12_solve,c='blue')
    plt.plot(x2,b23_solve,c='red')
    plt.plot(x2,b31_solve,c='green')

    plt.show()


    data=np.loadtxt('iris_test.txt')

    a=len(data)
    print(np.transpose(MakeConfusionMatrix(data,a)))
    c1_count=Count_data(data,1)
    c2_count=c1_count+Count_data(data,2)
    c3_count=c2_count+Count_data(data,3)

#같은 특징끼리 묶기
    c1_x1=[data[i][0] for i in range(0,c1_count)]
    c1_x2=[data[i][1] for i in range(0,c1_count)]

    c2_x1=[data[c1_count][0] for c1_count in range(c1_count,c2_count)]
    c2_x2=[data[c1_count][1] for c1_count in range(c1_count,c2_count)]

    c3_x1=[data[c2_count][0] for c2_count in range(c2_count,c3_count)]
    c3_x2=[data[c2_count][1] for c2_count in range(c2_count,c3_count)]

    plt.scatter(c1_x2,c1_x1,c='blue')
    plt.scatter(c2_x2,c2_x1,c='green')
    plt.scatter(c3_x2,c3_x1,c='red')
    plt.plot(x2,b12_solve,c='blue')
    plt.plot(x2,b23_solve,c='red')
    plt.plot(x2,b31_solve,c='green')
    plt.show()



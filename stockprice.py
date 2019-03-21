import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn import datasets,linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from
# from sklearn.metrics import mean_squared_error,r2_score
FIRST=False
SECOND=True

txtfile = 'msft_stockprices_dataset.csv'

class StockDailyInfo:
    def __init__(self,day,month,year,highPrice,lowPrice,openPrice,closePrice,volume):
        self.day=day
        self.month = month
        self.year = year
        self.highPrice = highPrice
        self.lowPrice = lowPrice
        self.openPrice = openPrice
        self.closePrice = closePrice
        self.volume = volume

    def printInfo(self):
        print(str(self.day)+"/"+str(self.month)+"/"+str(self.year),self.highPrice,self.lowPrice,self.openPrice,self.closePrice,self.volume)

def readFromCSV(txtfile):
    f = open(txtfile,'r')
    line = f.readlines()
    stock=[]
    for i in range(len(line)):
        if (i%2==1):
            text = line[i].split(',');
            for j in range(len(text)):
                if j==0:
                    date=text[j].split(' ')[0]
                    year=str(date).split('-')[0]
                    month=str(date).split('-')[1]
                    day=str(date).split('-')[2]
                elif j==1:
                    highPrice = eval(text[j])
                elif j==2:
                    lowPrice = eval(text[j])
                elif j==3:
                    openPrice = eval(text[j])
                elif j==4:
                    closePrice = eval(text[j])
                elif j==5:
                    volume = eval(text[j])
            stock.append(StockDailyInfo(day,month,year,highPrice,lowPrice,openPrice,closePrice,volume))
    return stock

if __name__ == "__main__":
    stock=readFromCSV(txtfile)
    length = len(stock)
    if FIRST:
        X=[]
        y=[]
        for i in range(length):
            X.append((int)(stock[i].openPrice))
            y.append(stock[i].closePrice)
        X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=int(time.time()))
        X_train=np.array(X_train)
        X_train = X_train.reshape(-1, 1)
        X_test=np.array(X_test)
        X_test = X_test.reshape(-1, 1)
        regr = linear_model.LinearRegression()
        regr.fit(X_train, y_train)
        disabetes_y_pred = regr.predict(X_test)
        within_cnt=0
        for i in range(len(X_test)):
            percentage=(disabetes_y_pred[i]-y_test[i])/y_test[i]
            if(percentage<0.01):
                within_cnt+=1
        print("the number that within the request",within_cnt/len(X_test))

        plt.scatter(X_test,y_test,color='black')
        plt.plot(X_test,disabetes_y_pred,color='blue',linewidth=3)
        plt.xticks(())
        plt.yticks(())
        plt.show()

    if SECOND:
        trainLength = length*4//5
        X_total = np.arange(length)
        X_value = X_total[:trainLength]
        X_test = X_total[trainLength:]
        X_total1 = np.array(X_total).reshape([length,1])
        closePriceList=np.zeros(trainLength,np.float16)
        test = np.zeros(length-trainLength,np.float16)
        for i in range(trainLength):
            closePriceList[i]=(stock[i].closePrice)
        for i in range(trainLength,length):
            test[i-trainLength]=stock[i].closePrice

        X_value1 = np.array(X_value).reshape([trainLength,1])
        closePriceList1 = np.array(closePriceList).reshape([trainLength,1])
        poly_reg = PolynomialFeatures(degree=2)
        X_ploy=poly_reg.fit_transform(X_value1)
        lin_reg_2 = linear_model.LinearRegression()
        lin_reg_2.fit(X_ploy,closePriceList1)
        X_test1 = np.array(X_test).reshape([length-trainLength,1])
        plt.figure(1)
        plt.plot(X_test1,(X_test1-lin_reg_2.predict(poly_reg.fit_transform(X_test1)))/X_test1)
        plt.figure(2)

        plt.plot(X_value,closePriceList,color='red')
        plt.plot(X_test,test,color='yellow')
        plt.plot(X_total,lin_reg_2.predict(poly_reg.fit_transform(X_total1)),color='blue')
        plt.show()
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 03:46:24 2021

@author: CORE i7
"""

#startimplementation
import pandas as pd
import sys
import csv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sqlite3
from pathlib import Path
from sqlalchemy import create_engine
import sklearn.metrics as metrics
from sklearn.metrics import mean_squared_error,r2_score


        
class createfr:

    # This class enables us to create a table in an sql3 lite database, 
   


    ''' We create a database and make connection to the database''' 
    
    conn = sqlite3.connect('tew3.db')        
    c = conn.cursor()
    engine = create_engine('sqlite:///tew3.db')
    
    def __init__(self, a,b,tblname):
        
        ''' 
        The variables of this class include, 
        1. The number of colums of our table in a range (a,b)  where a =1 always, 
        2. b is the last column number 
        3.tblname, is the name of our table that we want to create and is a string literal, e.g 'Ideal 
        
        '''

        self.a = a
        self.b =b
        self.tblname = tblname

        
    
    def idgener(self,a,b):
        
        '''
        generate a list of column names for our Ideal and training tables
        return: list of column names of our desired table 
        
        '''
        try:
            self.seq_y = []
            self.seq_y.append('x float')
            for i in range(a,b):
                
                x= 'y'+str(i) + ' float'
                self.seq_y.append(x)
            return  self.seq_y  
        
        except Exception as e:
            
            '''
             catches exceptions
            
            '''
            print(' Error on line {}' .format(sys.exc_info()[-1].tb_lineno),type(e).__name__,e)
    
            
            
    
    def idgtest(self,a,b):
        
      '''
      generate a x and y column names for our test table
      return: list of column names of our desired table 
        
      '''
      try:
          self.seq_y = []
          self.seq_y.append('x float')
                        
          x= 'y float'
          self.seq_y.append(x)
          return self.seq_y
      
      except Exception as e:
          print(' Error on line {}' .format(sys.exc_info()[-1].tb_lineno),type(e).__name__,e)
        

    
    def crtbl(self,a,b):
        
        ''' 
        
        try :  making connection to  the database and create a new table having columns
        from the list of column names
        
        except : catch exceptions in trying to connect to the data base
        
        '''
        try :
            
            Path('tew3.db').touch()
            
            self.conn = sqlite3.connect('tew3.db')        
            self.c = self.conn.cursor()
            seq_y = self.idgener(a,b)
            
            
            self.c.execute("CREATE TABLE IF NOT EXISTS tblname (%s)"%",".join(seq_y))
             
            print ("successfully created the "+self.tblname+" table.")  
           
        
        except Exception as e:
            print(' Error on line {}' .format(sys.exc_info()[-1].tb_lineno),type(e).__name__,e)
        

        
        
    def padtf(self,ccsv):
        
        ''' 
        
        create a pandas dataframe from csv and write it to the sq3lite table, 
        variable ccsv would be the string literal name of the csv e.g. 'train'  for the train.csv file
        
        '''
        
        try:
        
            
            '''
            create an ideal dataframe for ideal.csv file
            
            '''
            
            if ccsv == 'ideal' :
                
                try:
                    
                    '''
                    
                    create ideal dataframe and notify that its created
                    
                    '''
                    
                    tblname = pd.read_csv('ideal.csv')
                    
                    print('pandas dataframe for ideal functions created')
                    
              
                except  :
                    
                    '''
                    raise exception found in trying to read ideal csv
                    
                    '''
                    
                    print( ' we encounted an error in reading your Ideal csv file')
                    
                else:
                    
                    ''' 
                    
                    try: write data to sql database and notify if it has successfully done so
                    
                    except: raise all kinds of exceptions that could be found in trying to make the connection to database
                    
                    '''
                
                    try: 
                        
                        tblname.to_sql('ideal', self.conn, if_exists = 'append', index = False)
                        print("Successfully written data to the sql ideal table")
                    
                    except Exception as e:
                        print ('Having trouble with writing data to ideal sql table')
                        print(' Error on line {}' .format(sys.exc_info()[-1].tb_lineno),type(e).__name__,e)
                    
            elif ccsv == 'train':
                
               
                
                try:
                    
 
                    '''
                    
                    create train dataframe and notify that its created
                    
                    '''
 
                    
                    
                    tblname = pd.read_csv('train.csv')
                    print('pandas dataframe for train data created')
                    
                except:


                    '''
                    raise exception found in trying to read train.csv file
                    
                    '''


                    print( ' we encounted an error in reading your Train csv file')
                    
                else:
                    
                    
                    ''' 
                    
                    try: write data to sql train table and notify if it has successfully done so
                    
                    except: raise all kinds of exceptions that could be found in trying to make the connection to database
                    
                    '''
                    
                
                    try: 
                        tblname.to_sql('train', self.conn, if_exists = 'append', index = False)
                        print("Successfully written data to the sql train table")
                    
                    except Exception as e :
                        
                        print(' Error on line {}' .format(sys.exc_info()[-1].tb_lineno),type(e).__name__,e)
  
                    
                        print ('Having trouble with writing data to train sql table')
                      
                    
                     
            elif ccsv == 'test':
                
                
                
                try:
                    
                    '''
                    
                    create test dataframe and notify that its created
                    
                    '''
                    
                    tblname = pd.read_csv('test.csv')
                    print('pandas dataframe for test data created')
            
                except :
                    
                    '''
                    
                    raise exception found in trying to read test.csv file
                    
                    ''' 
                    
                    
                    
                    print( ' we encounted an error in reading your test csv file')
                    
                else:
                    
                    
                    ''' 
                    
                    try: write data to sql test table and notify if it has successfully done so
                    
                    except: raise all kinds of exceptions that could be found in trying to make the connection to database
                    
                    '''
                    
                    
                
                    try: 
                        tblname.to_sql('test', self.conn, if_exists = 'append', index = False)
                        print("Successfully written data to the sql test table")
                    
                    except Exception as e :
                        
                        print ('Having trouble with writing data to test sql table','\n')
                        
                        print(' Error on line {}' .format(sys.exc_info()[-1].tb_lineno),type(e).__name__,e)
  
                      
                   
        except:
            
            
            '''
            
            raise exception if the file selected file does not match any of the three csv files
            
            '''
            print(' There is no match to the csv files to be compiled by this program')
        
        
        else:
            
           
            
            ''' 
            
            Reading data from sql table for interaction
            
            '''
            
            if ccsv == 'ideal':
                '''
                
                try: creating a pandas dataframe that reads/loads data from ideal table
                return: return the created pandas dataframe for use in other program operations
                except: Raise exception when there's problem with loading the sql ideal table'
                
                '''
                
                try:
                    self.idfn = pd.read_sql_table("ideal", con=self.engine)
                    print(" Actively reading from Ideal table ")
                    return self.idfn
                
                
                except Exception as e:
                     
                    print(" Error in reading data from sql Ideal table  ",'\n')
                    
                    print(' Error on line {}' .format(sys.exc_info()[-1].tb_lineno),type(e).__name__,e)
  
                
            elif ccsv == 'test': 
                
                '''
                
                try: creating a pandas dataframe that reads/loads data from test table
                return: return the created pandas dataframe for use in other program operations
                except: Raise exception when there's problem with loading the sql test table'
                
                '''
                
                try:
                    self.tst = pd.read_sql_table("test", con=self.engine)
                    print("Actively reading from test sql table")
                    return self.tst  
                except:
                     
                    print(" Error in reading data from sql Test table  ")
 
                
            elif ccsv == 'train':
                
                '''
                
                try: creating a pandas dataframe that reads/loads data from train table
                return: return the created pandas dataframe for use in other program operations
                except: Raise exception when there's problem with loading the sql train table'
                
                '''
                
        
                try:
                    self.trn = pd.read_sql_table("train", con=self.engine)
                    print("Actively reading the train sql table ")
                    return self.trn
                
                except Exception as e:
                     
                    print(" Error in reading data from sql Ideal table  ")
                    
                    print(' Error on line {}' .format(sys.exc_info()[-1].tb_lineno),type(e).__name__,e)
  
            
           
              
    def selectIDb(self):
        
        ''''
        
        Reads data from ideal table row by row
        
        '''
        self.c.execute("SELECT *FROM ideal")
        
        rows = self.c.fetchall()
        
        for row in rows:
            print(row)
            
    def selectTs(self):
        
        ''''
        
        Reads data from test table row by row
        
        '''
        self.c.execute("SELECT *FROM test")
        
        rows = self.c.fetchall()
        
        for row in rows:
            print(row)
            
    def selectTrn(self):
        
        ''''
        
        Reads data from train table row by row
        
        '''
        self.c.execute("SELECT *FROM train")
        
        rows = self.c.fetchall()
        
        for row in rows:
            print(row)

class SelectBestFit:
    
    ''' 
    
    This class enables us to Select the 4 Function from 50 in the Ideal dataset 
    that best Matches our 4 training sets
    
    
    class inheritance from the createfr class enables us to connect to database and dataframes for
    all 3 datasets train, ideal and test.
        
    '''
        
    pt = createfr(1,51,tblname='ideal') 
    pt.idgener(1,51)
    pt.crtbl(1,51)
    idw = pt.padtf(ccsv='ideal')
    
    
    hts = createfr(1, 2, tblname='test')
    hts.idgtest(1, 2)
    hts.crtbl(1, 2)
    fr = hts.padtf(ccsv='test')
  
    htr = createfr(1, 5, tblname= 'train' )
    htr.idgener(1,5)
    htr.crtbl(1, 5)
    bg = htr.padtf(ccsv='train')
    
    def Ideal_datafr(self):
        
        '''
        
        returns the ideal dataframe instance idw
        
        '''
        return self.idw
    
    def test_df(self):
        
        '''
        
        returns the test dataframe instance fr
        
        '''        
        return self.fr
    
    def train_df(self):
        
        '''
        
        returns the train dataframe instance bg
        
        '''
        return self.bg
    
    
    def Trai_Ide(self): 
       
        '''
        
        Traincolumns creates a list of column names to be used in report creation
        
        try:
            
        creates an empty  csv file  (file.csv) 
        
        iterates through each column of the train dataframe and makes iteration in the ideal dataframe computing mean square
        errors between that train function and 50 ideal functions
        
                
        Only the mean square error of 1.41 is acceptable as the criterion for selection and are written in the csv file creating
        a Report      
        
        The unacceptable are simply printed for observation in console
        
        Open csv file to fin out which fuctions among the 50 are the best fit i.e those with the list mean square error
        
        '''
        
        TrainColumns = ["x","y1","y2","y3","y4"]
        i = -1
        try: 
            
            '''
            
            
            NB: ensure to enter a correct path for the csv file
            
            '''
            
            writer = csv.writer(open("C:/Users/CORE i7/Documents/IUBH/Programming with Python/Assignement/Program1/file.csv", 'w'))

            for train in self.bg:
        
                i=i+1
                
      
                
                for column in self.idw:
                    
                    '''
                    
                    computes the mean square between train each train dataframe and 50 other ideal functions
                    
                    
                    '''
                    try:
                        
                        Mean_Squared_Error = metrics.mean_squared_error(self.bg[train],self.idw[column],
                                                            sample_weight=None,multioutput='uniform_average',squared=True)
                        
                        Rse= r2_score(self.bg[train],self.idw[column])
                        '''
                        prints the report in console of all the iterations made
                        
                        '''
        
                        print('Train',TrainColumns[i], ': Ideal  :' , column, "\n\n" )
                        print('MSE: ', 
                                          Mean_Squared_Error, "\n\n")
                        print('R2: ', Rse) 
                        print('RMSE', np.sqrt(Mean_Squared_Error))
                    
                    except Exception as e:
                        
                        '''
                        
                        Handles the exception when program fails to compute the mean square error
                        
                        '''
                        print(' Hint: Ensure you have not duplicated the columns in the train and ideal tables, othwerise create new database and run again the program')
                        print(' Error on line {}' .format(sys.exc_info()[-1].tb_lineno),type(e).__name__,e)
  


                    else:   
                        
                        '''
                        
                        program continues to report creation
                        
                        '''
                        
                        if Mean_Squared_Error < 2:
                            
                            '''
                            
                            writes to csv creating a report
                            
                            '''
                            writer.writerow(["MSE  ",Mean_Squared_Error,
                                             'Train Funct ',TrainColumns[i],
                                             " Chosen Ideal Function",column,
                                             "RMSE",np.sqrt(Mean_Squared_Error),
                                             "R2", Rse])
                            
                
                        else:
                            
                            '''
                            
                            prints report of the unacceptable  values
                            
                            '''
                            
                            print('Train',TrainColumns[i], ': Ideal  :' , column, "\n\n" )
                            print('MSE: ', 
                                              Mean_Squared_Error, "\n\n")
                            print('R2: ', Rse) 
                            print('RMSE', np.sqrt(Mean_Squared_Error))

        except Exception as e:
            
            '''
            
            Handles the exceptions found when the program failes to write to csv 
            
            correct path for the csv file must be entered
            
            '''
            
            print("HINT: Check that the path and file name for your csv file is correct")
            print(' Error on line {}' .format(sys.exc_info()[-1].tb_lineno),type(e).__name__,e)
  



         
        

class FitModel:
        
    '''
    
    This class defines various helper methods for creating dataframes to be used when fitting models on different functions
    
    Inherites from SelectBestFit class for usage of the dataframe instances 
    
    '''
    
    
    brl = SelectBestFit()
    id_df = brl.Ideal_datafr()
    
    fw = SelectBestFit()
    tr_df = fw.train_df()
    
    n = SelectBestFit()
    ts_df = n.test_df()
    

    
    def idealall(self):
        
         
        '''
        
        returns ideal dataframe 
        
        '''
          
        return   self.id_df
    
    def trainall(self):
        
         
        '''
        
        returns train dataframe 
        
        '''
          
        return   self.tr_df
        
    def idealx(self):
         
        '''
        
        creates a seperate pandas column of the ideal x column 
        
        '''
          
        return   self.id_df.loc[:,['x']]

    def ideal44(self):
         
         
        '''
        
        creates a seperate pandas column of the ideal y44 column 
        
        '''
            
        return    self.id_df.loc[:,['y44']]
    
    def trainx(self):
         
        '''
        
        creates a seperate pandas column of the train x column 
        
        '''
         
        return   self.tr_df.loc[:,['x']] 

    def trainy1(self):
         
         
        '''
        
        creates a seperate pandas column of the train y1 column 
        
        '''
            
        return   self.tr_df.loc[:,['y1']]


    def ideal41(self):
         
         
        '''
        
        creates a seperate pandas column of the ideal y41 column 
        
        '''
            
        return    self.id_df.loc[:,['y41']]
    

    def trainy2(self):
         
         
        '''
        
        creates a seperate pandas column of the train y2 column 
        
        '''
          
        return   self.tr_df.loc[:,['y2']]

    def ideal34(self):
         
         
        '''
        
        creates a seperate pandas column of the ideal y34 column 
        
        '''
          
        return    self.id_df.loc[:,['y34']]
    

    def trainy3(self):
         
                 
        '''
        
        creates a seperate pandas column of the train y3 column 
        
        '''
                  
        return   self.tr_df.loc[:,['y3']]


    def ideal21(self):
         
         
        '''
        
        creates a seperate pandas column of the ideal y21 column 
        
        '''
          
        return    self.id_df.loc[:,['y21']]
    
    def ideal48(self):
         
         
        '''
        
        creates a seperate pandas column of the ideal y48 column 
        
        '''
          
        return    self.id_df.loc[:,['y48']]
    
    def ideal50(self):
         
         
        '''
        
        creates a seperate pandas column of the ideal y50 column 
        
        '''
          
        return    self.id_df.loc[:,['y50']]
    
    def ideal1(self):
         
         
        '''
        
        creates a seperate pandas column of the ideal y1 column 
        
        '''
          
        return    self.id_df.loc[:,['y1']]

    def trainy4(self):
         
                 
        '''
        
        creates a seperate pandas column of the train y4 column 
        
        '''
          
        return   self.tr_df.loc[:,['y4']]

    def tstds(self):
         
         
        '''
        
        creates pandas dataframe for the test set
        
        '''
          
        return self.ts_df
        
                     
      
     
class Ppln: 
    
    '''
    
     Makes Scatter plots
    Inherites dataframes from FitModel and SelectBestFit classes and their helper functions  
    
    '''

    bw = FitModel()
    
    '''
    x ideal function
    
    '''
    ikx =bw.idealx()
    
    '''
    y 21 ideal function
    
    '''
    ik21 = bw.ideal21()
    
    '''
    y 44 ideal function
    
    '''
    ik44 = bw.ideal44()
    
    '''
    y 41 ideal function
    
    '''
    ik41 = bw.ideal41()
    
    '''
    y 48 ideal function
    
    '''
    ik48 = bw.ideal48()
    
    '''
    y 50 ideal function
    
    '''
    ik50 = bw.ideal50()
    
    '''
    y 1 ideal function
    
    '''
    ik1 = bw.ideal1()
    
    '''
    y 34 ideal function
    
    '''
    ik34 = bw.ideal34()
    
    '''
    ideal dataframe
    
    '''
    ikall = bw.idealall()
    
    '''
    test data in a pandas dataframe
    
    '''
    iktest = bw.tstds()
    
    tw1 = bw.trainy1()
    tw2 = bw.trainy2()
    tw3 = bw.trainy3()
    tw4 = bw.trainy4()
    twx = bw.trainx()
    twall = bw.trainall()
    
    fw = SelectBestFit()
    tr_df = fw.train_df()
    
    n = SelectBestFit()
    ts_df = n.test_df()                 
    
    
    

     
    def __init__(self,x_ideal,y_ideal,x_train,y_train):
        
        '''
        
        Class variables for the x,y pair of the ideal and train functions needs to be defined
        
        '''
        
        self.x_ideal = x_ideal
        self.y_ideal = y_ideal
        self.x_train = x_train
        self.y_train = y_train
        
    def PloVis(self):
       
      
           
                try:
                    
                    if self.y_ideal.equals(self.ik21):
                        
                        '''
                        
                        title plot for y4 train 
                        
                        '''
                                             
                        plt.title('y4 train - Scatter Plot ')
                        
                        '''
                        x label 
                        
                        '''
                        
                        plt.xlabel('x train data')
                        
                    
                        
                        '''
                        y label
                        
                        ''' 
                        
                        plt.ylabel('y4 train data')
                        
                        '''
                         scatter plots  of train data
                        
                        '''
                        plt.style.use('ggplot')
                        sns.scatterplot(x ='x',y= 'y4' , data = self.twall , label='y4 train')
                        plt.plot(self.x_ideal,self.y_ideal,'g--',label='y21 ideal')
                        plt.legend()   
                        plt.show()
                        
                    elif self.y_ideal.equals(self.ik44):
                        
                        '''
                        
                        title plot for y1 train 
                        
                        '''
                                            
                    
                        plt.title(' y1 train - Scatter Plot ', pad = 20)
                        
                        '''
                        x label 
                        
                        '''
                        
                        plt.xlabel('x train data')
                        
                    
                        
                        '''
                        y label
                        
                        ''' 
                        
                        plt.ylabel('y1 train data')
                        
                        '''
                        
                         scatter plots  of train data 
                         
                        '''
                        plt.style.use('ggplot')                        
                        plt.plot(self.x_ideal,self.y_ideal, color = 'blue', label = 'y44 ideal')
                        sns.scatterplot(x='x',y= 'y1', color ='red', data = self.twall, label = 'y1 train')
                        plt.legend()
                        plt.show()
                       
                     
                    elif self.y_ideal.equals(self.ik41):
                        
                        '''
                        
                        title plot for y2 train 
                        
                        '''
                  
                    
                  
                        plt.title('y2 train - Scatter Plot ')
                        
                        
                        '''
                        x label 
                        
                        '''
                        
                        plt.xlabel('x train data')
                        
                    
                        
                        '''
                        y label
                        
                        ''' 
                        
                        plt.ylabel('y2 train data')
                        
                        '''
                        
                         scatter plots  of train data 
                         
                        '''
                        
                        sns.scatterplot(x='x', y = 'y2', color ='red', label ='y2 train', data = self.twall )
                        plt.plot(self.x_ideal, self.y_ideal, color = 'blue', label = 'y41 ideal')
                        plt.legend(loc='upper right') 
                        plt.show()
                        
                    elif self.y_ideal.equals(self.ik34):
                        
                        '''
                        
                        title plot for y3 train 
                        
                        '''
                  
                    
                  
                        plt.title('y3 train - Scatter Plot ')
                        
                        
                        '''
                        x label 
                        
                        '''
                        
                        plt.xlabel('x train data')
                        
                    
                        
                        '''
                        y label
                        
                        ''' 
                        
                        plt.ylabel('y3 train data')
                        
                        '''
                        
                         scatter plots  of train data 
                         
                        '''
                        sns.scatterplot(x='x', y ='y3', data = self.twall, label = 'y3 train')
                        plt.plot(self.x_ideal, self.y_ideal, color ='blue', label = 'y34')
                        plt.legend()     
                        plt.show()
                        
                except Exception as e:
                    print(' Error on line {}' .format(sys.exc_info()[-1].tb_lineno),type(e).__name__,e)
            
        
    

                
    def MetricsP(self):
        
        
        y1diff = self.ikall['y44'] - self.twall['y1']
        y2diff = self.ikall['y41'] - self.twall['y2']
        y3diff = self.ikall['y34'] - self.twall['y3']
        y4diff = self.ikall['y21'] - self.twall['y4']
        
        
        fr = [y1diff,y2diff,y3diff,y4diff]
        
        for nw1 in list(fr):
        
            sns.distplot(nw1)
            plt.axvline(x = np.mean(nw1), color = 'red', label = 'mean')
            plt.axvline(x =np.median(nw1), color = 'orange', label = 'median')
            plt.xlabel('Residuals')
            plt.legend(loc='upper right')
            plt.show()
            
                
        
  
     

   
                

def main():
    
    '''
    create ideal table and populate it with data 
    
    '''
    ph = createfr(1,51,tblname='ideal')
    ph.idgener(1,51)
    ph.crtbl(1,51)
    ph.padtf(ccsv='ideal')
   
    '''
    create test table and populate it with data 
    
    '''
    
    htest = createfr(1, 2, tblname='test')
    htest.idgtest(1, 2)
    htest.crtbl(1, 2)
    htest.padtf(ccsv='test')
    
    '''
    create train table and populate it with data 
    
    '''
    
    htrain = createfr(1, 5, tblname= 'train' )
    htrain.idgener(1,5)
    htrain.crtbl(1, 5)
    htrain.padtf(ccsv='train')
    
    '''
    Select from only 4 best fit functions from the given 50 ideal functions
    csv file is created - check to see which functions are the best fit  
    
    '''
    
    bf = SelectBestFit()
    bf.Trai_Ide()
    
    
    '''
    create x,y pairs of data from chosen ideal functions and training data 
    
    '''
    cs = FitModel()
    y44 = cs.ideal44()
    xidl = cs.idealx()
    xtrs = cs.trainx()
    ytr1 = cs.trainy1()
    ytr2 = cs.trainy2()
    ytr3 = cs.trainy3()
    ytr4 = cs.trainy4()
    y21 = cs.ideal21()
    y34 = cs.ideal34()
    y41 = cs.ideal41()
    
      
    
    '''
    class instance for Ppln to fit y44 ideal for making some analysis
    Plots y44 ideal  and y1 train
    
    '''
    y44cg = Ppln(xidl, y44, xtrs, ytr1)
    y44cg.PloVis()
        
    '''
    class instance for Ppln to fit y41 ideal for making some analysis
    PLots y41 ideal and y2 train
    
    '''
    y41cg = Ppln(xidl, y41, xtrs, ytr2)
    y41cg.PloVis()
  
    
    '''
    
    class instance for Ppln to fit y34 ideal for making some analysis
    Plots y34 ideal and y3 train
    
    '''
    y34cg = Ppln(xidl, y34, xtrs, ytr3)
    y34cg.PloVis()
  
    
    '''
    
    class instance for Ppln to fit y21 ideal for making some analysis
    Plots y21 ideal and y4 train
    
    '''
    y21cg = Ppln(xidl, y21, xtrs, ytr4)
    y21cg.PloVis()
        
    y44cg.MetricsP()
    
    
    

if __name__ == "__main__":
    main()
        

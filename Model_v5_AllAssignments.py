
# coding: utf-8

# In[1]:

from gurobipy import *
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import shapefile as shp
from collections import defaultdict
from datetime import datetime

# ### Extract Data

# In[3]:
starttime = time.time() 
#Get Settlement List
#change to use different settlement files (updates console and solution log files)
#Sfile = "prov_code_15"
Sfile = "NewData"
#Sfile = "afg_ppl_settlement_pnt_0s_removed"
print(Sfile)

#%%
#Get Settlement List
Settlements=pd.read_csv("%s.csv"%Sfile,sep=",")
Settlements = Settlements[["OBJECTID","POPULATION", "LAT_Y","LON_X"]]
S = Settlements.shape[0]
Settlement_List = Settlements['OBJECTID'].tolist()

#Get District Court List
Districts = pd.read_csv("District_Courts.csv",sep=",")
Districts = Districts[["DIST_CODE","POP","LON_X", "LAT_Y"]]
D = Districts.shape[0]
District_List = Districts['DIST_CODE'].tolist()

#Get Appeals Court List
Appeals = pd.read_csv("Appeals_Courts.csv",sep=",")
Appeals = Appeals[["PROV_CODE","POP","LON_X", "LAT_Y"]]
A = Appeals.shape[0]
Appeals_List = Appeals['PROV_CODE'].tolist()

#Get Data Reduction List
Reduced = pd.read_csv("Merged_Settlements.csv",sep=",")
R = Reduced.shape[0]
Reduced_List = Reduced['OBJECTID'].tolist()

All_S = pd.read_csv("afg_ppl_settlement_pnt.csv",sep=",")
All_S = All_S[["OBJECTID","POPULATION", "LAT_Y","LON_X"]]
Total = Reduced.shape[0]
All_S_List = All_S['OBJECTID'].tolist()

#Create Dictionaries
Settlement_Dict = Settlements.set_index('OBJECTID').T.to_dict('list')
District_Dict = Districts.set_index('DIST_CODE').T.to_dict('list')
Appeals_Dict = Appeals.set_index('PROV_CODE').T.to_dict('list')
Reduced_Dict = Reduced.set_index('OBJECTID').T.to_dict('list')
All_Settlement_Dict = All_S.set_index('OBJECTID').T.to_dict('list')

endtime = time.time()
print("read data time %.2f"%(endtime-starttime))
#%%
#plot district courts
#for d in District_List:
#    d_lon = District_Dict[d][1]
#    d_lat = District_Dict[d][2]
#    plt.scatter(d_lon,d_lat, color='Red', marker = 's',s=25, zorder= 2)
#
##plot appeals courts    
#for a in Appeals_List:
#    a_lon = Appeals_Dict[a][1]
#    a_lat = Appeals_Dict[a][2]
#    plt.scatter(a_lon, a_lat, color='Green', marker = '^',s = 25, zorder= 2)
#
#for i in All_S_List:          
#    s_lon = All_Settlement_Dict[i][2]
#    s_lat = All_Settlement_Dict[i][1]
#    plt.scatter(s_lon, s_lat, color='Blue', marker = 'o',s = 25)
# In[ ]:

#Set Parameters
#maxOpenD = 40
#maxOpenA = 4

pD=0.8
pA=0.8
maxOpenD = round(pD*D) #want number in provinces used....not it
maxOpenA = round(pA*A) # want number in provinces used...this is not it

maxDist_D = 210 #km (1 degree is about 70km)
maxDist_A = 350 #km (1 degree is about 70km)

#maxPopD- max capacity of each district court
#maxPopA - max capacity of each appeals court

#langS
#langD
#langA

# In[ ]:

starttime = time.time() 

#Create Dictionaries for District Courthouse Distances
Dist_D = {}

#Create Dictionaries for Appeals Courthouse Distances
Dist_A = {}

R = 6371e3

for s in Settlement_List:
    s_lon = Settlement_Dict[s][2]
    s_lat = Settlement_Dict[s][1]
    phi1 = np.radians(s_lat)
    
    for d in District_List:
        d_lon = District_Dict[d][1]
        d_lat = District_Dict[d][2]
        phi2 = np.radians(d_lat)
 
        #Distance to District Court
        delta_phi = np.radians(d_lat - s_lat)
        delta_lambda = np.radians(d_lon - s_lon)
        a = np.sin(delta_phi/2) * np.sin(delta_phi/2) + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2) * np.sin(delta_lambda/2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        Dist_D[s,d] = (R * c)/1000
        
    for a in Appeals_List:
        a_lon = Appeals_Dict[a][1]
        a_lat = Appeals_Dict[a][2]
        phi3 = np.radians(a_lat)

        #Distance to District Court
        delta_phi= np.radians(a_lat - s_lat)
        delta_lambda = np.radians(a_lon - s_lon)
        a1 = np.sin(delta_phi/2) * np.sin(delta_phi/2) + np.cos(phi1) * np.cos(phi3) * np.sin(delta_lambda/2) * np.sin(delta_lambda/2)
        c = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1-a1))
        
        Dist_A[s,a] = (R * c)/1000

endtime = time.time()
print("distance dictionary time %.2f"%(endtime-starttime))   
# In[ ]:
    
#Create Dictionaries for populations
s_pop = {}
totalPop = 0 

for s in Settlement_List:
    s_pop[s] = Settlement_Dict[s][0] #population of settlement
    totalPop = totalPop + s_pop[s] #total population of settlements
    
#CAPACITY OF COURTHOUSES DETERMINED BY CURRENT POPULATION IN PROVINCE/DISTRICT
d_pop = {}
increase_capD = 1 + ((1-pD)/D)
increase_capA = 1 + ((1-pA)/A)

for d in District_List:
    #d_pop[d] = District_Dict[d][0]*1.25
    d_pop[d] = District_Dict[d][0]*increase_capD
    #d_pop[d] = 30000000/350;
    
a_pop = {}
for a in Appeals_List:
    #a_pop[a] = Appeals_Dict[a][0]*1.25
    a_pop[a] = Appeals_Dict[a][0]*increase_capA
    #a_pop[a] = 30000000/30;


# ### Create Model

# In[ ]:

#Create Model
IP = Model("Afg_IP")

#IP.Params.OutputFlag = 0
IP.params.LazyConstraints = 1
IP.Params.LogFile="log_IP_" + Sfile + "_" + str(int(time.time())) +".log"
IP.Params.MIPFocus = 1 
IP.Params.MIPGap = .01
#IP.Params.ImproveStartTime = 600

 ##### Helper function
# In[ ]:

# Return value of variable
def VarVal(var):
    if (type(var) == gurobipy.Var):
        val = round(var.X)
    else:
        val = 0
    return val

# In[ ]:
starttime = time.time() 

# Create d_i_j variables
d_IP = {}

# Create a_i_k variables
a_IP = {}

for i in Settlement_List:
    d_IP[i] = {}
    a_IP[i] = {}
    
    for j in District_List:
        if Dist_D[i,j] < maxDist_D:
            d_IP[i][j] = IP.addVar(vtype=GRB.BINARY, name='d_IP')
        else:
            d_IP[i][j] = 0

    for k in Appeals_List:
        if Dist_A[i,k] < maxDist_A:
            a_IP[i][k] = IP.addVar(vtype=GRB.BINARY, name='a_IP')
        else:
            a_IP[i][k] = 0
            
IP.update()

# Create c_j_k variables
c_IP = {}
for j in District_List:
    c_IP[j] = {}
    for k in Appeals_List:
        c_IP[j][k] = IP.addVar(vtype=GRB.BINARY, name='c_IP')
        
IP.update()

#Create openD variables
openD_IP = {}
for j in District_List:
    openD_IP[j] = IP.addVar(vtype=GRB.BINARY, name='openD_IP')
IP.update()

#Create openA variables
openA_IP = {}
for k in Appeals_List:
    openA_IP[k] = IP.addVar(vtype=GRB.BINARY, name='openA_IP')
IP.update()

endtime = time.time()
print("create variables time %.2f"%(endtime-starttime))
   
##############################################################################
# ### Create CONSTRAINTS
# In[ ]:
expr_IP = LinExpr()

starttime = time.time()     
for i in Settlement_List:
    
    #One S -> D Assignment   
    expr_IP.clear()
    for j in District_List:
        
        if (type(d_IP[i][j]) == gurobipy.Var):
            expr_IP.add(d_IP[i][j])
        
    IP.addConstr(expr_IP == 1, name='StoD')
    
    #One S -> A Assignment   
    expr_IP.clear()
    for k in Appeals_List:
        
        if (type(a_IP[i][k]) == gurobipy.Var):
            expr_IP.add(a_IP[i][k])
   
    IP.addConstr(expr_IP == 1, name ='StoA')

IP.update()
endtime = time.time()
print("settlement constraints time %.2f"%(endtime-starttime))     
    
# In[ ]:

starttime = time.time()
for j in District_List:
    
    #assign OpenD
    expr_IP.clear()
    for i in Settlement_List:
        
        if (type(d_IP[i][j]) == gurobipy.Var):
            expr_IP.add(d_IP[i][j])
            
    IP.addConstr(expr_IP <= S * openD_IP[j])
    IP.addConstr(expr_IP >= openD_IP[j])
    
    #at most ONE D -> A Assignment   
    expr_IP.clear()
    for k in Appeals_List:
        
        if (type(c_IP[j][k]) == gurobipy.Var):
            expr_IP.add(c_IP[j][k])
            
    IP.addConstr(expr_IP == openD_IP[j], name='DtoA')
    
IP.update()
endtime = time.time()
print("district constraints time %.2f"%(endtime-starttime)) 

# In[ ]:
starttime = time.time()
for k in Appeals_List:
    
    #assign OpenA
    expr_IP.clear()
    for i in Settlement_List:
        
        if (type(a_IP[i][k]) == gurobipy.Var):
            expr_IP.add(a_IP[i][k])

    IP.addConstr(expr_IP <= S * openA_IP[k])
    IP.addConstr(expr_IP >= openA_IP[k])
    
    #assign OpenA 
#    expr_IP.clear()
#    for j in District_List:
#        
#        if (type(c_IP[j][k]) == gurobipy.Var):
#            expr_IP.add(c_IP[j][k])
#            
#    IP.addConstr(expr_IP  <= D * openA_IP[k])

IP.update()
endtime = time.time()
print("appeals constraints time %.2f"%(endtime-starttime)) 

# In[ ]:
##### Max Open Courthouse Constraints

IP.addConstr(quicksum(openD_IP[j] for j in District_List) <= maxOpenD, name ="openD_Limit")
IP.addConstr(quicksum(openA_IP[k] for k in Appeals_List) <= maxOpenA, name = "openA_Limit")

IP.update()

# ### Callback Function
# In[ ]:
def mycallback(model, where):        
    if where == GRB.Callback.MIPSOL:
        
        #read current MIP solution
        d_temp = {}
        a_temp = {}
        c_temp = {}

        for j in District_List:
            c_temp[j] = {}
            
            for k in Appeals_List:
                if (type(c_IP[j][k]) == gurobipy.Var):
                    c_temp[j][k] = round(IP.cbGetSolution(c_IP[j][k]))
                else:
                    c_temp[j][k] = 0
        
        for i in Settlement_List:
            d_temp[i] = {}
            a_temp[i] = {}
            
            for j in District_List:
                if (type(d_IP[i][j]) == gurobipy.Var):
                    d_temp[i][j] = round(IP.cbGetSolution(d_IP[i][j]))
                else:
                    d_temp[i][j] = 0
        
            for k in Appeals_List:
                if (type(a_IP[i][k]) == gurobipy.Var):
                    a_temp[i][k] = round(IP.cbGetSolution(a_IP[i][k]))
                else:
                    a_temp[i][k] = 0
                    
       
        #add constraint that settlements must be assigned to same appeals court as district court         
        for i in Settlement_List:
            for j in District_List:
                if (type(d_IP[i][j]) == gurobipy.Var):
                    for k in Appeals_List:
                        if (type(a_IP[i][k]) == gurobipy.Var) and \
                            (type(c_IP[j][k]) == gurobipy.Var):
                                
                            if (d_temp[i][j] + a_temp[i][k] - c_temp[j][k] > 1):
                                IP.cbLazy(d_IP[i][j] + a_IP[i][k] - c_IP[j][k] <= 1)  
                                
                            if (d_temp[i][j] - a_temp[i][k] + c_temp[j][k] > 1):
                                IP.cbLazy(d_IP[i][j] - a_IP[i][k] + c_IP[j][k] <= 1)

    
    #Capacity constraints 
        for j in District_List:
            expr_IP.clear()
            assigned = 0
            
            for i in Settlement_List:
                if (type(d_IP[i][j]) == gurobipy.Var):
                    assigned = assigned + d_temp[i][j]
                    expr_IP.add(s_pop[i]* d_IP[i][j])
            
            if assigned > d_pop[j]:
                IP.cbLazy(expr_IP  <= d_pop[j])
            
        for k in Appeals_List:
            expr_IP.clear()
            assigned = 0
            
            for i in Settlement_List:
                if (type(a_IP[i][k]) == gurobipy.Var):
                    assigned = assigned + a_temp[i][k]
                    expr_IP.add(s_pop[i]*a_IP[i][k])
            
            if assigned > d_pop[j]:
                IP.cbLazy(expr_IP  <= a_pop[k])
                
# In[ ]:
IP

#%%

# ### Set Objective Function
starttime = time.time()
expr_IP.clear()

for i in Settlement_List:
    
    for j in District_List:
        if (type(d_IP[i][j]) == gurobipy.Var):
            expr_IP.addTerms(Dist_D[i,j], d_IP[i][j])
            
    for k in Appeals_List:
        if (type(a_IP[i][k]) == gurobipy.Var):
            expr_IP.addTerms(Dist_A[i,k], a_IP[i][k])
            
IP.setObjective(expr_IP, GRB.MINIMIZE)
IP.update()

endtime = time.time()
#print("objective time %.2f "%(endtime-starttime))

# ### Optimize
# In[ ]:
#start_time = time.time()

IP.optimize(mycallback)

#endtime = time.time()
#print("IP optimization time %.2f"%(endtime-starttime))  

#%%
#assign merged settlements
d={}
a={}  
c={}

for i in All_S_List:  
    d[i] = {}
    a[i] = {} 
    for j in District_List:    
        d[i][j]=0 
               
    for k in Appeals_List:
        a[i][k]=0

for i in Settlement_List:      
    for s in (Reduced_Dict[i]): # for the merged settlements
        if (np.isnan(s) != True):
            
            #if i is assigned to the courthouse, assign s also
            for j in District_List:
                if(VarVal(d_IP[i][j]) == 1):
                    d[s][j]=1
                    
            for k in Appeals_List:
                if(VarVal(a_IP[i][k]) == 1):
                    a[s][k]=1
    
    for j in District_List: 
        if(VarVal(d_IP[i][j]) == 1):
            d[i][j]=1 
               
    for k in Appeals_List:
        if(VarVal(a_IP[i][k]) == 1):
            a[i][k]=1
#read c_IP
for j in District_List:
     c[j] = {} 
     for k in Appeals_List:
        if(VarVal(c_IP[j][k]) == 1):
            c[j][k]=1
        else:
            c[j][k]=0

#%%# ### Output Solution
outfile = open( 'Settlement_to_Appeals.txt', 'w' )
for key, value in sorted( a.items() ):
        outfile.write( str(key) + '\t' + str(value) + '\n' )

outfile = open( 'Settlement_to_District.txt', 'w' )
for key, value in sorted( d.items() ):
        outfile.write( str(key) + '\t' + str(value) + '\n' )
        
outfile = open( 'District_to_Appeals.txt', 'w' )
for key, value in sorted( c_IP.items() ):
        outfile.write( str(key) + '\t' + str(value) + '\n' )
        
outfile = open( 'Open_Appeals.txt', 'w' )
for key, value in sorted( openA_IP.items() ):
        outfile.write( str(key) + '\t' + str(value) + '\n' )

outfile = open( 'Open_Districts.txt', 'w' )
for key, value in sorted( openD_IP.items() ):
        outfile.write( str(key) + '\t' + str(value) + '\n' )
              
# In[ ]:
#start_time = time.time()
#
IP.write("out_IP_%s.sol"%Sfile)
#
#endtime = time.time()
#print("write solution file time %.2f"%(endtime-starttime))  

# In[ ]:

GRB.Attr

#%%
# ###  Plotting

 #%%
plt.figure(figsize=(10,8))
 
#define colors 
cm.get_cmap('Set3')

clr = itertools.cycle(cm.Set3(np.linspace(0,1,11)))


#plot assignments
for j in District_List:
    #location of district court j
    d_lon = District_Dict[j][1]
    d_lat = District_Dict[j][2]
        
    if(VarVal(openD_IP[j]) == 1):
        s_lon = []
        s_lat = []
        
        #find settlements assigned to district court j
        for i in All_S_List:
#            if (VarVal(d_IP[i][j]) == 1):
            if (d[i][j] == 1):
                #array of settlement locations assigned to district court j
                s_lon.append(All_Settlement_Dict[i][2])
                s_lat.append(All_Settlement_Dict[i][1])
        
        c=next(clr) #color of district
        plt.scatter(d_lon, d_lat, color=c, zorder= 3, marker = 's',edgecolor='black', s=40) #plot district court j
        plt.scatter(s_lon, s_lat, color=c, zorder= 1) #plot settlements assigned to district court j
        
        #find appeals court assigned to district court j
        for k in Appeals_List:
            a_lon = Appeals_Dict[k][1]
            a_lat = Appeals_Dict[k][2]
            
            #if (VarVal(c_IP[j][k]) == 1):
                #plt.plot([a_lon,d_lon], [a_lat,d_lat], zorder=2, color='black') #draw line from Dj to Ak
                #plt.scatter(a_lon, a_lat, color='aqua', zorder= 4, marker = '^',edgecolor='black', s=50) #plot appeals court k

    else:  #plot unassigned district courthouse    
        plt.scatter(d_lon, d_lat, color='blue', zorder= 5, marker = 's',edgecolor='blue', s=40)
        
##plot unused appeals courts        
#for k in Appeals_List:
#        if (VarVal(openA_IP[k]) != 1):
#            a_lon = Appeals_Dict[k][1]
#            a_lat = Appeals_Dict[k][2]
#            plt.scatter(a_lon, a_lat, color='blue', zorder= 5, marker = "p", s=100) #plot appeals court k


sf = shp.Reader("Afghanistan_Districts","rb")
for shape in sf.shapeRecords():
    x = [i[0] for i in shape.shape.points[:]]
    y = [i[1] for i in shape.shape.points[:]]
    plt.plot(x,y,color='k',linewidth=0.1)
plt.show()
########################

numDopen = 0
numAopen = 0
for j in District_List:
    numDopen = numDopen + round(openD_IP[j].X)
    
for k in Appeals_List:
    numAopen = numAopen + round(openA_IP[k].X)

print("district courts open %d" %numDopen)
print("appeals courts open %d" %numAopen)


##########################################################################
#%%
#plot PROVINCE ASSIGNMENTS 
plt.figure(figsize=(10,8))

cm.get_cmap('Set3')
clr = itertools.cycle(cm.Set3(np.linspace(0,1,11)))

#find appeals court assigned to district court j
for k in Appeals_List:
    
    a_lon = Appeals_Dict[k][1]
    a_lat = Appeals_Dict[k][2]
    
    if(VarVal(openA_IP[k]) == 1):
        plt.scatter(a_lon, a_lat, color='aqua', zorder= 4, marker = '^',edgecolor='black', s=90) #plot appeals court k
               
        c=next(clr) #color of province
        
        for j in District_List:
            if (VarVal(c_IP[j][k]) == 1):
                d_lon = District_Dict[j][1]
                d_lat = District_Dict[j][2]        
                plt.plot([a_lon,d_lon], [a_lat,d_lat], zorder=2, color='black') #draw line from Dj to Ak
                plt.scatter(d_lon, d_lat, color=c, zorder= 3, marker = 's',edgecolor='black', s=15) #plot district court j
        
        s_lon = []
        s_lat = []
            
        for i in All_S_List:
            if (a[i][k] == 1):
                #array of settlement locations assigned to district court j
                s_lon.append(All_Settlement_Dict[i][2])
                s_lat.append(All_Settlement_Dict[i][1])
        plt.scatter(s_lon, s_lat, color=c, zorder= 1) #plot settlements assigned to district court j
    
    else:
        plt.scatter(a_lon, a_lat, color='red', zorder= 4, marker = '^',edgecolor='black', s=90) #plot appeals court k
##############################################################################

#%%
##check assignments
for i in Settlement_List:
    for j in District_List:
        for k in Appeals_List:
            
            if (VarVal(d_IP[i][j]) == 1) and (VarVal(c_IP[j][k]) == 1) : #S to D and D to A
                    if (type(a_IP[i][k]) == gurobipy.Var) and (VarVal(a_IP[i][k]) != 1): # S to A INCORRECT
                            print("ERROR (%i, %i, %i) d=%i a=%i c=%i" 
                                  %(i,j,k,VarVal(d_IP[i][j]),VarVal(a_IP[i][k]), VarVal(c_IP[j][k])))
                            
            if (VarVal(d_IP[i][j]) == 1) and (VarVal(c_IP[j][k]) == 0) : #S to D and D to A
                    if (type(a_IP[i][k]) == gurobipy.Var) and (VarVal(a_IP[i][k]) != 0): # S to A INCORRECT
                            print("ERROR (%i, %i, %i) d=%i a=%i c=%i" 
                                  %(i,j,k,VarVal(d_IP[i][j]),VarVal(a_IP[i][k]), VarVal(c_IP[j][k])))
            
print("done!")
        

import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import re
import numpy as np
import os
import glob
indiabatplayers = "C:\\Users\\u6062896\\Documents\\CricketProject\\cricinfoData\\Players\\Batting\\India"
australiabatplayers = 'C:\\Users\\u6062896\\Documents\\CricketProject\\cricinfoData\\Players\\Batting\\Australia'
indiaballplayers = 'C:\\Users\\u6062896\\Documents\\CricketProject\\cricinfoData\\Players\\Bowling\\India'
australiaballplayers = 'C:\\Users\\u6062896\\Documents\\CricketProject\\cricinfoData\\Players\\Bowling\\Australia'

dfiplyr = []
dfibplyr = []

ibatFiles = glob.glob(indiabatplayers + "/*Innings_all.csv")
iballFiles = glob.glob(indiaballplayers + "/*Innings_all.csv")

for batplayer in ibatFiles:
    player = batplayer.split("\\")[9]
    player = player.replace("Innings_all.csv","")
    df = pd.read_csv(batplayer,index_col=None, header=0)
    df['Player'] = player
    dfiplyr.append(df)

for ballplayer in iballFiles:
    player = ballplayer.split("\\")[9]
    player = player.replace("Innings_all.csv","")
    df = pd.read_csv(ballplayer,index_col=None, header=0)
    df['Player'] = player
    dfibplyr.append(df)

iframe = pd.concat(dfiplyr, axis = 0, ignore_index = True)
ibframe = pd.concat(dfibplyr, axis = 0, ignore_index = True)
iframe['Start Date'] = pd.to_datetime(iframe[' Start Date'], format = ' %d %b %Y')
ibframe['Start Date'] = pd.to_datetime(ibframe[' Start Date'], format = ' %d %b %Y')
iframe.drop([' Start Date'], axis=1, inplace=True)
ibframe.drop([' Start Date'], axis=1, inplace=True)

dfaplyr = []
dfabplyr = []

abatFiles = glob.glob(australiabatplayers + "/*Innings_all.csv")
aballFiles = glob.glob(australiaballplayers + "/*Innings_all.csv")

for batplayer in abatFiles:
    player = batplayer.split("\\")[9]
    player = player.replace("Innings_all.csv","")
    df = pd.read_csv(batplayer,index_col=None, header=0)
    df['Player'] = player
    dfaplyr.append(df)

for ballplayer in aballFiles:
    player = ballplayer.split("\\")[9]
    player = player.replace("Innings_all.csv","")
    df = pd.read_csv(ballplayer,index_col=None, header=0)
    df['Player'] = player
    dfabplyr.append(df)

aframe = pd.concat(dfaplyr, axis = 0, ignore_index = True)
abframe = pd.concat(dfabplyr, axis = 0, ignore_index = True)

aframe['Start Date'] = pd.to_datetime(aframe[' Start Date'], format = ' %d %b %Y')
abframe['Start Date'] = pd.to_datetime(abframe[' Start Date'], format = ' %d %b %Y')
aframe.drop([' Start Date'], axis=1, inplace=True)
abframe.drop([' Start Date'], axis=1, inplace=True)

allindiaframe = pd.read_csv("C:\\Users\\u6062896\\Documents\\CricketProject\\cricinfoData\\indiaallinningsodinumInnings.csv")
allindiabat = allindiaframe[[' Target', ' Inns', ' Result', ' Start Date', ' ODI_NUM']]
allindiabat['Start Date'] = pd.to_datetime(allindiabat[' Start Date'], format = ' %d %b %Y')
allindiabat.drop([' Start Date'], axis=1, inplace=True)
iplrbatdat = pd.merge(iframe, allindiabat, on = 'Start Date')
iplrboldat = pd.merge(ibframe, allindiabat, on = 'Start Date')

allaustraliaframe = pd.read_csv("C:\\Users\\u6062896\\Documents\\CricketProject\\cricinfoData\\australiaallinningsodinumInnings.csv")
allaustraliabat = allaustraliaframe[[' Target', ' Inns', ' Result', ' Start Date', ' ODI_NUM']]
allaustraliabat['Start Date'] = pd.to_datetime(allaustraliabat[' Start Date'], format = ' %d %b %Y')
allaustraliabat.drop([' Start Date'], axis=1, inplace=True)
aplrbatdat = pd.merge(aframe, allaustraliabat, on = 'Start Date')
aplrboldat = pd.merge(abframe, allaustraliabat, on = 'Start Date')

batteams = []
batteams.append(iplrbatdat)
batteams.append(aplrbatdat)

bolteams = []
bolteams.append(iplrboldat)
bolteams.append(aplrboldat)

allbatting = pd.concat(batteams, axis = 0, ignore_index = True) 
allbowling = pd.concat(bolteams, axis = 0, ignore_index=True) 

allbatting.to_csv("C:\\Users\\u6062896\\Documents\\CricketProject\\cricinfoData\\Players\\allbat.csv", index = False)
allbowling.to_csv("C:\\Users\\u6062896\\Documents\\CricketProject\\cricinfoData\\Players\\allbol.csv", index = False)

indiacummulative = "C:\\Users\\u6062896\\Documents\\CricketProject\\cricinfoData\\Players\\India"
australiacumulative = "C:\\Users\\u6062896\\Documents\\CricketProject\\cricinfoData\\Players\\Australia"

dficum = []
dfacum = []
_list = [] 
icumFiles = glob.glob(indiacummulative + "/*Cummulative.csv")
acumFiles = glob.glob(australiacumulative + "/*Cummulative.csv")

for file in icumFiles:
    player = file.split("\\")[8]
    player = player.replace("Cummulative.csv","")
    df = pd.read_csv(file,index_col=None, header=0)
    df['Player'] = player
    dficum.append(df)

for file in acumFiles:
    player = file.split("\\")[8]
    player = player.replace("Cummulative.csv","")
    df = pd.read_csv(file,index_col=None, header=0)
    df['Player'] = player
    dfacum.append(df)

_list = dficum+dfacum
allcum = pd.concat(_list, axis = 0, ignore_index = True)
allcum.to_csv("C:\\Users\\u6062896\\Documents\\CricketProject\\cricinfoData\\Players\\allcum.csv", index = False)
allcum.rename(columns={'[Mat': 'Mat', ' ODI_NUM]': 'ODI_NUM'}, inplace=True)

indiasummary = "C:\\Users\\u6062896\\Documents\\CricketProject\\cricinfoData\\Players\\India"
isumFiles = glob.glob(indiasummary + "/*Summary.csv")
dfisum = []
for file in isumFiles:
    player = file.split("\\")[8]
    player = player.replace("Summary.csv","")
    df = pd.read_csv(file, nrows=1, header=1)
    af = df.iloc[0:1, 1:13]
    af['Player'] = player
    dfisum.append(af)

aussummary = "C:\\Users\\u6062896\\Documents\\CricketProject\\cricinfoData\\Players\\Australia"
asumFiles = glob.glob(aussummary + "/*Summary.csv")
dfasum = []
for file in asumFiles:
    player = file.split("\\")[8]
    player = player.replace("Summary.csv","")
    df = pd.read_csv(file, nrows=1, header=1)
    af = df.iloc[0:1, 1:13]
    af['Player'] = player
    dfasum.append(af)

listsum = dfisum + dfasum
allsum = pd.concat(listsum, axis = 0, ignore_index = True)

allsum.to_csv("C:\\Users\\u6062896\\Documents\\CricketProject\\cricinfoData\\Players\\allsum.csv", index = False)

#Algorithm
u1 = 20
u2 = 5
u3 = 0.3
u4 = 0.7
u5 = 4
u6 = 0.35
u7 = 0.65
plrstat = []
#Batting Scores Calculations
for index, row in allsum.iterrows():
    record = []
    player = row['Player']
    crsum = allsum.loc[allsum['Player'] == player]
    crmatplayd = crsum.loc[:,' Mat']
    crnumcent = crsum.loc[:, " 100"]
    crbtavg = crsum.loc[:, " Bat Av"]
    crbtavg = crbtavg.replace(' -',0,regex=True).astype(float)
    crbtavg = crbtavg.astype(float).values[0]
    allinnplr = allbatting.loc[allbatting['Player'] == player]
    batinnplr = allinnplr.Runs.str.count("DNB").sum()
    batinnplr = int(batinnplr)
    u = np.sqrt((crmatplayd-batinnplr)/crmatplayd)
    v = u1*crnumcent
    w = u3*v + u4*crbtavg
    crscore = u*w
    M = allinnplr.nlargest(5, 'Start Date')
    M['Runs'] = M['Runs'].replace('\*',0,regex=True)
    #map(lambda x: re.sub(r'\W+', '', x))
    M['Runs'] = M['Runs'].replace('DNB',0,regex=True)
    #.map(lambda x: re.sub(r'[a-zA-Z]', 0, x))
    M['Runs'] = M['Runs'].apply(float)
    recscore = M['Runs'].mean()
    crscores = crscore.astype(float).values[0]
    data = [[player, crscores, recscore]]
    record = pd.DataFrame(data, columns=['Player','CareerScore', 'RecentScore'])
    print(record)
    #record.append(crscores)
    #record.append(recscore)
    plrstat.append(record)

playerstatdf = pd.concat(plrstat, axis = 0, ignore_index = True)
playerstatdf

#Bowling Scores calculation
bolstat = []
for index, row in allsum.iterrows():
    record = []
    player = row['Player']
    crsum = allsum.loc[allsum['Player'] == player]
    crmatplayd = crsum.loc[:,' Mat']
    crnum5s = crsum.loc[:, " 5"]
    crboavg = crsum.loc[:, " Bowl Av"]
    crwkts = crsum.loc[:, ' Wkts']
    crnum5s = crnum5s.replace(' -',0,regex=True).astype(float)
    crnum5s = crnum5s.astype(float).values[0]
    crboavg = crboavg.replace(' -',0,regex=True).astype(float)
    crboavg = crboavg.astype(float).values[0]
    crwkts = crwkts.replace(' -',0,regex=True).astype(float)
    crwkts = crwkts.astype(float).values[0]
    allinnplr = allbowling.loc[allbowling['Player'] == player]
    allinnplr.rename(columns={'[Overs': 'Overs'}, inplace=True)
    bolinnplr = allinnplr.Overs.str.count("DNB").sum()
    bolinnplr = int(bolinnplr)
    bolecon = allinnplr.loc[:, ' Econ']
    bolecon = bolecon.replace(' -',0,regex=True).astype(float)
    bolecon = bolecon.astype(float).values[0]
    u = np.sqrt((crmatplayd-bolinnplr)/crmatplayd)
    v = u1*crnum5s + u2*crwkts
    w = crboavg*bolecon
    bowlscore = (u*v)/w
    bowlscore = bowlscore.astype(float).values[0]
    data = [[player, bowlscore]]
    record = pd.DataFrame(data, columns=['Player','BowlingScore'])
    bolstat.append(record)

playerbolstatdf = pd.concat(bolstat, axis = 0, ignore_index = True)
playerbolstatdf

#Relative comparison of players to model team strengths
ibatstrength = np.sum(playerstatdf['CareerScore'].isin(dfiplyr['Player']))
abatstrength = np.sum(playerstatdf['CareerScore'].isin(dfaplyr['Player']))
ibolstrength = np.sum(playerstatdf['BowlingScore'].isin(dfibplyr['Player']))
abolstrength = np.sum(playerstatdf['BowlingScore'].isin(dfabplyr['Player']))

relativestrength = (ibatstrength/abolstrength)-(abatstrength/ibolstrength)

frame = [allbatting, allbowling]
df = pd.concat(frame, axis = 1, ignore_index = True)



def predictMatch(squada, squadb,Toss,Venue):
    # Split the targets into training/testing sets
    df_x_train = df[:-20]
    df_x_test = df[-20:]

    # Split the targets into training/testing sets
    df_y_train = df.Result[:-20]
    df_y_test = df.Result[-20:]

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(df_x_train, df_y_train)

    # Make predictions using the testing set
    df_y_pred = regr.predict(df_x_test)

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f"
        % mean_squared_error(df_y_test, df_y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(df_y_test, df_y_pred))

    # Plot outputs
    plt.scatter(df_x_test, df_y_test,  color='black')
    plt.plot(df_x_test, df_y_pred, color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()

    submission_df = df_y_pred.copy()
    submission_df.to_csv('submission3.csv', index=False)

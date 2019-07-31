from django.shortcuts import render, get_object_or_404
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.ticker as plticker
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def Soccer_home(request):
	if request.method == 'POST':
		h_team = request.POST['home']
		a_team = request.POST['away']
		results = pd.read_csv('C:/Users/Amey/Desktop/dag/MLproject/Soccer_pred/results.csv')
		worldcup_teams = ['Australia', 'Iran', 'Japan', 'Korea Republic', 
		            'Saudi Arabia', 'Egypt', 'Morocco', 'Nigeria', 
		            'Senegal', 'Tunisia', 'Costa Rica', 'Mexico', 
		            'Panama', 'Argentina', 'Brazil', 'Colombia', 
		            'Peru', 'Uruguay', 'Belgium', 'Croatia', 
		            'Denmark', 'England', 'France', 'Germany', 
		            'Iceland', 'Poland', 'Portugal', 'Russia', 
		            'Serbia', 'Spain', 'Sweden', 'Switzerland']
		winner = []
		for i in range (len(results['home_team'])):
		    if results ['home_score'][i] > results['away_score'][i]:
		        winner.append(results['home_team'][i])
		    elif results['home_score'][i] < results ['away_score'][i]:
		        winner.append(results['away_team'][i])
		    else:
		        winner.append('Draw')
		results['winning_team'] = winner
		df_teams_home = results[results['home_team'].isin(worldcup_teams)]
		df_teams_away = results[results['away_team'].isin(worldcup_teams)]
		df_teams = pd.concat((df_teams_home, df_teams_away))
		df_teams.drop_duplicates()
		df_teams.count()
		year = []
		for row in df_teams['date']:
		    year.append(int(row[:4]))
		df_teams['match_year'] = year
		df_teams_1930 = df_teams[df_teams.match_year >= 1930]
		#print(df_teams_1930.head())
		df_teams_1930 = df_teams.drop(['date', 'home_score', 'away_score', 'tournament', 'city', 'country', 'match_year', 'neutral'], axis=1)
		#print(df_teams_1930.head())
		df_teams_1930 = df_teams_1930.reset_index(drop=True)
		df_teams_1930.loc[df_teams_1930.winning_team == df_teams_1930.home_team,'winning_team']=2
		df_teams_1930.loc[df_teams_1930.winning_team == 'Draw', 'winning_team']=1
		df_teams_1930.loc[df_teams_1930.winning_team == df_teams_1930.away_team, 'winning_team']=0
		#print(df_teams_1930.head())
		final = pd.get_dummies(df_teams_1930, prefix=['home_team', 'away_team'], columns=['home_team', 'away_team'])
		X = final.drop(['winning_team'], axis=1)
		y = final["winning_team"]
		y = y.astype('int')
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
		logreg = LogisticRegression(multi_class='auto',solver='liblinear')
		logreg.fit(X_train, y_train)
		score = logreg.score(X_train, y_train)
		score2 = logreg.score(X_test, y_test)
		#print("Training set accuracy: ", '%.3f'%(score))
		#print("Test set accuracy: ", '%.3f'%(score2))
		#print(final.head())
		pred_set = []
		#print("World cup teams:",worldcup_teams)
		#h_team = input("Enter home team:")
		#a_team = input("Enter away team:")
		pred_set.append({'home_team': h_team, 'away_team': a_team, 'winning_team': None})
		pred_set = pd.DataFrame(pred_set)
		backup_pred_set = pred_set
		pred_set = pd.get_dummies(pred_set, prefix=['home_team', 'away_team'], columns=['home_team', 'away_team'])
		missing_cols = set(final.columns) - set(pred_set.columns)
		for c in missing_cols:
		    pred_set[c] = 0
		pred_set = pred_set[final.columns]
		pred_set = pred_set.drop(['winning_team'], axis=1)
		#print(pred_set.head())
		predictions = logreg.predict(pred_set)
		print("\n"+backup_pred_set.iloc[0, 1] + " and " + backup_pred_set.iloc[0, 0])
		if predictions[0] == 2:
		    p = backup_pred_set.iloc[0, 1]
		elif predictions[0] == 1:
		    p = "Draw"
		elif predictions[0] == 0:
		    p = backup_pred_set.iloc[0, 0]
		print('Probability of ' + backup_pred_set.iloc[0, 1] + ' winning: ', '%.3f'%(logreg.predict_proba(pred_set)[0][2]))
		print('Probability of Draw: ', '%.3f'%(logreg.predict_proba(pred_set)[0][1]))
		print('Probability of ' + backup_pred_set.iloc[0, 0] + ' winning: ', '%.3f'%(logreg.predict_proba(pred_set)[0][0]))
		print("")
		context = {
				'Home':backup_pred_set.iloc[0, 1],
				'Away':backup_pred_set.iloc[0, 0],
				'Winner':p,
				'HPW':'%.3f'%(logreg.predict_proba(pred_set)[0][2]),
				'APW':'%.3f'%(logreg.predict_proba(pred_set)[0][0]),
				'PD':'%.3f'%(logreg.predict_proba(pred_set)[0][1]),
		}
		return render(request, 'Soccer_pred/op.html', context)

	return render(request, 'Soccer_pred/wc.html')
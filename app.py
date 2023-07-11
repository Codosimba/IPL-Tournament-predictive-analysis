import streamlit as st
import pickle
import pandas as pd
teams = ['Royal Challengers Bangalore', 'Delhi Capitals', 'Mumbai Indians',
       'Kings XI Punjab', 'Kolkata Knight Riders', 'Sunrisers Hyderabad',
       'Rajasthan Royals', 'Chennai Super Kings']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah', 'Dubai', 'Mohali', 'Bengaluru']

## Import the model
import pickle
pipe = pickle.load(open('log_model.pkl','rb'))



st.title('IPL Win Prediction') 

col1,col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select Batting Team',sorted(teams))

with col2:
    bowling_team = st.selectbox('Select Bowling Team',sorted(teams))


select_city = st.selectbox('Select host city',sorted(cities))

target = st.number_input('Target')

### Definig Match conditions
col3,col4,col5 = st.columns(3)

with col3:
    score = st.number_input('Score')

with col4:
    overs_completed = st.number_input('Overs Completed')

with col5:
    wickets = st.number_input('Wickets fallen')

if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - overs_completed*6
    wickets = 10 - wickets
    crr = score/overs_completed
    rrr = (runs_left*6)/balls_left
    
    input_df =  pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],'city':[select_city],'runs_left':[runs_left],'balls_left':[balls_left],'wickets_left':[wickets],'total_run_x':[target],'crr':[crr],'rrr':[rrr]})

    result = pipe.predict_proba(input_df)
    st.text(result)




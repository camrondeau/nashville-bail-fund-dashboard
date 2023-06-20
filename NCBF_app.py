###########################################################################################################
###########################################################################################################
###########################################################################################################
######## coded by = Cameron Rondeau
######## insitution = Nashville Community Bail Fund
######## application = Interactive Dashboard
###########################################################################################################
###########################################################################################################
###########################################################################################################

# import streamlit
import streamlit as st

# import python modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import altair as alt
from scipy.stats import gaussian_kde

st.set_page_config(page_title='NCBF Dashboard', page_icon=':bar_chart:', layout='centered')

st.title('Nashville Community Bail Fund Dashboard')

# read in file
uploaded_file = st.file_uploader("Upload a file (Export entire database as CSV file from Bail Fund App)")

progress_text = "Running Analysis. Please wait."
percent_complete = 0

font_css = """
<style>
button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
  font-size: 24px;
}
</style>
"""

st.write(font_css, unsafe_allow_html=True)

listTabs = ["Summary Data", "Demographics\u2001"]

tab1, tab2 = st.tabs(listTabs)


if uploaded_file is not None:
    with tab1:
        my_bar = st.progress(percent_complete, progress_text) ########################################

        data_upload = pd.read_csv(uploaded_file)

        percent_complete += 10
        my_bar.progress(percent_complete, text=progress_text) ############################################  

        # add age variable
        def age(dob):
            if type(dob) == float:
                return None
            born = datetime.strptime(dob, "%m/%d/%Y").date()
            today = date.today()
            return today.year - born.year - ((today.month, 
                                              today.day) < (born.month, 
                                                            born.day))

        data_upload['Age'] = data_upload['Date of Birth'].apply(age)


        # Demographic features like race and gender are dominated by 2-3 categories,
        # with all other categories making up less than 1% of all data.
        # Grouping these other categories will assist in analysis and visualizations

        # group gender feature into Male/Female/Other-Unknown
        def summary_gender(row):
            if row['Gender'] == 'Cis Man' or row['Gender'] == 'Male':
                return 'Male'
            elif row['Gender'] == 'Cis Woman' or row['Gender'] == 'Female':
                return 'Female'
            else:
                return 'Unknown/Other'

        data_upload['Gender'] = data_upload.apply(lambda row: summary_gender(row), axis=1)

        percent_complete += 10
        my_bar.progress(percent_complete, text=progress_text) ############################################

        # group race feature into Black/White/Hispanic-Latinx/Other-Unknown
        def summary_race(row):
            if row['Race'] == 'Black':
                return 'Black'
            elif row['Race'] == 'White':
                return 'White'
            elif row['Race'] == 'Hispanic/Latinx':
                return 'Hispanic/Latinx'
            else:
                return 'Unknown/Other'

        data_upload['Race'] = data_upload.apply(lambda row: summary_race(row), axis=1)



        # sum together all cases
        data_upload['Bail Amount Summed'] = data_upload.fillna(0)['[#1] Bail Amount'] + data_upload.fillna(0)['[#2] Bail Amount'] + data_upload.fillna(0)['[#3] Bail Amount'] + data_upload.fillna(0)['[#4] Bail Amount'] 
        data_upload['Initial Bail Amount Summed'] = data_upload.fillna(0)['[#1] Initial Bail Amount'] + data_upload.fillna(0)['[#2] Initial Bail Amount'] + data_upload.fillna(0)['[#3] Initial Bail Amount'] + data_upload.fillna(0)['[#4] Initial Bail Amount'] 
        data_upload['Bail Amount Posted Summed'] = data_upload.fillna(0)['[#1] Bail Amount Posted'] + data_upload.fillna(0)['[#2] Bail Amount Posted'] + data_upload.fillna(0)['[#3] Bail Amount Posted'] + data_upload.fillna(0)['[#4] Bail Amount Posted'] 
        data_upload['Amount Returned Summed'] = data_upload.fillna(0)['[#1] Amount Returned'] + data_upload.fillna(0)['[#2] Amount Returned'] + data_upload.fillna(0)['[#3] Amount Returned'] + data_upload.fillna(0)['[#4] Amount Returned'] 

        percent_complete += 10
        my_bar.progress(percent_complete, text=progress_text) ############################################

        # collect important variables from full dataset (to be confirmed)
        important_vars = ['__document_id', 'Date & Time Created', 'Client Name', 'First Name', 'Last Name', 'Date of Birth', 'Age',
                          'Race', 'Gender', 'Request Status', 'Bail Status', 'Total Bail Amount', 
                          'Bail Amount Summed', 'Initial Bail Amount Summed', 'Bail Amount Posted Summed', 'Amount Returned Summed',  
                          'Arrest Date', 'OCA Number', 'Next Court Appearance Date', 'Attorney Name', 'Employment status?', 
                          'Is this person experiencing homelessness?', 'Facility where person is held', '[#1] Case Number', 
                          '[#1] Charges', '[#1] Charge Type', '[#1] Bail Amount', '[#1] Initial Bail Amount', '[#1] Bail Amount Posted', 
                          '[#1] Date Posted', '[#1] Bail Disposition', '[#1] Bail Disposition Date','[#1] Case Disposition',  
                          '[#1] Case Disposition Date', '[#1] Bail Refunded?', '[#1] Amount Returned',
                          '[#1] Date Returned', '[#1] Disposition', '[#1] Bail Posted By Who', '[#2] Charge Type', '[#3] Charge Type', '[#4] Charge Type',
                          '[#2] Charges', '[#3] Charges', '[#4] Charges']

        # select important columns from original dataset
        data = data_upload[important_vars]

        #if st.checkbox('Show data used in analysis'):
        #    st.subheader('Working data')
        #    st.write(data)


        # combine New and In Process into one
        data['Bail Status'].replace(['Partial Return'], 'Returned', inplace=True)

        # combine New and In Process into one
        data['Request Status'].replace(['New', 'In Process'], 'New or In Process', inplace=True)

        def status_sum(row):
            if row['Request Status'] == 'New or In Process':
                return 'Completed'
            else:
                return row['Request Status']

        data['Bail Request Status'] = data.apply(lambda row: status_sum(row), axis=1)

        

        posted_dataset = data[data['Bail Request Status'] == 'Completed']
        not_posted_dataset = data[data['Bail Request Status'] == 'Bail Not Made']

        def open_close(row):
            if row['Bail Status'] == 'Returned' or row['Bail Status'] == 'Eligible for Return' or row['Bail Status'] == 'Final Forfeit':
                return 'Closed'
            elif row['Bail Status'] == 'Posted' or row['Bail Status'] == 'FTA':
                return 'Open'

        posted_dataset['Case Status'] = posted_dataset.apply(lambda row: open_close(row), axis=1)

        open_dataset = posted_dataset[posted_dataset['Case Status'] == 'Open']
        closed_dataset = posted_dataset[posted_dataset['Case Status'] == 'Closed']



        # 1. Number of people bailed out by NCBF
        # based on entries in the dataset (not accounting for duplicate case numbers or same person being bailed out twice)
        # numbers based on 'Total Bail Amount' column
        c = data['Request Status'].value_counts().astype(str)
        p = data['Request Status'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
        bail_sum = data.groupby(by='Request Status')['Total Bail Amount'].sum().apply(lambda x: "${:,.0f}".format(x))
        bail_sum2 = data.groupby(by='Request Status')['Total Bail Amount'].sum()
        bail_mean = data.groupby(by='Request Status')['Total Bail Amount'].mean().apply(lambda x: "${:,.0f}".format(x))
        bail_requests = pd.concat([c, p, bail_sum, bail_mean], axis=1, keys=['Count', 'Percentage', 'Total Bail Amount', 'Average Bail Amount'])
        bail_requests2 = pd.concat([c, p, bail_sum2, bail_mean], axis=1, keys=['Count', 'Percentage', 'Total Bail Amount', 'Average Bail Amount'])
        bail_requests.index.name = 'Request Status'
        bail_requests.reset_index(inplace=True)
        bail_requests2.index.name = 'Request Status'
        bail_requests2.reset_index(inplace=True)

          
    
    
    
    
    
    
    
    
    
    
    
    
    
        st.header('Bail Requests')
        
        #st.write('Total Bail Requests:', str(len(data)))

        st.dataframe(bail_requests.set_index(bail_requests.columns[0]), use_container_width=True)

        

        c1, c2 = st.columns(2)
        with c1:
            fig = px.pie(bail_requests, values='Count', names='Request Status', title='Bail Request Counts')
            #fig.update_layout(legend_title=None, legend_y=0.5)
            #fig.update_traces(textinfo='percent+label', textposition='inside')
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.pie(bail_requests2, values='Total Bail Amount', names='Request Status', title='Bail Request Amounts')
            #fig.update_layout(legend_title=None, legend_y=0.5)
            #fig.update_traces(textinfo='percent+label', textposition='inside')
            st.plotly_chart(fig, use_container_width=True)
    
    
    
    
        percent_complete += 10
        my_bar.progress(percent_complete, text=progress_text) ############################################
    
    
    
    
    
    
    
    
    
    
    
    
        st.header('Bail Status')
        
        st.write('For cases in which the NCBF has posted bail')

        case_status_fiter = st.selectbox('Filter by Case Status', ('All', 'Open', 'Closed'))

        if case_status_fiter != 'All':
            filtered_case_status = posted_dataset[posted_dataset['Case Status'] == case_status_fiter]
        else:
            filtered_case_status = posted_dataset

        # Total Bail Amount = Column filled in
        # Bail Summed = Adding each [#1] [#2]... together
        c = filtered_case_status['Bail Status'].value_counts().astype(str)
        p = filtered_case_status['Bail Status'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
        bail_sum = filtered_case_status.groupby(by='Bail Status')['Total Bail Amount'].sum().apply(lambda x: "${:,.0f}".format(x))
        bail_sum2 = filtered_case_status.groupby(by='Bail Status')['Total Bail Amount'].sum()
        bail_status = pd.concat([c,p, bail_sum], axis=1, keys=['Count', 'Percentage', 'Total Bail Amount'])
        bail_status2 = pd.concat([c,p, bail_sum2], axis=1, keys=['Count', 'Percentage', 'Total Bail Amount'])
        bail_status.index.name = 'Bail Status'
        bail_status.reset_index(inplace=True)
        bail_status2.index.name = 'Bail Status'
        bail_status2.reset_index(inplace=True)
        bail_status['Case Status'] = bail_status.apply(lambda row: open_close(row), axis=1)
        bail_status = bail_status[['Case Status', 'Bail Status', 'Count', 'Percentage', 'Total Bail Amount']]

        st.dataframe(bail_status.set_index(bail_status.columns[0]), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.pie(bail_status, values='Count', names='Bail Status', title='Bail Status Counts')
            #fig.update_layout(legend_title=None, legend_y=0.5)
            #fig.update_traces(textinfo='percent+label', textposition='inside')
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.pie(bail_status2, values='Total Bail Amount', names='Bail Status', title='Amounts by Bail Status')
            #fig.update_layout(legend_title=None, legend_y=0.5)
            #fig.update_traces(textinfo='percent+label', textposition='inside')
            st.plotly_chart(fig, use_container_width=True)
            
        
        
        
        percent_complete += 10
        my_bar.progress(percent_complete, text=progress_text) ############################################
        
        
        
        
        
        
        
        
        
        
        st.header('Bail Outcomes')
        
        st.write('For closed cases in which the NCBF has posted bail')
        
        closed_cases = data[(data['Request Status'] == 'Completed') & ((data['Bail Status'] == 'Returned') | (data['Bail Status'] == 'Final Forfeit') | (data['Bail Status'] == 'Eligible for Return'))]
        def success_outcome_full(row):
            if row['[#1] Bail Disposition'] == 'Bond Surrendered' or row['[#1] Bail Disposition'] == 'Bond Relieved' or row['[#1] Bail Disposition'] == 'Bond Revoked' or row['[#1] Bail Disposition'] == 'Capias Served' or row['[#1] Bail Disposition'] == 'Final Forfeit':
                return 'Unsuccessful'
            elif row['[#1] Bail Disposition'] == 'Abated By Death':
                return 'Abated By Death'
            else:
                return 'Successful'

        closed_cases['Outcome'] = closed_cases.apply(lambda row: success_outcome_full(row), axis=1)
        
        
        bail_status_fiter = st.selectbox('Filter by Case Outcome', ('All', 'Successful', 'Unsuccessful'))

        
        if bail_status_fiter != 'All':
            filtered_bail_status = closed_cases[closed_cases['Outcome'] == bail_status_fiter]
        else:
            filtered_bail_status = closed_cases

        def summary_bail_outcome(row):
            if row['[#1] Bail Disposition'] == 'Guilty of Lesser':
                return 'Guilty'
            elif row['[#1] Bail Disposition'] == 'Judgement Deferred' or row['[#1] Bail Disposition'] == 'No True Bill' or row['[#1] Bail Disposition'] == 'Not Guilty, Trial':
                return 'Dismissed'
            elif row['[#1] Bail Disposition'] == 'Bond Surrendered' or row['[#1] Bail Disposition'] == 'Bond Relieved' or row['[#1] Bail Disposition'] == 'Bond Revoked':
                return 'Bond Surrendered'
            else:
                return row['[#1] Bail Disposition']

        filtered_bail_status['Bail Disp Custom'] = filtered_bail_status.apply(lambda row: summary_bail_outcome(row), axis=1)
        
        def success_outcome(row):
            if row['Disposition'] == 'Bond Surrendered' or row['Disposition'] == 'Bond Relieved' or row['Disposition'] == 'Bond Revoked' or row['Disposition'] == 'Capias Served' or row['Disposition'] == 'Final Forfeit':
                return 'Unsuccessful'
            elif row['Disposition'] == 'Abated By Death':
                return 'Abated By Death'
            else:
                return 'Successful'

        
        # Based on Case #1 'Bail Disposition' data
        c = filtered_bail_status['Bail Disp Custom'].value_counts().astype(str)
        p = filtered_bail_status['Bail Disp Custom'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
        bail_sum = filtered_bail_status.groupby(by='Bail Disp Custom')['Total Bail Amount'].sum().apply(lambda x: "${:,.0f}".format(x))
        outcomes = pd.concat([c,p, bail_sum], axis=1, keys=['Count', 'Percentage', 'Total Bail Amount'])
        outcomes.index.name = 'Disposition'
        outcomes.reset_index(inplace=True)
        outcomes['Outcome'] = outcomes.apply(lambda row: success_outcome(row), axis=1)
        outcomes = outcomes[['Outcome', 'Disposition', 'Count', 'Percentage', 'Total Bail Amount']]
        
        st.table(outcomes.set_index(outcomes.columns[0]))
        #st.dataframe(outcomes.set_index(outcomes.columns[0]), use_container_width=True)
        
        fig = px.pie(outcomes, values='Count', names='Disposition')
        #fig.update_layout(legend_title=None, legend_y=0.5)
        #fig.update_traces(textinfo='percent+label', textposition='inside')
        st.plotly_chart(fig, use_container_width=True)
        
        
        percent_complete += 10
        my_bar.progress(percent_complete, text=progress_text) ############################################
        
        def success_outcome(row):
            if row['[#1] Bail Disposition'] == 'Bond Surrendered' or row['[#1] Bail Disposition'] == 'Bond Relieved' or row['[#1] Bail Disposition'] == 'Bond Revoked' or row['[#1] Bail Disposition'] == 'Capias Served' or row['[#1] Bail Disposition'] == 'Final Forfeit':
                return 'Unsuccessful'
            elif row['[#1] Bail Disposition'] == 'Abated By Death':
                return 'Abated By Death'
            else:
                return 'Successful'

        closed_cases['Outcome'] = closed_cases.apply(lambda row: success_outcome(row), axis=1)
        
        
        
        
        
        
        
        st.header('Charge Type')
        
        st.write('Shows the highest charge if there are multiple charges in one arrest')
        
        def max_charge_type(row):
            if row['[#1] Charge Type'] == 'Felony' or row['[#1] Charge Type'] == 'Felony & Misdemeanor' or row['[#2] Charge Type'] == 'Felony' or row['[#2] Charge Type'] == 'Felony & Misdemeanor' or row['[#3] Charge Type'] == 'Felony' or row['[#3] Charge Type'] == 'Felony & Misdemeanor' or row['[#4] Charge Type'] == 'Felony' or row['[#4] Charge Type'] == 'Felony & Misdemeanor':
                return 'Felony'
            else:
                return row['[#1] Charge Type']

        closed_cases['Highest Offense'] = closed_cases.apply(lambda row: max_charge_type(row), axis=1)
        data['Highest Offense'] = data.apply(lambda row: max_charge_type(row), axis=1)

        c = data['Highest Offense'].value_counts().astype(str)
        p = data['Highest Offense'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
        charges_full = pd.concat([c,p], axis=1, keys=['Counts', 'Percentage'])

        charges_full.index.name = 'Charge Type'
        charges_full.reset_index(inplace=True)
        
        st.subheader('All Cases')
        st.dataframe(charges_full.set_index(charges_full.columns[0]), use_container_width=True)
        
        
        c = closed_cases['Highest Offense'].value_counts().astype(str)
        p = closed_cases['Highest Offense'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
        charges = pd.concat([c,p], axis=1, keys=['Counts', 'Percentage'])

        charges.index.name = 'Charge Type'
        charges.reset_index(inplace=True)
        
        st.subheader('Closed Cases')
        st.dataframe(charges.set_index(charges.columns[0]), use_container_width=True)
        
        
        
        st.header("Domestic Violence")
        
        
        # Can't find perfect way to get all domestic violence charges, but searched for any charge containing "dom"
        dom_charges = closed_cases[
            (closed_cases['[#1] Charges'].str.contains("(?i)dom", na=False)) |
            (closed_cases['[#1] Charges'].str.contains("(?i)dv", na=False)) |
            (closed_cases['[#2] Charges'].str.contains("(?i)dom", na=False)) |
            (closed_cases['[#2] Charges'].str.contains("(?i)dv", na=False)) |
            (closed_cases['[#3] Charges'].str.contains("(?i)dom", na=False)) |
            (closed_cases['[#3] Charges'].str.contains("(?i)dv", na=False)) |
            (closed_cases['[#4] Charges'].str.contains("(?i)dom", na=False)) |
            (closed_cases['[#4] Charges'].str.contains("(?i)dv", na=False))
        ]
        
        dom_charges['Bail Disp Custom'] = dom_charges.apply(lambda row: summary_bail_outcome(row), axis=1)
        
        st.write('There are a total of ', str(len(dom_charges)), ' domestic violence charges out of ', str(len(closed_cases)), ' total closed cases.', 'This is ', str(round(len(dom_charges)/len(closed_cases)*100,2)) + '% of all closed cases.')
        st.write("Domestic violence cases were found by filtering for any charges that included 'dom' or 'dv' (case insensitive)")
        
        st.subheader('Status of Domestic Violence Cases')
        c = dom_charges['Bail Disp Custom'].value_counts().astype(str)
        p = dom_charges['Bail Disp Custom'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
        bail_sum = dom_charges.groupby(by='Bail Disp Custom')['Total Bail Amount'].sum().apply(lambda x: "${:,.0f}".format(x))
        dv = pd.concat([c,p, bail_sum], axis=1, keys=['Count', 'Percentage', 'Total Bail Amount'])
        dv.index.name = 'Disposition'
        dv.reset_index(inplace=True)
        
        st.table(dv.set_index(dv.columns[0]))
        #st.dataframe(dv.set_index(dv.columns[0]), use_container_width=True)
        
        
        percent_complete += 10
        my_bar.progress(percent_complete, text=progress_text) ############################################
        
        
        
        
        
        
        
        
        
        st.header("FTA's")
        
        ftas = posted_dataset[posted_dataset['Bail Status'] == 'FTA']
        ftas = ftas[['__document_id', 'First Name', 'Last Name', 'Age', 'Gender', 'Race', 'Request Status', 'Bail Status', 'Total Bail Amount', 'Arrest Date', '[#1] Case Number', '[#1] Charges', '[#1] Charge Type']]
        
        st.write('There are a total of ', str(len(ftas)), " FTA's out of ", str(len(posted_dataset)), ' total posted bails (' + str(round(len(ftas)/len(posted_dataset)*100,2)) + '%).', 'Total bail posted for these cases is ', "${:,.0f}".format(sum(ftas['Total Bail Amount'])))
        
        st.dataframe(ftas.set_index(ftas.columns[0]), use_container_width=True)
        
        
        
        
        
        
        
        
        
        st.header("Final Forfeits")
        forfeits = closed_cases[closed_cases['Bail Status'] == 'Final Forfeit']
        st.write('There are a total of ', str(len(forfeits)), " Final Forfeit's out of ", str(len(closed_cases)), ' total closed cases (' + str(round(len(forfeits)/len(closed_cases)*100,2)) + '%).', 'Total bail posted for these cases is ', "${:,.0f}".format(sum(forfeits['Total Bail Amount'])))
        forfeits = forfeits[['__document_id', 'First Name', 'Last Name', 'Age', 'Gender', 'Race', 'Request Status', 'Bail Status', 'Total Bail Amount', 'Arrest Date', '[#1] Case Number', '[#1] Charges', '[#1] Charge Type']]
        
        st.dataframe(forfeits.set_index(forfeits.columns[0]), use_container_width=True)
        
              
            
            
            
            
            
    with tab2:
                
        st.header('Distribution of Participant Ages')
        
        age_filtered = data[data['Age'] > 15]

        age_completed = age_filtered[age_filtered['Bail Request Status'] == 'Completed']['Age']
        age_notmade = age_filtered[age_filtered['Bail Request Status'] == 'Bail Not Made']['Age']

        hist_data = [age_completed, age_notmade]

        group_labels = ['Completed', 'Bail Not Made']
        
        

        # Compute KDEs for the data
        kde1 = gaussian_kde(age_completed)
        kde2 = gaussian_kde(age_notmade)

        # Generate x-values for the plots
        x = np.linspace(min(age_completed.min(), age_notmade.min()), max(age_completed.max(), age_notmade.max()), 100)

        # Compute KDE values for the x-values
        y1 = kde1(x)
        y2 = kde2(x)

        # Create a dataframe with the x-values and KDE values
        df1 = pd.DataFrame({'Age': x, 'Density': y1, 'Bail Request Status': 'Completed'})
        df2 = pd.DataFrame({'Age': x, 'Density': y2, 'Bail Request Status': 'Bail Not Made'})
        df = pd.concat([df1, df2])
        
        color_scale = alt.Scale(domain=['Completed', 'Bail Not Made'], range=['#142c5b', '#9B59B6'])

        # Create the KDE line plot
        line_plot = alt.Chart(df).mark_line().encode(
            x='Age',
            y='Density',
            color=alt.Color('Bail Request Status', scale=color_scale)
        )

        # Create the shaded area plot
        area_plot = alt.Chart(df).mark_area(opacity=0.25).encode(
            x='Age',
            y='Density',
            color=alt.Color('Bail Request Status', scale=color_scale)
        )

        # Combine the line and area plots
        plot = line_plot + area_plot

        # Display the plot using Streamlit
        st.altair_chart(plot, use_container_width=True)
        
        st.write('Summary Statistics:')
        
        stats = age_filtered.groupby('Bail Request Status')['Age'].agg(['min', 'mean', 'max'])

        stats['mean'] = stats['mean'].round(2)
        stats = stats.astype(str)

        stats.columns = ['Minimum Age', 'Mean Age', 'Maximum Age']

        # Display the DataFrame
        stats_df = pd.DataFrame(stats).reset_index()
        
        st.dataframe(stats_df.set_index(stats_df.columns[0]), use_container_width=True)
        
        
        
        percent_complete += 10
        my_bar.progress(percent_complete, text=progress_text) ############################################
        
        
        
        
        st.header('Distribution of Participant Genders')
        
        # Calculate percentages
        percentages = data.groupby(['Gender', 'Bail Request Status']).size().reset_index(name='Count')
        percentages_col = percentages.groupby('Bail Request Status', as_index=False)['Count'].apply(lambda x: x / x.sum() * 100)
        percentages['Percentage'] = percentages_col.reset_index(level=0, drop=True)
        
        fig = go.Figure()
        
        colors = {'Completed': '#142c5b', 'Bail Not Made': '#9B59B6'}
        
        # Add bars for each gender
        for status in ['Completed', 'Bail Not Made']:
            status_data = percentages[percentages['Bail Request Status'] == status]
            fig.add_trace(go.Bar(
                x=status_data['Gender'],
                y=status_data['Percentage'],
                name=status,
                marker=dict(color=colors[status])
            ))

        # Update layout
        fig.update_layout(
            barmode='group',
            xaxis_title='Gender',
            yaxis_title='Percentage',
            title='Distribution of Gender by Percentage'
        )
        
        fig.update_layout(xaxis = {"categoryorder":"total descending"})

        # Display the Plotly figure using Streamlit
        st.plotly_chart(fig, use_container_width=True)
        

        
        fig = go.Figure()
        
        colors = {'Completed': '#142c5b', 'Bail Not Made': '#9B59B6'}
        
        # Add bars for each gender
        for status in ['Completed', 'Bail Not Made']:
            status_data = percentages[percentages['Bail Request Status'] == status]
            fig.add_trace(go.Bar(
                x=status_data['Gender'],
                y=status_data['Count'],
                name=status,
                marker=dict(color=colors[status])
            ))

        # Update layout
        fig.update_layout(
            barmode='group',
            xaxis_title='Gender',
            yaxis_title='Count',
            title='Distribution of Gender by Count'
        )
        
        fig.update_layout(xaxis = {"categoryorder":"total descending"})

        # Display the Plotly figure using Streamlit
        st.plotly_chart(fig, use_container_width=True)
        

        
        
        
        
        
    
        
        percent_complete += 10
        my_bar.progress(percent_complete, text=progress_text) ############################################
        
        
        st.header('Distribution of Participant Races')
        
        # Calculate percentages
        percentages = data.groupby(['Race', 'Bail Request Status']).size().reset_index(name='Count')
        percentages_col = percentages.groupby('Bail Request Status', as_index=False)['Count'].apply(lambda x: x / x.sum() * 100)
        percentages['Percentage'] = percentages_col.reset_index(level=0, drop=True)

        fig = go.Figure()
        
        colors = {'Completed': '#142c5b', 'Bail Not Made': '#9B59B6'}
        
        # Add bars for each gender
        for status in ['Completed', 'Bail Not Made']:
            status_data = percentages[percentages['Bail Request Status'] == status]
            fig.add_trace(go.Bar(
                x=status_data['Race'],
                y=status_data['Percentage'],
                name=status,
                marker=dict(color=colors[status])
            ))

        # Update layout
        fig.update_layout(
            barmode='group',
            xaxis_title='Race',
            yaxis_title='Percentage',
            title='Distribution of Race by Percentage'
        )
        
        fig.update_layout(xaxis = {"categoryorder":"total descending"})

        # Display the Plotly figure using Streamlit
        st.plotly_chart(fig, use_container_width=True)
        
        
        
        
        
        
        fig = go.Figure()
        
        colors = {'Completed': '#142c5b', 'Bail Not Made': '#9B59B6'}
        
        # Add bars for each gender
        for status in ['Completed', 'Bail Not Made']:
            status_data = percentages[percentages['Bail Request Status'] == status]
            fig.add_trace(go.Bar(
                x=status_data['Race'],
                y=status_data['Count'],
                name=status,
                marker=dict(color=colors[status])
            ))

        # Update layout
        fig.update_layout(
            barmode='group',
            xaxis_title='Race',
            yaxis_title='Count',
            title='Distribution of Race by Count'
        )
        
        fig.update_layout(xaxis = {"categoryorder":"total descending"})

        # Display the Plotly figure using Streamlit
        st.plotly_chart(fig, use_container_width=True)
        
        

        percent_complete += 10
        my_bar.progress(percent_complete, text=progress_text) ############################################
        my_bar.empty()